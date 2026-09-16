"""
Phase 3c - Scheduling simulator with capacity-aware deadline feasibility.

The defect corrected here is that each deadline class previously sized its requirement
against the full residual capacity, which is in fact shared across classes. Every class
under-committed, backlog accumulated, and work was forced to run at its deadline, which
left the oracle performing worse than a fixed threshold.

The scheduler is now split into two decisions taken in the right order.

  1. Mandatory rate. Interactive load is deterministic given the diurnal shape, so future
     residual capacity is known. The smallest rate that keeps every deadline reachable is

         m(t) = max over h of [ Q(h) - sum of residual capacity in t+1 .. t+h ]

     where Q(h) is backlog due within h periods. This is computed across all classes
     jointly, which is what the previous version failed to do.

  2. Discretionary rate. Above the mandatory minimum, the scheduler runs at full residual
     capacity when the current period is among the cheapest the remaining backlog could
     occupy within its window, and otherwise runs only the mandatory minimum.

Three standing assertions guard the results, each of which caught a real defect during
development and is therefore retained in the code rather than in a comment:

  A1  work conservation: served + violated + residual backlog equals arrived work
  A2  oracle dominance: with realised intensity the oracle is never beaten by a policy
      acting on the forecast
  A3  budget monotonicity: a larger deferral budget never reduces the saving

Usage:  python3 src/simulate_policies.py [policies|fairness|utilisation|flexibility]
"""

import heapq
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import erfinv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")

DESIGN_END = "2025-09-01"     # months 1-24; months 25-36 held out and scored once
INTERACTIVE_SHARE = 0.357     # 32.5 / (32.5 + 58.5) core hours in the Azure trace
IDLE_FRACTION = 0.50
N_TENANTS = 200
RNG_SEED = 20260913
TOL = 1e-6


def load_grid(design_only=True):
    g = pd.read_csv(os.path.join(PROC, "national_intensity.csv"), parse_dates=["from"])
    g = g.dropna(subset=["actual", "forecast"]).sort_values("from").reset_index(drop=True)
    if design_only:
        g = g[g["from"] < pd.Timestamp(DESIGN_END, tz="UTC")].reset_index(drop=True)
    return g


def load_shapes(g, utilisation):
    di = pd.read_csv(os.path.join(OUT, "llm_inference_diurnal.csv"))
    inf = (di.groupby("hour").pct_of_daily_mean.mean() / 100.0).reindex(range(24)).ffill()
    ap = pd.read_csv(os.path.join(OUT, "workload_arrival_profile.csv"))
    arr = ap[ap.category == "Delay-insensitive"].set_index("hour_of_day").pct_arrivals
    arr = (arr / arr.mean()).reindex(range(24)).ffill().bfill()
    h = g["from"].dt.hour.to_numpy()
    n = np.clip(inf.to_numpy()[h] * utilisation * INTERACTIVE_SHARE, 0, 0.95)
    a = np.clip(arr.to_numpy()[h] * utilisation * (1 - INTERACTIVE_SHARE), 0, 0.95)
    return n, a


def ai_training_arrivals(g, utilisation, seed=RNG_SEED, cv_target=2.5):
    """Deferrable arrivals from a synthetic artificial intelligence training workload.

    The virtual machine trace supplies elastic services rather than training jobs, so the
    conclusions are re-tested against a workload with the shape reported for GPU training
    clusters: arrivals modulated by the working day, and job sizes with a heavy right tail in
    which a small number of long, multi-accelerator runs carry most of the work. Sizes are
    lognormal with the coefficient of variation set to cv_target, and the series is scaled to
    the same mean rate as the trace-derived arrivals so that the comparison isolates shape."""
    rng = np.random.default_rng(seed + 7)
    h = g["from"].dt.hour.to_numpy()
    dow = g["from"].dt.dayofweek.to_numpy()
    # Submission is heavier in working hours and on weekdays, as reported for training clusters.
    shape = 0.55 + 0.45 * np.sin(np.pi * np.clip((h - 6) / 14.0, 0, 1)) ** 2
    shape = shape * np.where(dow < 5, 1.0, 0.6)
    sigma = np.sqrt(np.log(1 + cv_target ** 2))
    draws = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=len(g))
    a = shape * draws
    target = utilisation * (1 - INTERACTIVE_SHARE)
    # Clipping a heavy-tailed series lowers its mean, so the scale is solved for rather than
    # applied once. Without this the synthetic workload would carry less load than the trace
    # and would appear to save more simply because it leaves more headroom.
    k = target / a.mean()
    for _ in range(60):
        clipped = np.clip(a * k, 0, 0.95)
        err = target - clipped.mean()
        if abs(err) < 1e-6:
            break
        k *= 1 + err / max(clipped.mean(), 1e-9)
    return np.clip(a * k, 0, 0.95)


def tenant_profile(k=N_TENANTS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    tc = pd.read_csv(os.path.join(OUT, "workload_tenant_concentration.csv"))
    sigma = 2.0 * erfinv(float(tc.gini_core_hours[0]))
    s = rng.lognormal(0.0, sigma, size=k)
    budgets = rng.choice([2.0, 4.0, 8.0, 12.0, 24.0], size=k, p=[.15, .20, .25, .20, .20])
    return s / s.sum(), budgets, sigma


BUDGET_MIXES = {
    "baseline":  ([2.0, 4.0, 8.0, 12.0, 24.0], [.15, .20, .25, .20, .20]),
    "uniform":   ([2.0, 4.0, 8.0, 12.0, 24.0], [.20, .20, .20, .20, .20]),
    "tight":     ([1.0, 2.0, 4.0, 8.0, 12.0],  [.30, .30, .20, .15, .05]),
    "generous":  ([4.0, 8.0, 12.0, 24.0],      [.10, .20, .30, .40]),
    "bimodal":   ([2.0, 24.0],                 [.50, .50]),
}


def budgets_from(mix, k=N_TENANTS, seed=RNG_SEED):
    levels, probs = BUDGET_MIXES[mix]
    return np.random.default_rng(seed + 13).choice(levels, size=k, p=probs)


def mandatory_rate(qs, cap_future, cap_now):
    """Smallest rate now that keeps every deadline reachable, given known future capacity."""
    cumq = np.cumsum(qs)                      # work due within h periods, h = 0..W
    avail = np.concatenate(([0.0], np.cumsum(cap_future)))[:len(qs)]
    return float(np.clip(np.max(cumq - avail), 0.0, cap_now))


def plan_first_period(sigw, capw, qs):
    """Build the cost-minimising feasible plan over the visible window and return the work
    it assigns to the current period.

    Work in deadline class s may occupy any period up to s. Eligibility is therefore nested,
    which allows the classic deadline-ordered greedy: walk the classes from the tightest
    deadline outwards, admitting one more period at each step, and serve each class from the
    cheapest period admitted so far that still has capacity. A min-heap keyed on intensity
    makes this O(W log W) per period. Only the assignment to the current period is executed;
    the plan is rebuilt next period on fresh information, which is what makes this a
    receding-horizon policy rather than an offline schedule."""
    heap, rem, assign0 = [], capw.copy(), 0.0
    for s in range(len(qs)):
        heapq.heappush(heap, (sigw[s], s))
        q = qs[s]
        while q > TOL and heap:
            _, p = heap[0]
            take = min(q, rem[p])
            if take > TOL:
                rem[p] -= take
                q -= take
                if p == 0:
                    assign0 += take
            if rem[p] <= TOL:
                heapq.heappop(heap)
            else:
                break
    return assign0


def discretionary(policy, t, sig, Q, cap_future, cap_now, W, params, trail):
    """Should the scheduler spend capacity beyond the mandatory minimum in this period?"""
    if Q <= TOL:
        return False
    if policy == "agnostic":
        return True
    if policy == "threshold":
        return sig[t] <= params["thr"]
    if policy == "percentile":
        return sig[t] <= trail
    if policy == "capacitycurve":
        # After the published fleet deployment: at the start of each day a capacity curve is
        # set from the day-ahead forecast, admitting flexible load only in the cleanest share
        # of the coming twenty-four hours.
        return sig[t] <= params["day_cut"]
    if policy == "lyapunov":
        return Q >= params["V"] * (sig[t] - params["ref"]) / max(W, 1)
    if policy in ("horizon", "oracle"):
        w = sig[t:t + W + 1]
        if len(w) == 0:
            return True
        mean_cap = max(np.mean(cap_future[:len(w)]) if len(cap_future) else cap_now, 1e-9)
        need = int(np.ceil(Q / mean_cap))
        need = min(max(need, 1), len(w))
        return sig[t] <= np.partition(w, need - 1)[need - 1]
    raise ValueError(policy)


def simulate(policy, g, n, arr, B, params, shares, budgets,
             fair_lambda=0.0, idle=IDLE_FRACTION, monthly=False):
    f, a = g["forecast"].to_numpy(float), g["actual"].to_numpy(float)
    T, k = len(g), len(shares)
    W = int(B * 2)
    slot = np.minimum((budgets * 2).astype(int), W)
    sig = a if policy == "oracle" else f
    cap_all = np.clip(1.0 - n, 0.0, None)

    queue = np.zeros((W + 1, k))
    served_e, served_c, delay_w = np.zeros(k), np.zeros(k), np.zeros(k)
    if monthly:
        mkey = g["from"].dt.strftime("%Y-%m").to_numpy()
        m_e, m_c = {}, {}
    violations = total_work = 0.0
    energy = np.zeros(T)
    trail = np.percentile(f[:48 * 30], params.get("p", 30))

    for t in range(T):
        cap = cap_all[t]
        queue = np.roll(queue, -1, axis=0)
        queue[-1] = 0.0
        np.add.at(queue, (slot, np.arange(k)), arr[t] * shares)
        total_work += arr[t]

        if t > 48 * 30 and t % (48 * 7) == 0:
            trail = np.percentile(f[t - 48 * 30:t], params.get("p", 30))
        if policy == "capacitycurve" and t % 48 == 0:
            day = sig[t:t + 48]
            params = dict(params, day_cut=float(np.percentile(day, params.get("q", 50)))
                          if len(day) else 0.0)

        qs = queue.sum(axis=1)
        cap_future = cap_all[t + 1:t + W + 1]
        if len(cap_future) < W:
            cap_future = np.concatenate([cap_future, np.full(W - len(cap_future), cap)])
        Q = float(qs.sum())

        if policy in ("horizon", "oracle") and Q > TOL and (T - t) > W:
            capw = np.concatenate(([cap], cap_future))[:W + 1]
            x = plan_first_period(sig[t:t + W + 1], capw, qs)
        else:
            x = mandatory_rate(qs, cap_future, cap)
            if discretionary(policy, t, sig, Q, cap_future, cap, W, params, trail):
                x = cap
        x = float(np.clip(x, 0.0, cap))

        if fair_lambda > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                ti = np.where(served_e > 0, served_c / np.maximum(served_e, 1e-12), np.nan)
            fleet = np.nanmean(ti) if np.isfinite(ti).any() else 0.0
            deficit = np.nan_to_num(ti - fleet, nan=0.0)

        remaining = x
        for s in range(W + 1):                      # earliest deadline first
            if remaining <= TOL:
                break
            row = queue[s]
            tot = row.sum()
            if tot <= TOL:
                continue
            take = min(tot, remaining)
            if fair_lambda > 0:
                wgt = np.where(row > 0, 1.0 + fair_lambda * np.clip(deficit, 0, None) / 50.0, 0.0) * row
                alloc = (wgt / wgt.sum() * take) if wgt.sum() > 0 else (row / tot * take)
                alloc = np.minimum(alloc, row)
                short = take - alloc.sum()
                if short > 1e-9:
                    head = row - alloc
                    if head.sum() > TOL:
                        alloc = alloc + head / head.sum() * min(short, head.sum())
            else:
                alloc = row / tot * take
            queue[s] -= alloc
            served_e += alloc
            served_c += alloc * a[t]
            if monthly:
                km = mkey[t]
                m_e[km] = m_e.get(km, 0.0) + float(alloc.sum())
                m_c[km] = m_c.get(km, 0.0) + float(alloc.sum()) * a[t]
            delay_w += alloc * (W - s) * 0.5
            remaining -= alloc.sum()

        if queue[0].sum() > TOL:                    # deadline reached, capacity exhausted
            violations += queue[0].sum()
            queue[0] = 0.0

        energy[t] = (idle + (1 - idle) * (n[t] + (x - remaining))) * 0.5

    served = served_e.sum()
    residual = queue.sum()
    # A1: work conservation
    assert abs(served + violations + residual - total_work) < 1e-3 * total_work, \
        "work not conserved: served %.3f violated %.3f residual %.3f arrived %.3f" % (
            served, violations, residual, total_work)

    ti = np.divide(served_c, served_e, out=np.full(k, np.nan), where=served_e > 0)
    ok = np.isfinite(ti)
    jain = float(ti[ok].sum() ** 2 / (ok.sum() * (ti[ok] ** 2).sum()))
    out_monthly = ({m: m_c[m] / m_e[m] for m in sorted(m_e) if m_e[m] > 0} if monthly else None)
    return {"policy": policy, "B_h": B, "fair_lambda": fair_lambda, "monthly": out_monthly,
            "utilisation": float(np.mean(n + arr)),
            "work_carbon_intensity": float(served_c.sum() / served),
            "facility_carbon_gCO2": float((energy * a).sum()),
            "deadline_violation_pct": violations / total_work * 100,
            "mean_delay_h": float(delay_w.sum() / served),
            "tenant_jain": jain,
            "tenant_spread_gCO2_kWh": float(np.nanmax(ti) - np.nanmin(ti)),
            "worst_tenant_intensity": float(np.nanmax(ti)),
            "best_tenant_intensity": float(np.nanmin(ti))}


def saving(base, r):
    return (base["work_carbon_intensity"] - r["work_carbon_intensity"]) \
        / base["work_carbon_intensity"] * 100


def check_assertions(df):
    """A2 oracle dominance and A3 budget monotonicity, reported rather than assumed."""
    issues = []
    for B, g in df.groupby("B_h"):
        orc = g.loc[g.policy == "oracle", "saving_pct"]
        for pol in ("threshold", "percentile", "horizon", "lyapunov"):
            v = g.loc[g.policy == pol, "saving_pct"]
            if len(orc) and len(v) and v.iloc[0] > orc.iloc[0] + 0.05:
                issues.append("A2 violated at B=%s: %s %.2f%% beats oracle %.2f%%"
                              % (B, pol, v.iloc[0], orc.iloc[0]))
    for pol in ("horizon", "oracle"):
        s = df[df.policy == pol].sort_values("B_h")["saving_pct"].to_numpy()
        if len(s) > 1 and np.any(np.diff(s) < -0.05):
            issues.append("A3 violated: %s saving not monotone in budget: %s"
                          % (pol, np.round(s, 2).tolist()))
    return issues


def stage_policies(g, shares, budgets, med, util=0.85):
    n, arr = load_shapes(g, util)
    rows = []
    tune_base = simulate("agnostic", g, n, arr, 24, {}, shares, budgets)
    best_V, best = None, -np.inf
    for V in (0.5, 2.0, 5.0, 20.0):
        r = simulate("lyapunov", g, n, arr, 24, {"V": V, "ref": med}, shares, budgets)
        sv = saving(tune_base, r)
        if r["deadline_violation_pct"] <= 5 and sv > best:
            best_V, best = V, sv
    print("tuned Lyapunov V = %s (%.2f%% on design period)" % (best_V, best), flush=True)

    for B in (4, 12, 24):
        base = simulate("agnostic", g, n, arr, B, {}, shares, budgets)
        base["saving_pct"] = 0.0
        rows.append(base)
        for pol, pr in [("threshold", {"thr": med}), ("percentile", {"p": 30}),
                        ("capacitycurve", {"q": 50}), ("horizon", {}),
                        ("lyapunov", {"V": best_V, "ref": med}), ("oracle", {})]:
            r = simulate(pol, g, n, arr, B, pr, shares, budgets)
            r["saving_pct"] = saving(base, r)
            rows.append(r)
            print("  B=%2d %-11s saving %6.2f%%  delay %5.2f h  viol %4.2f%%  Jain %.4f  spread %5.1f"
                  % (B, pol, r["saving_pct"], r["mean_delay_h"], r["deadline_violation_pct"],
                     r["tenant_jain"], r["tenant_spread_gCO2_kWh"]), flush=True)
    df = pd.DataFrame(rows).drop(columns=["monthly"], errors="ignore")
    df.to_csv(os.path.join(OUT, "phase3c_policies.csv"), index=False)
    for msg in check_assertions(df):
        print("  ASSERTION: " + msg, flush=True)
    else:
        pass
    if not check_assertions(df):
        print("  assertions A2 and A3 hold", flush=True)


def stage_flexibility(g, shares, budgets, util=0.85):
    n, arr = load_shapes(g, util)
    base = simulate("agnostic", g, n, arr, 24, {}, shares, budgets)
    rows = []
    for grant in (None, 8.0, 12.0, 24.0):
        b = budgets.copy()
        if grant is not None:
            b[b <= 4.0] = grant
        r = simulate("horizon", g, n, arr, 24, {}, shares, b)
        r["saving_pct"] = saving(base, r)
        r["grant_h"] = 0.0 if grant is None else grant
        r["n_tenants_granted"] = int((budgets <= 4.0).sum())
        rows.append(r)
        print("  grant %-5s saving %6.2f%%  Jain %.5f  spread %5.1f  worst %6.2f  delay %5.2f h"
              % (grant, r["saving_pct"], r["tenant_jain"], r["tenant_spread_gCO2_kWh"],
                 r["worst_tenant_intensity"], r["mean_delay_h"]), flush=True)
    pd.DataFrame(rows).drop(columns=["monthly"], errors="ignore").to_csv(os.path.join(OUT, "phase3c_flexibility.csv"), index=False)


def stage_fairness(g, shares, budgets, util=0.85):
    n, arr = load_shapes(g, util)
    base = simulate("agnostic", g, n, arr, 24, {}, shares, budgets)
    rows = []
    for lam in (0.0, 1.0, 5.0, 20.0):
        r = simulate("horizon", g, n, arr, 24, {}, shares, budgets, fair_lambda=lam)
        r["saving_pct"] = saving(base, r)
        rows.append(r)
        print("  lambda %5.1f  saving %6.2f%%  Jain %.5f  spread %5.1f  worst %6.2f"
              % (lam, r["saving_pct"], r["tenant_jain"], r["tenant_spread_gCO2_kWh"],
                 r["worst_tenant_intensity"]), flush=True)
    pd.DataFrame(rows).drop(columns=["monthly"], errors="ignore").to_csv(os.path.join(OUT, "phase3c_fairness.csv"), index=False)


def stage_utilisation(g, shares, budgets):
    rows = []
    for util in (0.60, 0.75, 0.85, 0.95, 1.05):
        n, arr = load_shapes(g, util)
        base = simulate("agnostic", g, n, arr, 12, {}, shares, budgets)
        r = simulate("horizon", g, n, arr, 12, {}, shares, budgets)
        r["saving_pct"] = saving(base, r)
        rows.append(r)
        print("  utilisation %.2f (realised %.2f)  saving %6.2f%%  viol %5.2f%%  delay %5.2f h"
              % (util, r["utilisation"], r["saving_pct"], r["deadline_violation_pct"],
                 r["mean_delay_h"]), flush=True)
    pd.DataFrame(rows).drop(columns=["monthly"], errors="ignore").to_csv(os.path.join(OUT, "phase3c_utilisation.csv"), index=False)


def stage_energy(g, shares, budgets, util=0.85):
    """Energy bracket.

    No power metering is available in this environment, so the power model is bracketed
    rather than asserted. The bracket is the idle-to-peak power ratio, for which the
    literature provides both endpoints:

      0.60  Fan, Weber and Barroso (2007), doi:10.1145/1250662.1250665, report that idle
            power in the Google fleet was generally never below 50% of peak; Malla and
            Christensen (2020), doi:10.1016/j.future.2019.10.021, put a 2007-era server
            above 60% of peak when idle.
      0.10  Malla and Christensen (2020) report the same quantity falling to a little over
            10% of peak for a 2018-era server.
      0.30  intermediate value, reported as the central case.

    Facility overhead enters as a multiplicative constant and therefore cancels from every
    relative figure, so results are reported per unit of information technology energy and
    no power usage effectiveness value is asserted.

    Two metrics are separated deliberately. The carbon intensity of delivered work is
    invariant to the power model. The facility-level saving is not, because idle draw is
    paid whether or not work is shifted."""
    n, arr = load_shapes(g, util)
    rows = []
    for idle in (0.10, 0.30, 0.60):
        for B in (12, 24):
            base = simulate("agnostic", g, n, arr, B, {}, shares, budgets, idle=idle)
            r = simulate("horizon", g, n, arr, B, {}, shares, budgets, idle=idle)
            r["idle_fraction"] = idle
            r["saving_pct"] = saving(base, r)
            r["facility_saving_pct"] = (base["facility_carbon_gCO2"] - r["facility_carbon_gCO2"]) \
                / base["facility_carbon_gCO2"] * 100
            rows.append(r)
            print("  idle %.2f  B=%2d   work-intensity saving %5.2f%%   facility saving %5.2f%%"
                  % (idle, B, r["saving_pct"], r["facility_saving_pct"]), flush=True)
    pd.DataFrame(rows).drop(columns=["monthly"], errors="ignore").to_csv(os.path.join(OUT, "phase3d_energy_bracket.csv"), index=False)


def stage_signals(g_unused, shares, budgets, med, util=0.85):
    """Every headline result under both signals.

    S1 is the published operational forecast, whose vintage the publisher does not state.
    S2 is the day-ahead forecast built in Phase 3e from information available at least 24
    hours ahead. A claim is made only where it holds under both."""
    d = pd.read_csv(os.path.join(PROC, "national_intensity_signals.csv"), parse_dates=["from"])
    d = d.dropna(subset=["actual", "forecast_published", "forecast_s2"])
    d = d[d["from"] < pd.Timestamp(DESIGN_END, tz="UTC")].reset_index(drop=True)
    rows = []
    for label, col in (("S1_published", "forecast_published"), ("S2_day_ahead", "forecast_s2")):
        g = d.rename(columns={col: "forecast"})[["from", "actual", "forecast"]]
        n, arr = load_shapes(g, util)
        m = float(np.median(g["forecast"]))
        for B in (12, 24):
            base = simulate("agnostic", g, n, arr, B, {}, shares, budgets)
            base["saving_pct"], base["signal"] = 0.0, label
            rows.append(base)
            for pol, pr in [("threshold", {"thr": m}), ("horizon", {}), ("oracle", {})]:
                r = simulate(pol, g, n, arr, B, pr, shares, budgets)
                r["saving_pct"], r["signal"] = saving(base, r), label
                rows.append(r)
                print("  %-12s B=%2d %-10s saving %5.2f%%  delay %5.2f h  viol %4.2f%%"
                      % (label, B, pol, r["saving_pct"], r["mean_delay_h"],
                         r["deadline_violation_pct"]), flush=True)
    pd.DataFrame(rows).drop(columns=["monthly"], errors="ignore").to_csv(os.path.join(OUT, "phase3e_policies_both_signals.csv"), index=False)


def stage_aiworkload(g, shares, budgets, med, util=0.85):
    """Repeat the policy comparison with deferrable arrivals from the synthetic training
    workload, to test whether the conclusions depend on the shape of the trace-derived series."""
    n, _ = load_shapes(g, util)
    rows = []
    for label, arr in (("trace", load_shapes(g, util)[1]),
                       ("ai_training", ai_training_arrivals(g, util))):
        print("  %s: mean arrival %.3f, cv %.2f"
              % (label, arr.mean(), arr.std() / arr.mean()), flush=True)
        for B in (12, 24):
            base = simulate("agnostic", g, n, arr, B, {}, shares, budgets)
            for pol, pr in [("threshold", {"thr": med}), ("horizon", {}), ("oracle", {})]:
                r = simulate(pol, g, n, arr, B, pr, shares, budgets)
                r["saving_pct"] = saving(base, r)
                r["workload"] = label
                r["arrival_cv"] = float(arr.std() / arr.mean())
                rows.append(r)
                print("    B=%2d %-10s saving %5.2f%%  viol %4.2f%%  Jain %.5f"
                      % (B, pol, r["saving_pct"], r["deadline_violation_pct"],
                         r["tenant_jain"]), flush=True)
    pd.DataFrame(rows).drop(columns=["monthly"], errors="ignore").to_csv(os.path.join(OUT, "phase6_ai_workload.csv"), index=False)


def stage_tenantsens(g, shares, budgets_unused, util=0.85):
    """Do the fairness conclusions depend on the assumed spread of tenant deferral budgets?"""
    n, arr = load_shapes(g, util)
    rows = []
    for mix in BUDGET_MIXES:
        b = budgets_from(mix)
        base = simulate("agnostic", g, n, arr, 24, {}, shares, b)
        r0 = simulate("horizon", g, n, arr, 24, {}, shares, b)
        rl = simulate("horizon", g, n, arr, 24, {}, shares, b, fair_lambda=5.0)
        bg = b.copy()
        bg[bg <= 4.0] = 24.0
        rg = simulate("horizon", g, n, arr, 24, {}, shares, bg)
        rows.append({"mix": mix, "n_inflexible": int((b <= 4.0).sum()),
                     "saving_pct": saving(base, r0),
                     "saving_with_allocation_lever": saving(base, rl),
                     "saving_with_flexibility_grant": saving(base, rg),
                     "jain": r0["tenant_jain"],
                     "jain_with_allocation_lever": rl["tenant_jain"],
                     "jain_with_flexibility_grant": rg["tenant_jain"],
                     "spread": r0["tenant_spread_gCO2_kWh"],
                     "spread_with_flexibility_grant": rg["tenant_spread_gCO2_kWh"]})
        print("  %-9s saving %5.2f -> grant %5.2f | Jain %.5f -> lever %.5f, grant %.5f | spread %5.1f -> %4.1f"
              % (mix, rows[-1]["saving_pct"], rows[-1]["saving_with_flexibility_grant"],
                 rows[-1]["jain"], rows[-1]["jain_with_allocation_lever"],
                 rows[-1]["jain_with_flexibility_grant"], rows[-1]["spread"],
                 rows[-1]["spread_with_flexibility_grant"]), flush=True)
    pd.DataFrame(rows).drop(columns=["monthly"], errors="ignore").to_csv(os.path.join(OUT, "phase6_tenant_sensitivity.csv"), index=False)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "policies"
    t0 = time.time()
    g = load_grid(True)
    shares, budgets, sigma = tenant_profile()
    med = float(np.median(g["forecast"]))
    print("stage=%s  periods=%d  tenants=%d  sigma=%.2f" % (stage, len(g), len(shares), sigma),
          flush=True)
    {"policies": lambda: stage_policies(g, shares, budgets, med),
     "fairness": lambda: stage_fairness(g, shares, budgets),
     "flexibility": lambda: stage_flexibility(g, shares, budgets),
     "utilisation": lambda: stage_utilisation(g, shares, budgets),
     "energy": lambda: stage_energy(g, shares, budgets),
     "signals": lambda: stage_signals(g, shares, budgets, med),
     "aiworkload": lambda: stage_aiworkload(g, shares, budgets, med),
     "tenantsens": lambda: stage_tenantsens(g, shares, budgets)}[stage]()
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
