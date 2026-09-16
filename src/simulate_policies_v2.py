"""
Phase 3b - Scheduling simulator, corrected.

Three defects found in the first implementation (retained as simulate_policies_v1.py) are
addressed here.

1. Per-class scheduling decisions. Deferrable work is held in classes indexed by the slack
   remaining before its deadline. The first implementation set a single threshold from the
   aggregate backlog against the whole window, which made the receding-horizon policy
   non-monotonic in the deferral budget. Each class now decides against its own remaining
   window, so a longer budget can never hurt.

2. Fairness acts on priority, not on share. Capping each tenant's share of served capacity
   left the outcome unchanged, because the inequality arises from deadline ordering rather
   than from allocation share. Clean-period capacity is now allocated in proportion to each
   tenant's accumulated carbon deficit, so tenants that have so far received dirtier service
   are served first when conditions are good.

3. Utilisation is swept. With ample headroom the residual-capacity constraint never binds
   and the two workload classes do not interact. Mean utilisation is now a swept parameter,
   holding the trace-derived split between interactive and deferrable work fixed.

Usage:  python3 src/simulate_policies.py [policies|fairness|utilisation]
"""

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

DESIGN_END = "2025-09-01"     # months 1-24; months 25-36 are held out and scored once
INTERACTIVE_SHARE = 0.357     # 32.5 / (32.5 + 58.5) core hours in the Azure trace
IDLE_FRACTION = 0.50
N_TENANTS = 200
RNG_SEED = 20260913


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
    n = inf.to_numpy()[h] * utilisation * INTERACTIVE_SHARE
    a = arr.to_numpy()[h] * utilisation * (1 - INTERACTIVE_SHARE)
    return np.clip(n, 0, 0.95), np.clip(a, 0, 0.95)


def tenant_profile(k=N_TENANTS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    tc = pd.read_csv(os.path.join(OUT, "workload_tenant_concentration.csv"))
    sigma = 2.0 * erfinv(float(tc.gini_core_hours[0]))
    s = rng.lognormal(0.0, sigma, size=k)
    shares = s / s.sum()
    budgets = rng.choice([2.0, 4.0, 8.0, 12.0, 24.0], size=k, p=[.15, .20, .25, .20, .20])
    return shares, budgets, sigma


def class_decision(policy, t, sig, q_class, slack, cap, params, trail):
    """Should this deadline class run in the current period? Each class is judged against
    the window it has left, so the decision does not depend on other classes' deadlines."""
    if slack == 0:
        return True
    if q_class <= 1e-12:
        return False
    if policy == "agnostic":
        return True
    if policy == "threshold":
        return sig[t] <= params["thr"]
    if policy == "percentile":
        return sig[t] <= trail
    if policy == "lyapunov":
        return q_class >= params["V"] * (sig[t] - params["ref"]) / max(slack, 1)
    if policy in ("horizon", "oracle"):
        w = sig[t:t + slack + 1]
        if len(w) == 0:
            return True
        need = int(np.ceil(q_class / max(cap, 1e-9)))
        need = min(max(need, 1), len(w))
        return sig[t] <= np.partition(w, need - 1)[need - 1]
    raise ValueError(policy)


def simulate(policy, g, n, arr, B, params, shares, budgets,
             fair_lambda=0.0, idle=IDLE_FRACTION):
    f, a = g["forecast"].to_numpy(float), g["actual"].to_numpy(float)
    T, k = len(g), len(shares)
    W = int(B * 2)
    slot = np.minimum((budgets * 2).astype(int), W)
    sig = a if policy == "oracle" else f

    queue = np.zeros((W + 1, k))
    served_e, served_c = np.zeros(k), np.zeros(k)
    delay_w = np.zeros(k)
    violations = total_work = 0.0
    energy = np.zeros(T)
    trail = np.percentile(f[:48 * 30], params.get("p", 30))

    for t in range(T):
        cap = max(1.0 - n[t], 0.0)
        queue = np.roll(queue, -1, axis=0)
        queue[-1] = 0.0
        np.add.at(queue, (slot, np.arange(k)), arr[t] * shares)
        total_work += arr[t]

        if t > 48 * 30 and t % (48 * 7) == 0:
            trail = np.percentile(f[t - 48 * 30:t], params.get("p", 30))

        if fair_lambda > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                ti = np.where(served_e > 0, served_c / np.maximum(served_e, 1e-12), np.nan)
            fleet = np.nanmean(ti) if np.isfinite(ti).any() else 0.0
            deficit = np.nan_to_num(ti - fleet, nan=0.0)
        else:
            deficit = None

        remaining = cap
        for s in range(W + 1):
            if remaining <= 1e-12:
                break
            row = queue[s]
            tot = row.sum()
            if tot <= 1e-12:
                continue
            if not class_decision(policy, t, sig, tot, s, cap, params, trail):
                continue
            take = min(tot, remaining)
            if fair_lambda > 0:
                w = np.where(row > 0, 1.0 + fair_lambda * np.clip(deficit, 0, None) / 50.0, 0.0) * row
                alloc = (w / w.sum() * take) if w.sum() > 0 else (row / tot * take)
                alloc = np.minimum(alloc, row)
                short = take - alloc.sum()
                if short > 1e-9:
                    head = row - alloc
                    if head.sum() > 1e-12:
                        alloc = alloc + head / head.sum() * min(short, head.sum())
            else:
                alloc = row / tot * take
            queue[s] -= alloc
            served_e += alloc
            served_c += alloc * a[t]
            delay_w += alloc * (W - s) * 0.5
            remaining -= alloc.sum()

        if queue[0].sum() > 1e-12:
            violations += queue[0].sum()
            queue[0] = 0.0

        energy[t] = (idle + (1 - idle) * (n[t] + (cap - remaining))) * 0.5

    served = served_e.sum()
    ti = np.divide(served_c, served_e, out=np.full(k, np.nan), where=served_e > 0)
    ok = np.isfinite(ti)
    jain = float(ti[ok].sum() ** 2 / (ok.sum() * (ti[ok] ** 2).sum()))
    return {"policy": policy, "B_h": B, "fair_lambda": fair_lambda,
            "utilisation": float(np.mean(n + arr)),
            "work_carbon_intensity": float(served_c.sum() / served) if served else np.nan,
            "facility_carbon_gCO2": float((energy * a).sum()),
            "deadline_violation_pct": violations / total_work * 100,
            "mean_delay_h": float(delay_w.sum() / served) if served else np.nan,
            "tenant_jain": jain,
            "tenant_spread_gCO2_kWh": float(np.nanmax(ti) - np.nanmin(ti)),
            "worst_tenant_intensity": float(np.nanmax(ti)),
            "best_tenant_intensity": float(np.nanmin(ti))}


def saving(base, r):
    return (base["work_carbon_intensity"] - r["work_carbon_intensity"]) \
        / base["work_carbon_intensity"] * 100


def stage_policies(g, shares, budgets, med, util=0.85):
    n, arr = load_shapes(g, util)
    rows = []
    tune_base = simulate("agnostic", g, n, arr, 24, {}, shares, budgets)
    best_V, best = None, -np.inf
    for V in (0.5, 2.0, 5.0, 10.0):
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
                        ("horizon", {}), ("lyapunov", {"V": best_V, "ref": med}),
                        ("oracle", {})]:
            r = simulate(pol, g, n, arr, B, pr, shares, budgets)
            r["saving_pct"] = saving(base, r)
            rows.append(r)
            print("  B=%2d %-11s saving %6.2f%%  delay %5.2f h  viol %4.2f%%  Jain %.4f  spread %5.1f"
                  % (B, pol, r["saving_pct"], r["mean_delay_h"], r["deadline_violation_pct"],
                     r["tenant_jain"], r["tenant_spread_gCO2_kWh"]), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "phase3b_policies.csv"), index=False)


def stage_fairness(g, shares, budgets, util=0.85):
    n, arr = load_shapes(g, util)
    base = simulate("agnostic", g, n, arr, 24, {}, shares, budgets)
    rows = []
    for lam in (0.0, 1.0, 5.0, 20.0):
        r = simulate("horizon", g, n, arr, 24, {}, shares, budgets, fair_lambda=lam)
        r["saving_pct"] = saving(base, r)
        rows.append(r)
        print("  lambda %5.1f  saving %6.2f%%  Jain %.5f  spread %5.1f  worst %6.2f  best %6.2f"
              % (lam, r["saving_pct"], r["tenant_jain"], r["tenant_spread_gCO2_kWh"],
                 r["worst_tenant_intensity"], r["best_tenant_intensity"]), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "phase3b_fairness.csv"), index=False)


def stage_utilisation(g, shares, budgets):
    rows = []
    for util in (0.60, 0.75, 0.90, 0.95, 1.05, 1.20):
        n, arr = load_shapes(g, util)
        base = simulate("agnostic", g, n, arr, 12, {}, shares, budgets)
        r = simulate("horizon", g, n, arr, 12, {}, shares, budgets)
        r["saving_pct"] = saving(base, r)
        rows.append(r)
        print("  utilisation %.2f  saving %6.2f%%  viol %5.2f%%  delay %5.2f h  Jain %.4f"
              % (util, r["saving_pct"], r["deadline_violation_pct"], r["mean_delay_h"],
                 r["tenant_jain"]), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "phase3b_utilisation.csv"), index=False)


def stage_flexibility(g, shares, budgets, util=0.85):
    """The equity question restated as a flexibility question.

    If tenant-level inequality is structural — driven by how much slack each tenant has
    rather than by how capacity is divided — then it cannot be removed inside the scheduler,
    and the only effective intervention is to grant slack to the tenants that lack it. This
    experiment raises the deferral budget of the least flexible tenants and measures both the
    fleet saving and the equity outcome."""
    n, arr = load_shapes(g, util)
    base = simulate("agnostic", g, n, arr, 24, {}, shares, budgets)
    rows = []
    for grant in (None, 8.0, 12.0, 24.0):
        b = budgets.copy()
        if grant is not None:
            b[b <= 4.0] = grant
        r = simulate("horizon", g, n, arr, 24, {}, shares, b)
        r["saving_pct"] = saving(base, r)
        r["grant_h"] = grant if grant is not None else 0.0
        r["n_tenants_granted"] = int((budgets <= 4.0).sum())
        rows.append(r)
        print("  grant %-5s saving %6.2f%%  Jain %.5f  spread %5.1f  worst %6.2f  delay %5.2f h"
              % (grant, r["saving_pct"], r["tenant_jain"], r["tenant_spread_gCO2_kWh"],
                 r["worst_tenant_intensity"], r["mean_delay_h"]), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "phase3b_flexibility.csv"), index=False)


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
     "utilisation": lambda: stage_utilisation(g, shares, budgets),
     "flexibility": lambda: stage_flexibility(g, shares, budgets)}[stage]()
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
