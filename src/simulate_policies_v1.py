"""
Phase 3 - Scheduling simulator.

A data centre of unit capacity serves two classes in each half-hourly settlement period:

  non-deferrable  n(t)  interactive inference, served on arrival, capacity taken first
  deferrable      queued work with a delivery deadline of B hours and an adjustable rate

The scheduler chooses how much deferrable capacity x(t) to run. Deadlines are hard: work
reaching its deadline is forced to run regardless of intensity, which is what makes the
carbon saving a genuine trade-off rather than an unbounded delay.

Six policies are compared. All decide on the forecast; all are scored on realised intensity.

  agnostic     serve deferrable work on arrival                          (baseline)
  threshold    run when forecast intensity is below a fixed value
  percentile   threshold set at the p-th percentile of trailing forecasts
  horizon      receding-horizon allocation over the forecast window
  lyapunov     drift-plus-penalty control trading backlog against intensity
  oracle       receding horizon on realised intensity                    (upper bound)

Tenant allocation is a separate decision from the total rate. Capacity is shared across
tenants in proportion to backlog (unconstrained) or under a per-tenant cap (fair), which
determines who receives the clean periods and who is left with the dirty ones.

Energy is deliberately reported in relative terms as the primary result, since the
percentage saving is invariant to the absolute power scale. Absolute figures depend on the
idle power fraction, which is varied as a declared sensitivity rather than asserted.
"""

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")

DESIGN_END = "2025-09-01"          # months 1-24 only; months 25-36 are held out
MEAN_DEFERRABLE = 0.45             # mean share of capacity, from the trace core-hour split
MEAN_INTERACTIVE = 0.25
IDLE_FRACTION = 0.50               # declared sensitivity; varied in the sweep
RNG = np.random.default_rng(20260913)


# ----------------------------------------------------------------------------- inputs
def load_grid(design_only=True):
    g = pd.read_csv(os.path.join(PROC, "national_intensity.csv"), parse_dates=["from"])
    g = g.dropna(subset=["actual", "forecast"]).sort_values("from").reset_index(drop=True)
    if design_only:
        g = g[g["from"] < pd.Timestamp(DESIGN_END, tz="UTC")].reset_index(drop=True)
    return g


def load_shapes(g):
    """Diurnal shapes for the two classes, aligned to the grid clock."""
    di = pd.read_csv(os.path.join(OUT, "llm_inference_diurnal.csv"))
    inf = di.groupby("hour").pct_of_daily_mean.mean() / 100.0
    ap = pd.read_csv(os.path.join(OUT, "workload_arrival_profile.csv"))
    arr = ap[ap.category == "Delay-insensitive"].set_index("hour_of_day").pct_arrivals
    arr = (arr / arr.mean()).reindex(range(24)).ffill().bfill()
    h = g["from"].dt.hour.to_numpy()
    n = inf.reindex(range(24)).to_numpy()[h] * MEAN_INTERACTIVE
    a = arr.to_numpy()[h] * MEAN_DEFERRABLE
    return np.clip(n, 0, 0.9), np.clip(a, 0, 0.9)


def tenant_budgets(k, rng=RNG):
    """Tenants differ in how much delay they tolerate. Heterogeneity is what makes the
    fairness question real: a carbon-greedy scheduler can only place work in clean periods
    for tenants whose deadlines permit waiting, so the least flexible tenants are
    systematically served in whatever conditions prevail when their deadline falls due."""
    levels = np.array([2.0, 4.0, 8.0, 12.0, 24.0])
    weights = np.array([0.15, 0.20, 0.25, 0.20, 0.20])
    return rng.choice(levels, size=k, p=weights)


def tenant_shares(k=200):
    """Draw k tenants from the empirical deferrable core-hour distribution."""
    tc = pd.read_csv(os.path.join(OUT, "workload_tenant_concentration.csv"))
    gini = float(tc.gini_core_hours[0])
    # Lognormal calibrated to reproduce the observed Gini: G = erf(sigma/2).
    from scipy.special import erfinv
    sigma = 2.0 * erfinv(gini)
    s = RNG.lognormal(mean=0.0, sigma=sigma, size=k)
    return s / s.sum(), sigma


# ------------------------------------------------------------------------- policies
def decide_rate(policy, t, f, a, backlog, urgent, capacity, params, trail):
    """Deferrable capacity to run in period t. `urgent` must run to meet deadlines."""
    free = max(capacity - urgent, 0.0)
    if free <= 0:
        return urgent
    if policy == "agnostic":
        return min(urgent + backlog, capacity)
    if policy == "threshold":
        return capacity if f[t] <= params["thr"] else urgent
    if policy == "percentile":
        return capacity if f[t] <= trail else urgent
    if policy in ("horizon", "oracle"):
        sig = a if policy == "oracle" else f
        w = sig[t:t + params["W"]]
        # Run at full rate when the current period is among the cleanest in the window
        # that the remaining backlog could occupy.
        need = int(np.ceil(backlog / max(capacity, 1e-9)))
        if need <= 0:
            return urgent
        cut = np.partition(w, min(need, len(w) - 1))[min(need, len(w) - 1)]
        return capacity if sig[t] <= cut else urgent
    if policy == "lyapunov":
        # Drift-plus-penalty: run when queue pressure exceeds the carbon penalty.
        return capacity if backlog >= params["V"] * (f[t] - params["ref"]) else urgent
    raise ValueError(policy)


# ------------------------------------------------------------------------ simulator
def simulate(policy, g, n, arr, B, params, shares, fair_cap=None, idle=IDLE_FRACTION,
             budgets=None):
    f = g["forecast"].to_numpy(float)
    a = g["actual"].to_numpy(float)
    T = len(g)
    W = int(B * 2)
    params = dict(params, W=W)

    k = len(shares)
    if budgets is None:
        budgets = np.full(k, B)
    slot = np.minimum((budgets * 2).astype(int), W)   # each tenant's own slack on arrival
    queue = np.zeros((W + 1, k))          # work by remaining slack, per tenant
    served_energy = np.zeros(k)           # core-hours served, per tenant
    served_carbon = np.zeros(k)           # gCO2 attributed, per tenant
    delay_weighted = np.zeros(k)
    violations = 0.0
    total_work = 0.0
    energy = np.zeros(T)
    trail = np.percentile(f[:max(48 * 30, 1)], params.get("p", 30))

    for t in range(T):
        cap = max(1.0 - n[t], 0.0)                      # residual capacity
        queue = np.roll(queue, -1, axis=0)
        queue[-1] = 0.0
        np.add.at(queue, (slot, np.arange(k)), arr[t] * shares)   # arrivals at own slack
        total_work += arr[t]

        urgent = queue[0].sum()                         # at deadline, must run now
        if urgent > cap:                                # capacity shortfall
            violations += urgent - cap
            urgent = cap
        backlog = queue.sum()

        if t % (48 * 7) == 0 and t > 48 * 30:           # weekly threshold refresh
            trail = np.percentile(f[max(0, t - 48 * 30):t], params.get("p", 30))

        x = decide_rate(policy, t, f, a, backlog, urgent, cap,
                        dict(params, thr=params.get("thr", 0.0), ref=params.get("ref", 0.0)), trail)
        x = float(np.clip(x, urgent, cap))

        # Allocate the served capacity across tenants, oldest work first.
        remaining = x
        for s in range(W + 1):
            if remaining <= 1e-12:
                break
            row = queue[s]
            tot = row.sum()
            if tot <= 1e-12:
                continue
            take = min(tot, remaining)
            if fair_cap is None:
                alloc = row / tot * take                # proportional to backlog
            else:
                eq = take / max((row > 0).sum(), 1)     # equal share among active tenants
                alloc = np.minimum(row, fair_cap * eq)
                short = take - alloc.sum()
                if short > 1e-12:                       # redistribute the remainder
                    headroom = row - alloc
                    if headroom.sum() > 1e-12:
                        alloc += headroom / headroom.sum() * min(short, headroom.sum())
            queue[s] -= alloc
            served_energy += alloc
            served_carbon += alloc * a[t]
            delay_weighted += alloc * (W - s) * 0.5
            remaining -= alloc.sum()

        util = n[t] + x
        energy[t] = (idle + (1 - idle) * util) * 0.5    # power profile, half-hour periods

    total_carbon = float((energy * a).sum())
    served = served_energy.sum()
    tenant_intensity = np.divide(served_carbon, served_energy,
                                 out=np.full(k, np.nan), where=served_energy > 0)
    ok = ~np.isnan(tenant_intensity)
    jain = float(tenant_intensity[ok].sum() ** 2 / (ok.sum() * (tenant_intensity[ok] ** 2).sum()))
    return {
        "policy": policy, "B_h": B, "fair_cap": fair_cap,
        "total_carbon_gCO2_per_core": total_carbon,
        "work_carbon_intensity": float(served_carbon.sum() / served) if served else np.nan,
        "served_core_periods": float(served),
        "deadline_violation_pct": violations / total_work * 100,
        "mean_delay_h": float(delay_weighted.sum() / served) if served else np.nan,
        "tenant_intensity_jain": jain,
        "tenant_intensity_spread": float(np.nanmax(tenant_intensity) - np.nanmin(tenant_intensity)),
        "worst_tenant_intensity": float(np.nanmax(tenant_intensity)),
        "best_tenant_intensity": float(np.nanmin(tenant_intensity))}


def main():
    t0 = time.time()
    g = load_grid(design_only=True)
    n, arr = load_shapes(g)
    shares, sigma = tenant_shares()
    print("design period: %s to %s, %d periods; tenant lognormal sigma %.2f"
          % (g["from"].min().date(), g["from"].max().date(), len(g), sigma), flush=True)

    med = float(np.median(g["forecast"]))
    budgets = tenant_budgets(len(shares))
    print("tenant deferral budgets (h): " + ", ".join(
        "%g:%d" % (v, (budgets == v).sum()) for v in np.unique(budgets)), flush=True)

    # Tune the Lyapunov weight on the design period only.
    best_V, best_sav = None, -np.inf
    tb = simulate("agnostic", g, n, arr, 24, {}, shares, budgets=budgets)
    for V in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        r = simulate("lyapunov", g, n, arr, 24, {"V": V, "ref": med}, shares, budgets=budgets)
        sav = (tb["work_carbon_intensity"] - r["work_carbon_intensity"]) / tb["work_carbon_intensity"] * 100
        if r["deadline_violation_pct"] <= 5 and sav > best_sav:
            best_V, best_sav = V, sav
    print("tuned Lyapunov V = %s (design-period saving %.2f%%)" % (best_V, best_sav), flush=True)

    rows = []
    for B in (4, 12, 24):
        base = simulate("agnostic", g, n, arr, B, {}, shares, budgets=budgets)
        rows.append(base)
        for policy, params in [("threshold", {"thr": med}),
                               ("percentile", {"p": 30}),
                               ("horizon", {}),
                               ("lyapunov", {"V": best_V, "ref": med}),
                               ("oracle", {})]:
            r = simulate(policy, g, n, arr, B, params, shares, budgets=budgets)
            r["saving_vs_agnostic_pct"] = (base["work_carbon_intensity"] - r["work_carbon_intensity"]) \
                / base["work_carbon_intensity"] * 100
            rows.append(r)
            print("  B=%2d %-11s intensity %6.2f  saving %5.2f%%  delay %5.2f h  viol %4.2f%%  Jain %.4f"
                  % (B, policy, r["work_carbon_intensity"], r["saving_vs_agnostic_pct"],
                     r["mean_delay_h"], r["deadline_violation_pct"], r["tenant_intensity_jain"]),
                  flush=True)
    df = pd.DataFrame(rows)
    df.loc[df.policy == "agnostic", "saving_vs_agnostic_pct"] = 0.0
    df.to_csv(os.path.join(OUT, "phase3_policy_design_period.csv"), index=False)

    # Cost of fairness: the same policy under progressively tighter per-tenant caps.
    frows = []
    base24 = simulate("agnostic", g, n, arr, 24, {}, shares, budgets=budgets)
    for cap in (None, 3.0, 2.0, 1.5, 1.0):
        r = simulate("horizon", g, n, arr, 24, {}, shares, fair_cap=cap, budgets=budgets)
        r["saving_vs_agnostic_pct"] = (base24["work_carbon_intensity"] - r["work_carbon_intensity"]) \
            / base24["work_carbon_intensity"] * 100
        frows.append(r)
        print("  fair_cap %-4s saving %5.2f%%  Jain %.4f  spread %5.1f  worst %6.2f"
              % (cap, r["saving_vs_agnostic_pct"], r["tenant_intensity_jain"],
                 r["tenant_intensity_spread"], r["worst_tenant_intensity"]), flush=True)
    pd.DataFrame(frows).to_csv(os.path.join(OUT, "phase3_fairness_design_period.csv"), index=False)
    print("\nelapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
