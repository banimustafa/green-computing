"""
Phase 4 - Spatial extension.

Deferrable work may now be placed in time and across the fourteen distribution regions of
the British grid. Interactive load stays where it is: it is latency-sensitive and is served
locally, so it consumes regional capacity that the scheduler cannot reclaim.

Two limits shape what this phase can claim, and both are stated rather than worked around.

  No regional outturn exists. The system operator publishes a regional carbon intensity
  forecast and no realised regional series, so spatial results are scored on the forecast
  field. To keep the comparison internally consistent, the temporal-only baseline is scored
  on the same field. The figures are therefore a bound on what regional placement offers,
  not a measured saving.

  The bound is loose by construction. The cleanest regions are low-demand and wind-rich and
  could not physically absorb arbitrary load, which is precisely why the exposure cap is
  introduced rather than reported as an afterthought.

A stress test perturbs the regional forecasts with the empirical national forecast-error
distribution, resampled in blocks to preserve autocorrelation, and rescores. That estimates
how much of the spatial gain would survive error of the magnitude observed nationally. It is
a synthetic ground truth and is labelled as such.

Usage:  python3 src/simulate_spatial.py [months]
"""

import heapq
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")

W = 24                     # planning horizon in half-hourly periods (12 hours)
INTERACTIVE_SHARE = 0.357
UTILISATION = 0.85
N_TENANTS = 200
SEED = 20260913
TOL = 1e-9
BLOCK = 48                 # block length for the bootstrap, one day


def load_regional(months):
    r = pd.read_csv(os.path.join(PROC, "regional_intensity.csv"),
                    usecols=["from", "regionid", "forecast"], parse_dates=["from"])
    r = r[r.regionid.between(1, 14)]
    piv = r.pivot_table(index="from", columns="regionid", values="forecast").dropna()
    piv = piv.iloc[:int(months * 30.44 * 48)]
    return piv


def load_national_errors():
    d = pd.read_csv(os.path.join(PROC, "national_intensity.csv"), parse_dates=["from"])
    d = d.dropna(subset=["actual", "forecast"])
    return (d["forecast"] - d["actual"]).to_numpy()


def shapes(index, utilisation=UTILISATION):
    di = pd.read_csv(os.path.join(OUT, "llm_inference_diurnal.csv"))
    inf = (di.groupby("hour").pct_of_daily_mean.mean() / 100.0).reindex(range(24)).ffill()
    ap = pd.read_csv(os.path.join(OUT, "workload_arrival_profile.csv"))
    arr = ap[ap.category == "Delay-insensitive"].set_index("hour_of_day").pct_arrivals
    arr = (arr / arr.mean()).reindex(range(24)).ffill().bfill()
    h = index.hour
    n = np.clip(inf.to_numpy()[h] * utilisation * INTERACTIVE_SHARE, 0, 0.95)
    a = np.clip(arr.to_numpy()[h] * utilisation * (1 - INTERACTIVE_SHARE), 0, 0.95)
    return n, a


def plan(sig_win, cap_win, qs, exposure_cap=None):
    """Cheapest-slot plan over a window of periods by regions.

    sig_win : (P, R) intensity of each period-region slot
    cap_win : (P, R) spare capacity of each slot
    qs      : (P,) backlog by deadline class, class s admissible in periods 0..s
    Returns the work assigned to the current period, and its distribution across regions.
    """
    P, R = sig_win.shape
    rem = cap_win.copy()
    heap = []
    assign0 = np.zeros(R)
    for s in range(min(len(qs), P)):
        for rgn in range(R):
            heapq.heappush(heap, (float(sig_win[s, rgn]), s, rgn))
        q = float(qs[s])
        while q > TOL and heap:
            _, p, rgn = heap[0]
            take = min(q, rem[p, rgn])
            if take > TOL:
                rem[p, rgn] -= take
                q -= take
                if p == 0:
                    assign0[rgn] += take
            if rem[p, rgn] <= TOL:
                heapq.heappop(heap)
            else:
                break
    return assign0


def simulate(mode, sig, score, n, arr, budgets, shares, exposure_cap=None):
    """mode: 'agnostic' | 'temporal' | 'spatial' | 'spatiotemporal'."""
    T, R = sig.shape
    k = len(shares)
    cap_all = np.tile(np.clip((1.0 - n) / R, 0.0, None)[:, None], (1, R))  # spare capacity per site
    slot = np.minimum((budgets * 2).astype(int), W)
    if mode in ("agnostic", "spatial"):
        slot = np.zeros(k, dtype=int)                        # no deferral permitted
    queue = np.zeros((W + 1, k))
    served_e = served_c = 0.0
    region_load = np.zeros(R)
    total = 0.0

    for t in range(T - W - 1):
        queue = np.roll(queue, -1, axis=0)
        queue[-1] = 0.0
        np.add.at(queue, (slot, np.arange(k)), arr[t] * shares)
        total += arr[t]
        qs = queue.sum(axis=1)
        if qs.sum() <= TOL:
            continue

        cap_win = cap_all[t:t + W + 1]
        if mode == "agnostic":
            # Each site runs its own share of the work immediately, where it arrived.
            assign = np.minimum(qs[0] / R, cap_win[0])
        elif mode == "temporal":
            # Deferral is permitted but routing is not: every site plans over its own
            # signal and its own capacity, holding its share of the work.
            assign = np.zeros(R)
            for rgn in range(R):
                a1 = plan(sig[t:t + W + 1, [rgn]], cap_win[:, [rgn]], qs / R, None)
                assign[rgn] = a1[0]
        elif mode == "spatial":
            # Routing is permitted but deferral is not: place now, in the cleanest sites.
            order = np.argsort(sig[t])
            left, assign = qs[0], np.zeros(R)
            for rgn in order:
                take = min(left, cap_win[0, rgn])
                assign[rgn] = take
                left -= take
                if left <= TOL:
                    break
        else:
            cw = cap_win
            if exposure_cap is not None and region_load.sum() > TOL:
                # Cumulative exposure: a region that has already taken more than
                # exposure_cap times an equal share of all shifted work is closed until the
                # rest of the fleet catches up. The constraint is on accumulated exposure,
                # not on any single period, because concentration builds over time.
                over = (region_load / region_load.sum()) > (exposure_cap / R)
                if over.any():
                    cw = cap_win.copy()
                    cw[:, over] = 0.0
            assign = plan(sig[t:t + W + 1], cw, qs, exposure_cap)

        placed = assign.sum()
        if placed > TOL:
            share_of_q = min(placed, qs.sum())
            drawn = 0.0
            for s in range(W + 1):                           # earliest deadline first
                if drawn >= share_of_q - TOL:
                    break
                take = min(queue[s].sum(), share_of_q - drawn)
                if take <= TOL:
                    continue
                queue[s] -= queue[s] / queue[s].sum() * take
                drawn += take
            served_e += placed
            served_c += float((assign * score[t]).sum())
            region_load += assign

    return {"mode": mode, "exposure_cap": exposure_cap,
            "work_carbon_intensity": served_c / served_e if served_e else np.nan,
            "served": served_e,
            "top_region_share_pct": region_load.max() / region_load.sum() * 100,
            "regions_used": int((region_load > 1e-6).sum()),
            "herfindahl": float(((region_load / region_load.sum()) ** 2).sum())}


def block_bootstrap_errors(errs, size, rng):
    out = np.empty(size)
    i = 0
    while i < size:
        start = rng.integers(0, len(errs) - BLOCK)
        take = min(BLOCK, size - i)
        out[i:i + take] = errs[start:start + take]
        i += take
    return out


def main():
    months = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    util = float(sys.argv[2]) if len(sys.argv) > 2 else UTILISATION
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    piv = load_regional(months)
    sig = piv.to_numpy(float)
    n, arr = shapes(piv.index, util)
    s = rng.lognormal(0.0, 1.92, size=N_TENANTS)
    shares = s / s.sum()
    budgets = rng.choice([2.0, 4.0, 8.0, 12.0, 24.0], size=N_TENANTS, p=[.15, .20, .25, .20, .20])
    print("periods %d (%.1f months), regions %d, utilisation %.2f"
          % (len(piv), months, sig.shape[1], util), flush=True)

    rows = []
    base = simulate("agnostic", sig, sig, n, arr, budgets, shares)
    base["saving_pct"] = 0.0
    rows.append(base)
    for mode, cap in [("temporal", None), ("spatial", None), ("spatiotemporal", None),
                      ("spatiotemporal", 2.0), ("spatiotemporal", 1.5), ("spatiotemporal", 1.1)]:
        r = simulate(mode, sig, sig, n, arr, budgets, shares, exposure_cap=cap)
        r["saving_pct"] = (base["work_carbon_intensity"] - r["work_carbon_intensity"]) \
            / base["work_carbon_intensity"] * 100
        rows.append(r)
        print("  %-15s cap %-4s saving %6.2f%%  top region %5.1f%%  regions used %2d  HHI %.3f"
              % (mode, cap, r["saving_pct"], r["top_region_share_pct"], r["regions_used"],
                 r["herfindahl"]), flush=True)
    pd.DataFrame(rows).assign(utilisation=util, months=months).to_csv(
        os.path.join(OUT, "phase4_spatial_u%02d.csv" % int(util * 100)), index=False)

    # Stress test: score the same decisions against a perturbed field.
    errs = load_national_errors()
    stress = []
    for rep in range(3):
        noise = np.stack([block_bootstrap_errors(errs, len(sig), rng) for _ in range(sig.shape[1])], axis=1)
        score = np.clip(sig - noise, 1.0, None)
        b = simulate("agnostic", sig, score, n, arr, budgets, shares)
        for mode, cap in [("temporal", None), ("spatiotemporal", None), ("spatiotemporal", 2.0)]:
            r = simulate(mode, sig, score, n, arr, budgets, shares, exposure_cap=cap)
            stress.append({"rep": rep, "mode": mode, "exposure_cap": cap,
                           "saving_pct": (b["work_carbon_intensity"] - r["work_carbon_intensity"])
                           / b["work_carbon_intensity"] * 100})
    st = pd.DataFrame(stress)
    st.assign(utilisation=util).to_csv(
        os.path.join(OUT, "phase4_spatial_stress_u%02d.csv" % int(util * 100)), index=False)
    print("\nSTRESS TEST (decisions on forecast, scored on perturbed field)")
    print(st.groupby(["mode", "exposure_cap"], dropna=False).saving_pct.agg(["mean", "std"]).round(2).to_string())
    print("\nelapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
