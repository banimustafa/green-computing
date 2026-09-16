"""
Marginal-signal sensitivity.

The scheduler decides on the published average carbon intensity, because that is what the system
operator publishes. The emissions actually displaced by moving a unit of work are marginal. No
marginal emissions series is published for the British system, so one is estimated here from
public data using the regression method of Siler-Evans and colleagues
(doi:10.1021/es300145v): within bins of season and time of day, the change in system emissions
between consecutive settlement periods is regressed on the change in system generation, and the
slope is taken as the marginal emission factor for that bin.

Policies then decide on the published average forecast, exactly as before, and are scored twice:
on realised average intensity, and on the estimated marginal series. The comparison shows how much
of the reported saving survives a marginal accounting.

The estimate is ours, not the operator's, and is reported as a sensitivity rather than as a
measurement. Its limitations are stated in the manuscript.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_policies import (load_grid, load_shapes, simulate, tenant_profile,  # noqa: E402
                               saving, PROC, OUT)

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
# Published fuel-specific intensity factors, gCO2 per kWh, as used throughout the study.
FACTORS = {"GAS": 394, "COAL": 937, "NUCLEAR": 0, "WIND": 0, "WIND_EMB": 0, "HYDRO": 0,
           "IMPORTS": 328, "BIOMASS": 120, "OTHER": 300, "SOLAR": 0, "STORAGE": 0}
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def estimate_mef():
    f = os.path.join(RAW, "neso_historic_generation_mix.csv")
    if not os.path.exists(f):
        raise SystemExit("neso_historic_generation_mix.csv is required for this stage; see README")
    d = pd.read_csv(f, parse_dates=["DATETIME"])
    d = d[d.DATETIME >= "2018-01-01"].copy()
    cols = [c for c in FACTORS if c in d.columns]
    # System emissions in tonnes per hour, and system generation in MW.
    d["E"] = sum(d[c].fillna(0) * FACTORS[c] for c in cols) / 1e6
    d["G"] = d[cols].fillna(0).sum(axis=1)
    d = d.sort_values("DATETIME")
    d["dE"] = d.E.diff()
    d["dG"] = d.G.diff()
    d["season"] = d.DATETIME.dt.month.map(SEASON)
    d["hour"] = d.DATETIME.dt.hour
    d = d.dropna(subset=["dE", "dG"])
    d = d[(d.dG.abs() > 50) & (d.dG.abs() < 5000)]      # drop flat and implausible steps

    rows = []
    for (s, h), g in d.groupby(["season", "hour"]):
        if len(g) < 100:
            continue
        # E is held as sum(MW x factor)/1e6, so the regression slope is the marginal factor
        # divided by 1e6; the multiplication restores gCO2 per kWh.
        slope = float(np.polyfit(g.dG, g.dE, 1)[0]) * 1e6
        rows.append({"season": s, "hour": h, "mef_gCO2_kWh": slope, "n": len(g)})
    mef = pd.DataFrame(rows)
    mef.to_csv(os.path.join(OUT, "phase7_marginal_factors.csv"), index=False)
    return mef


def main():
    mef = estimate_mef()
    print("estimated marginal factors: %d bins, mean %.0f gCO2/kWh, range %.0f to %.0f"
          % (len(mef), mef.mef_gCO2_kWh.mean(), mef.mef_gCO2_kWh.min(), mef.mef_gCO2_kWh.max()),
          flush=True)

    g = load_grid(design_only=False)
    g = g[g["from"] >= pd.Timestamp("2025-09-01", tz="UTC")].reset_index(drop=True)
    key = pd.DataFrame({"season": g["from"].dt.month.map(SEASON), "hour": g["from"].dt.hour})
    marg = key.merge(mef, on=["season", "hour"], how="left").mef_gCO2_kWh.to_numpy()
    marg = np.where(np.isfinite(marg), marg, np.nanmean(mef.mef_gCO2_kWh))
    print("average signal: mean %.1f, s.d. %.1f | marginal signal: mean %.1f, s.d. %.1f"
          % (g.actual.mean(), g.actual.std(), marg.mean(), marg.std()), flush=True)

    shares, budgets, _ = tenant_profile()
    n, arr = load_shapes(g, 0.85)
    med = float(np.median(g["forecast"]))
    rows = []
    for scoring, field in (("average", g["actual"].to_numpy(float)), ("marginal", marg)):
        gs = g.copy()
        gs["actual"] = field
        base = simulate("agnostic", gs, n, arr, 24, {}, shares, budgets)
        for pol, pr in [("threshold", {"thr": med}), ("horizon", {}), ("oracle", {})]:
            r = simulate(pol, gs, n, arr, 24, pr, shares, budgets)
            r["scoring"] = scoring
            r["saving_pct"] = saving(base, r)
            rows.append(r)
            print("  scored on %-8s %-10s saving %5.2f%%" % (scoring, pol, r["saving_pct"]),
                  flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "phase7_marginal_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()
