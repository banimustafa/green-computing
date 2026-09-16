"""
Phase 1 - Parsing, validation, and audit of the retrieved grid data.

Outputs (all written to data/processed/ and outputs/):
  national_intensity.csv      half-hourly forecast and actual, 36 months
  national_generation.csv     half-hourly realised fuel mix
  regional_intensity.csv      half-hourly forecast and mix for 15 regions
  audit_coverage.csv          record counts, gaps, missingness per series
  audit_accounting.csv        recomputed vs published intensity (validation chain)
  audit_forecast_error.csv    forecast error by season and by hour of day
  audit_deferral_potential.csv  oracle vs forecast-guided saving by deferral budget
  audit_spatial_spread.csv    across-region dispersion of the decision signal

The deferral-potential analysis is the gating result for the study: it establishes
whether a saving survives once the scheduler is restricted to the forecast that was
actually published, rather than the realised intensity known only after the event.
"""

import json
import os
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "data", "raw")
PROC = os.path.join(HERE, "..", "data", "processed")
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(PROC, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


# ----------------------------------------------------------------------------------
# 1. Parse the cached responses into tidy panels
# ----------------------------------------------------------------------------------
def parse_national_intensity():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "national_intensity_*.json"))):
        for rec in json.load(open(f))["data"]:
            it = rec.get("intensity", {})
            rows.append((rec["from"], it.get("forecast"), it.get("actual")))
    df = pd.DataFrame(rows, columns=["from", "forecast", "actual"])
    df["from"] = pd.to_datetime(df["from"], utc=True, format="ISO8601")
    return df.drop_duplicates("from").sort_values("from").reset_index(drop=True)


def parse_national_generation():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "national_generation_*.json"))):
        for rec in json.load(open(f))["data"]:
            r = {"from": rec["from"]}
            for m in rec["generationmix"]:
                r[m["fuel"]] = m["perc"]
            rows.append(r)
    df = pd.DataFrame(rows)
    df["from"] = pd.to_datetime(df["from"], utc=True, format="ISO8601")
    return df.drop_duplicates("from").sort_values("from").reset_index(drop=True)


def parse_regional():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "regional_*.json"))):
        blk = json.load(open(f))["data"]
        rid, name = blk["regionid"], blk["shortname"]
        for rec in blk["data"]:
            r = {"from": rec["from"], "regionid": rid, "region": name,
                 "forecast": rec.get("intensity", {}).get("forecast")}
            for m in rec.get("generationmix", []):
                r[m["fuel"]] = m["perc"]
            rows.append(r)
    df = pd.DataFrame(rows)
    df["from"] = pd.to_datetime(df["from"], utc=True, format="ISO8601")
    return df.drop_duplicates(["regionid", "from"]).sort_values(["regionid", "from"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------
# 2. Validation chain: recompute intensity from the realised mix and declared factors
# ----------------------------------------------------------------------------------
def accounting_check(nat, gen):
    factors = json.load(open(os.path.join(RAW, "intensity_factors.json")))["data"][0]
    # The published mix aggregates gas and imports; the representative factors below are
    # the combined-cycle gas factor and the mean of the three published import factors.
    fmap = {"biomass": factors["Biomass"], "coal": factors["Coal"],
            "gas": factors["Gas (Combined Cycle)"], "nuclear": factors["Nuclear"],
            "hydro": factors["Hydro"], "solar": factors["Solar"], "wind": factors["Wind"],
            "other": factors["Other"],
            "imports": np.mean([factors["Dutch Imports"], factors["French Imports"],
                                factors["Irish Imports"]])}
    m = gen.merge(nat[["from", "actual"]], on="from", how="inner").dropna(subset=["actual"])
    recomputed = sum(m[k].fillna(0) * v for k, v in fmap.items() if k in m.columns) / 100.0
    d = recomputed - m["actual"]
    return pd.DataFrame([{
        "n_periods": len(m),
        "pearson_r": float(np.corrcoef(recomputed, m["actual"])[0, 1]),
        "mean_abs_diff_gCO2_kWh": float(d.abs().mean()),
        "median_abs_diff_gCO2_kWh": float(d.abs().median()),
        "mean_signed_diff_gCO2_kWh": float(d.mean()),
        "pct_within_10_gCO2_kWh": float((d.abs() <= 10).mean() * 100),
        "pct_within_25_gCO2_kWh": float((d.abs() <= 25).mean() * 100)}])


# ----------------------------------------------------------------------------------
# 3. Forecast error
# ----------------------------------------------------------------------------------
def forecast_error(nat):
    d = nat.dropna(subset=["actual", "forecast"]).copy()
    d["err"] = d["forecast"] - d["actual"]
    d["season"] = d["from"].dt.month.map(SEASON)
    d["hour"] = d["from"].dt.hour
    d["year"] = d["from"].dt.year

    def agg(g):
        return pd.Series({"n": len(g), "mean_actual": g["actual"].mean(),
                          "bias_ME": g["err"].mean(), "MAE": g["err"].abs().mean(),
                          "RMSE": float(np.sqrt((g["err"] ** 2).mean())),
                          "MAPE_pct": float((g["err"].abs() / g["actual"].clip(lower=1)).mean() * 100),
                          "corr": float(np.corrcoef(g["forecast"], g["actual"])[0, 1])})

    overall = agg(d).to_frame().T.assign(stratum="overall", level="all")
    by_season = d.groupby("season").apply(agg, include_groups=False).reset_index().rename(
        columns={"season": "stratum"}).assign(level="season")
    by_hour = d.groupby("hour").apply(agg, include_groups=False).reset_index().rename(
        columns={"hour": "stratum"}).assign(level="hour_utc")
    by_year = d.groupby("year").apply(agg, include_groups=False).reset_index().rename(
        columns={"year": "stratum"}).assign(level="year")
    return pd.concat([overall, by_season, by_hour, by_year], ignore_index=True), d


# ----------------------------------------------------------------------------------
# 4. Gating analysis: oracle versus forecast-guided deferral
# ----------------------------------------------------------------------------------
def deferral_potential(d, budgets_hours=(1, 2, 4, 6, 12, 24)):
    """For a unit job arriving in each settlement period, compare three placements:
       immediate (carbon-agnostic), forecast-guided (choose the lowest forecast slot in
       the window, pay the realised intensity of that slot), and oracle (lowest realised
       slot in the window). All three are scored on realised intensity."""
    a = d["actual"].to_numpy(dtype=float)
    f = d["forecast"].to_numpy(dtype=float)
    n = len(a)
    rows = []
    for B in budgets_hours:
        w = int(B * 2) + 1  # settlement periods in the window, inclusive of arrival
        idx = np.arange(n)[:, None] + np.arange(w)[None, :]
        valid = idx[idx[:, -1] < n]
        aw, fw = a[valid], f[valid]
        immediate = aw[:, 0]
        oracle = aw.min(axis=1)
        guided = aw[np.arange(len(valid)), fw.argmin(axis=1)]
        delay_guided = fw.argmin(axis=1) * 0.5
        delay_oracle = aw.argmin(axis=1) * 0.5
        base = immediate.mean()
        rows.append({
            "deferral_budget_h": B, "n_jobs": len(valid),
            "immediate_gCO2_kWh": base,
            "forecast_guided_gCO2_kWh": guided.mean(),
            "oracle_gCO2_kWh": oracle.mean(),
            "saving_forecast_guided_pct": (base - guided.mean()) / base * 100,
            "saving_oracle_pct": (base - oracle.mean()) / base * 100,
            "realised_fraction_of_oracle_pct": (base - guided.mean()) / (base - oracle.mean()) * 100,
            "mean_delay_forecast_guided_h": delay_guided.mean(),
            "mean_delay_oracle_h": delay_oracle.mean(),
            "pct_jobs_worse_than_immediate": float((guided > immediate).mean() * 100)})
    return pd.DataFrame(rows)


def spatial_spread(reg):
    """Dispersion of the regional decision signal: what spatial shifting could exploit."""
    dno = reg[reg["regionid"].between(1, 14)]
    piv = dno.pivot_table(index="from", columns="regionid", values="forecast")
    piv = piv.dropna()
    spread = piv.max(axis=1) - piv.min(axis=1)
    out = pd.DataFrame([{
        "n_periods": len(piv), "n_regions": piv.shape[1],
        "mean_min_region_gCO2_kWh": piv.min(axis=1).mean(),
        "mean_max_region_gCO2_kWh": piv.max(axis=1).mean(),
        "mean_spread_gCO2_kWh": spread.mean(),
        "median_spread_gCO2_kWh": spread.median(),
        "p90_spread_gCO2_kWh": spread.quantile(0.9),
        "mean_saving_best_region_pct": ((piv.mean(axis=1) - piv.min(axis=1)) / piv.mean(axis=1) * 100).mean()}])
    share = dno.assign(period=dno["from"]).pivot_table(index="from", columns="regionid", values="forecast")
    cleanest = share.idxmin(axis=1).value_counts(normalize=True).mul(100).rename("pct_periods_cleanest")
    return out, cleanest.reset_index().rename(columns={"index": "regionid"})


def main():
    nat = parse_national_intensity()
    gen = parse_national_generation()
    reg = parse_regional()
    nat.to_csv(os.path.join(PROC, "national_intensity.csv"), index=False)
    gen.to_csv(os.path.join(PROC, "national_generation.csv"), index=False)
    reg.to_csv(os.path.join(PROC, "regional_intensity.csv"), index=False)

    expected = pd.date_range(nat["from"].min(), nat["from"].max(), freq="30min", tz="UTC")
    cov = pd.DataFrame([
        {"series": "national_intensity", "records": len(nat), "expected_periods": len(expected),
         "missing_periods": len(expected) - len(nat),
         "missing_actual": int(nat["actual"].isna().sum()),
         "missing_forecast": int(nat["forecast"].isna().sum()),
         "start": str(nat["from"].min()), "end": str(nat["from"].max())},
        {"series": "national_generation", "records": len(gen), "expected_periods": len(expected),
         "missing_periods": len(expected) - len(gen), "missing_actual": 0, "missing_forecast": 0,
         "start": str(gen["from"].min()), "end": str(gen["from"].max())},
        {"series": "regional_intensity", "records": len(reg),
         "expected_periods": len(expected) * reg["regionid"].nunique(),
         "missing_periods": len(expected) * reg["regionid"].nunique() - len(reg),
         "missing_actual": int(len(reg)), "missing_forecast": int(reg["forecast"].isna().sum()),
         "start": str(reg["from"].min()), "end": str(reg["from"].max())}])
    cov.to_csv(os.path.join(OUT, "audit_coverage.csv"), index=False)

    accounting_check(nat, gen).to_csv(os.path.join(OUT, "audit_accounting.csv"), index=False)
    fe, d = forecast_error(nat)
    fe.to_csv(os.path.join(OUT, "audit_forecast_error.csv"), index=False)
    deferral_potential(d).to_csv(os.path.join(OUT, "audit_deferral_potential.csv"), index=False)
    sp, cleanest = spatial_spread(reg)
    sp.to_csv(os.path.join(OUT, "audit_spatial_spread.csv"), index=False)
    cleanest.to_csv(os.path.join(OUT, "audit_cleanest_region_share.csv"), index=False)

    print(cov.to_string(index=False))
    print("\nACCOUNTING VALIDATION\n", pd.read_csv(os.path.join(OUT, "audit_accounting.csv")).to_string(index=False))
    print("\nFORECAST ERROR (overall + season)\n",
          fe[fe["level"].isin(["all", "season", "year"])][
              ["level", "stratum", "n", "mean_actual", "bias_ME", "MAE", "RMSE", "MAPE_pct", "corr"]
          ].to_string(index=False))
    print("\nDEFERRAL POTENTIAL\n",
          pd.read_csv(os.path.join(OUT, "audit_deferral_potential.csv")).round(2).to_string(index=False))
    print("\nSPATIAL SPREAD\n", sp.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
