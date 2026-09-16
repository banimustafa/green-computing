"""
Cross-check of the result files.

The forward audit confirms that stored values reach the manuscript. The full audit confirms that
every number in the manuscript comes from somewhere. Neither asks whether the stored values are
consistent with one another, which is the remaining place an error can sit undetected: a file can
be internally coherent, reach the manuscript faithfully, and still disagree with the file it was
derived from.

This script states the identities that must hold between and within the result files and tests each
one. An identity that fails is a defect in the results, not in the prose.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
PROC = os.path.join(ROOT, "data", "processed")

checks = []


def check(name, ok, detail="", tol=""):
    checks.append({"check": name, "result": "pass" if ok else "FAIL",
                   "detail": detail, "tolerance": tol})


def close(a, b, tol):
    return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= tol))


def load(n):
    return pd.read_csv(os.path.join(OUT, n))


def main():
    # ---------------------------------------------------------------- within-file identities
    h = load("phase5_held_out.csv")
    orc = h[h.policy == "oracle"].set_index(["signal", "B_h"]).mean_saving_pct
    recomputed = [100 * r.mean_saving_pct / orc[(r.signal, r.B_h)] for r in h.itertuples()]
    check("held-out: share of oracle equals saving divided by the oracle saving",
          close(recomputed, h.pct_of_oracle, 1e-6), "%d rows" % len(h), "1e-6")
    check("held-out: mean lies inside its bootstrap interval",
          bool(((h.ci_low <= h.mean_saving_pct) & (h.mean_saving_pct <= h.ci_high)).all()),
          "%d rows" % len(h))
    check("held-out: Holm-adjusted p is never below the raw p",
          bool((h.p_holm >= h.p_wilcoxon - 1e-12).all()), "%d rows" % len(h))
    check("held-out: months positive never exceeds months compared",
          bool((h.months_positive <= h.n_months).all()) and bool((h.n_months == 12).all()),
          "all strata have twelve months")

    # ---------------------------------------------------------------- monthly against aggregate
    m = load("phase5_monthly.csv")
    agg = m.groupby(["signal", "B_h", "policy"]).saving_pct.mean().reset_index()
    j = agg.merge(h[["signal", "B_h", "policy", "mean_saving_pct"]],
                  on=["signal", "B_h", "policy"], how="inner")
    check("monthly series averages to the reported held-out mean",
          close(j.saving_pct, j.mean_saving_pct, 5e-3), "%d policy strata compared" % len(j),
          "0.005 pp")
    check("monthly series covers twelve months in every stratum",
          bool((m.groupby(["signal", "B_h", "policy"]).month.nunique() == 12).all()))

    # ---------------------------------------------------------------- pairwise against held-out
    pw = load("phase7_pairwise_policies.csv")
    piv = h.set_index(["signal", "B_h", "policy"]).mean_saving_pct
    diff = [piv[(r.signal, r.B_h, r.policy_a)] - piv[(r.signal, r.B_h, r.policy_b)]
            for r in pw.itertuples()]
    check("pairwise differences equal the difference of the reported means",
          close(diff, pw.mean_difference_pp, 5e-3), "%d pairs" % len(pw), "0.005 pp")
    check("pairwise: mean difference lies inside its bootstrap interval",
          bool(((pw.ci_low <= pw.mean_difference_pp) &
                (pw.mean_difference_pp <= pw.ci_high)).all()))
    check("pairwise: every comparison uses twelve paired months",
          bool((pw.n_months == 12).all()))

    # ---------------------------------------------------------------- design period agreement
    p3 = load("phase3c_policies.csv")
    sig = load("phase3e_policies_both_signals.csv")
    s1 = sig[sig.signal == "S1_published"]
    pairs = []
    for B in (12, 24):
        for pol in ("threshold", "horizon", "oracle"):
            a = p3[(p3.B_h == B) & (p3.policy == pol)].saving_pct
            b = s1[(s1.B_h == B) & (s1.policy == pol)].saving_pct
            if len(a) and len(b):
                pairs.append((float(a.iloc[0]), float(b.iloc[0])))
    # The two runs do not cover identical windows: the day-ahead signal needs fourteen days of
    # lagged intensity, so the signal run begins 672 settlement periods later and covers 34,252
    # periods against 34,924. The identity is therefore stated with a tolerance that admits the
    # window difference, and the window difference is itself checked.
    gap = max(abs(x - y) for x, y in pairs)
    check("design-period savings agree between the policy run and the signal run",
          close([x for x, _ in pairs], [y for _, y in pairs], 0.10),
          "%d comparisons, largest difference %.3f pp, explained by the warm-up window"
          % (len(pairs), gap), "0.10 pp")
    nat_all = pd.read_csv(os.path.join(PROC, "national_intensity.csv"), parse_dates=["from"])
    sigp = pd.read_csv(os.path.join(PROC, "national_intensity_signals.csv"), parse_dates=["from"])
    n_pol = int((nat_all.dropna(subset=["actual", "forecast"])["from"] <
                 pd.Timestamp("2025-09-01", tz="UTC")).sum())
    n_sig = int((sigp.dropna()["from"] < pd.Timestamp("2025-09-01", tz="UTC")).sum())
    check("the signal run is shorter by exactly the day-ahead warm-up",
          n_pol - n_sig == 672, "%d periods against %d, difference %d"
          % (n_pol, n_sig, n_pol - n_sig), "672 periods, fourteen days")

    fx = load("phase3c_flexibility.csv")
    base24 = float(p3[(p3.B_h == 24) & (p3.policy == "horizon")].saving_pct.iloc[0])
    check("flexibility sweep without a grant reproduces the design-period planner result",
          close(float(fx[fx.grant_h == 0].saving_pct.iloc[0]), base24, 0.05),
          "no grant %.2f against %.2f" % (float(fx[fx.grant_h == 0].saving_pct.iloc[0]), base24),
          "0.05 pp")
    fr = load("phase3c_fairness.csv")
    check("fairness sweep at zero weight reproduces the same result",
          close(float(fr[fr.fair_lambda == 0].saving_pct.iloc[0]), base24, 0.05), tol="0.05 pp")
    ai = load("phase6_ai_workload.csv")
    tr = ai[(ai.workload == "trace") & (ai.B_h == 24) & (ai.policy == "horizon")].saving_pct
    check("trace arm of the workload sensitivity reproduces the design-period result",
          close(float(tr.iloc[0]), base24, 0.05), tol="0.05 pp")

    # ---------------------------------------------------------------- energy bracket invariance
    e = load("phase3d_energy_bracket.csv")
    for B, g in e.groupby("B_h"):
        check("energy bracket: work-carbon saving is invariant to the idle fraction at B=%d" % B,
              close(g.saving_pct, g.saving_pct.iloc[0], 1e-6),
              "values %s" % np.round(g.saving_pct.to_numpy(), 4).tolist(), "1e-6")
    check("energy bracket: facility saving falls as the idle fraction rises",
          bool(all(g.sort_values("idle_fraction").facility_saving_pct.is_monotonic_decreasing
                   for _, g in e.groupby("B_h"))))

    # ---------------------------------------------------------------- deferral potential identity
    dp = load("audit_deferral_potential.csv")
    lhs = (dp.immediate_gCO2_kWh - dp.forecast_guided_gCO2_kWh) / \
          (dp.immediate_gCO2_kWh - dp.oracle_gCO2_kWh) * 100
    check("deferral potential: realised share equals the ratio of the two savings",
          close(lhs, dp.realised_fraction_of_oracle_pct, 1e-6), "%d budgets" % len(dp), "1e-6")
    check("deferral potential: perfect foresight is never beaten by the forecast",
          bool((dp.saving_oracle_pct >= dp.saving_forecast_guided_pct).all()))

    # ---------------------------------------------------------------- workload arithmetic
    w = load("workload_class_summary.csv")
    check("workload classes: core-hour shares sum to one hundred per cent",
          close(w.pct_of_core_hours.sum(), 100.0, 1e-6), "%.6f" % w.pct_of_core_hours.sum())
    check("workload classes: machine counts sum to the trace total",
          int(w.n_vms.sum()) == 2695548, "%d" % int(w.n_vms.sum()))
    check("workload classes: machine shares sum to one hundred per cent",
          close(w.pct_of_vms.sum(), 100.0, 0.01), "%.4f" % w.pct_of_vms.sum(), "0.01")
    ls = load("llm_inference_summary.csv")
    check("inference summary: peak-to-mean equals peak divided by mean",
          close(ls.peak_requests_per_30min / ls.mean_requests_per_30min, ls.peak_to_mean_ratio,
                1e-6), "%d workloads" % len(ls), "1e-6")
    check("inference summary: request total matches the panel",
          int(ls.total_requests.sum()) ==
          int(pd.read_csv(os.path.join(PROC, "llm_inference_halfhourly.csv")).requests.sum()))

    # ---------------------------------------------------------------- coverage against the panel
    cov = load("audit_coverage.csv")
    nat = pd.read_csv(os.path.join(PROC, "national_intensity.csv"))
    r = cov[cov.series == "national_intensity"].iloc[0]
    check("coverage: recorded periods equal the rows in the national panel",
          int(r.records) == len(nat), "%d against %d" % (int(r.records), len(nat)))
    check("coverage: missing equals expected minus recorded",
          int(r.missing_periods) == int(r.expected_periods) - int(r.records))
    reg = pd.read_csv(os.path.join(PROC, "regional_intensity.csv"), usecols=["regionid"])
    check("coverage: regional record count equals the regional panel",
          int(cov[cov.series == "regional_intensity"].records.iloc[0]) == len(reg))

    # ---------------------------------------------------------------- forecast error recomputed
    fe = load("audit_forecast_error.csv")
    d = nat.dropna(subset=["actual", "forecast"])
    err = d.forecast - d.actual
    o = fe[fe.level == "all"].iloc[0]
    check("forecast error: overall MAE recomputes from the panel",
          close(err.abs().mean(), o.MAE, 1e-6), "%.6f against %.6f" % (err.abs().mean(), o.MAE),
          "1e-6")
    check("forecast error: overall RMSE recomputes from the panel",
          close(np.sqrt((err ** 2).mean()), o.RMSE, 1e-6), tol="1e-6")
    check("forecast error: stratum counts sum to the overall count within each level",
          bool(all(close(fe[fe.level == lv].n.sum(), o.n, 1.0)
                   for lv in ("season", "hour_utc"))), "season and hour strata", "1 period")

    # ---------------------------------------------------------------- signals and marginal
    sa = load("phase3e_signal_accuracy.csv")
    sg = pd.read_csv(os.path.join(PROC, "national_intensity_signals.csv"))
    ho = sg[(sg["from"] >= "2025-09-01")].dropna(subset=["actual", "forecast_published"])
    mae = (ho.forecast_published - ho.actual).abs().mean()
    check("signal accuracy: held-out MAE of the published forecast recomputes from the panel",
          close(mae, float(sa[(sa.signal == "S1 published") &
                              (sa.split == "held-out")].MAE.iloc[0]), 1e-6), "%.6f" % mae, "1e-6")
    mf = load("phase7_marginal_factors.csv")
    check("marginal factors: four seasons by twenty-four hours gives ninety-six bins",
          len(mf) == 96 and mf.season.nunique() == 4 and mf.hour.nunique() == 24,
          "%d bins" % len(mf))
    check("marginal factors: every estimate is positive and below the coal factor",
          bool(((mf.mef_gCO2_kWh > 0) & (mf.mef_gCO2_kWh < 937)).all()),
          "range %.0f to %.0f" % (mf.mef_gCO2_kWh.min(), mf.mef_gCO2_kWh.max()))

    # ---------------------------------------------------------------- spatial arithmetic
    for f in ("phase4_spatial_u85.csv", "phase4_spatial_u45.csv"):
        sp = load(f)
        check("%s: the carbon-agnostic baseline saves nothing" % f,
              close(float(sp[sp["mode"] == "agnostic"].saving_pct.iloc[0]), 0.0, 1e-9))
        check("%s: combined placement is at least as good as either alone" % f,
              float(sp[(sp["mode"] == "spatiotemporal") & sp.exposure_cap.isna()].saving_pct.iloc[0])
              >= max(float(sp[sp["mode"] == "temporal"].saving_pct.iloc[0]),
                     float(sp[sp["mode"] == "spatial"].saving_pct.iloc[0])))
        check("%s: regional shares are consistent with the number of regions" % f,
              bool((sp[sp.saving_pct > 0].top_region_share_pct >= 100.0 / 14 - 1e-6).all()),
              "equal share is %.2f per cent" % (100.0 / 14))

    # ---------------------------------------------------------------- tenant sensitivity
    ts = load("phase6_tenant_sensitivity.csv")
    check("tenant sensitivity: the flexibility grant never lowers the saving",
          bool((ts.saving_with_flexibility_grant >= ts.saving_pct - 1e-9).all()))
    check("tenant sensitivity: the flexibility grant never widens the tenant spread",
          bool((ts.spread_with_flexibility_grant <= ts.spread + 1e-9).all()))
    check("tenant sensitivity: the allocation lever moves the saving by less than 0.01 pp",
          bool(((ts.saving_with_allocation_lever - ts.saving_pct).abs() < 0.01).all()),
          "largest change %.4f pp"
          % (ts.saving_with_allocation_lever - ts.saving_pct).abs().max(), "0.01 pp")

    df = pd.DataFrame(checks)
    df.to_csv(os.path.join(OUT, "crosscheck_results.csv"), index=False)
    print(df.to_string(index=False))
    n_fail = int((df.result == "FAIL").sum())
    print("\n%d checks, %d passed, %d failed" % (len(df), len(df) - n_fail, n_fail))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
