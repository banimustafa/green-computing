"""
Phase 10 - Consistency audit.

Every quantitative claim in the manuscript is checked against the result file that produced
it. The audit is mechanical rather than editorial: it reads the stored CSVs, formats each
value exactly as the manuscript states it, and confirms the string is present. A value that
cannot be found is reported rather than silently tolerated.

The audit also records the corrections applied between draft v1 and v1.1, which become the
tracked changes in the Word deliverable.
"""

import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
DOCS = os.path.join(ROOT, "docs")

rows = []


def check(text, group, label, value, fmt="%.2f", source=""):
    """A value counts as present if it appears in the manuscript either bare or with
    thousands separators, since the prose groups large counts and the CSVs do not."""
    s = fmt % value
    variants = {s}
    if fmt == "%d":
        variants.add("{:,}".format(int(value)))
    rows.append({"group": group, "claim": label, "value": s, "source": source,
                 "status": "match" if any(v in text for v in variants) else "NOT FOUND"})


def main():
    t = open(os.path.join(ROOT, "submission", "manuscript.tex")).read()

    h = pd.read_csv(os.path.join(OUT, "phase5_held_out.csv"))
    for sig in ("S1_published", "S2_day_ahead"):
        for B in (4, 12, 24):
            for pol in ("threshold", "percentile", "lyapunov", "horizon", "oracle"):
                r = h[(h.signal == sig) & (h.B_h == B) & (h.policy == pol)]
                if len(r) and (sig == "S1_published" or pol in ("horizon",)):
                    check(t, "held-out", "%s B=%d %s saving" % (sig, B, pol),
                          float(r.mean_saving_pct.iloc[0]), source="phase5_held_out.csv")
                    if pol == "horizon":
                        check(t, "held-out", "%s B=%d share of foresight" % (sig, B),
                              float(r.pct_of_oracle.iloc[0]), source="phase5_held_out.csv")
    hz = h[(h.signal == "S1_published") & (h.policy == "horizon")]
    for _, r in hz.iterrows():
        check(t, "held-out", "B=%d interval low" % r.B_h, r.ci_low, source="phase5_held_out.csv")
        check(t, "held-out", "B=%d interval high" % r.B_h, r.ci_high, source="phase5_held_out.csv")

    a = pd.read_csv(os.path.join(OUT, "phase3e_signal_accuracy.csv"))
    for _, r in a[a.split == "held-out"].iterrows():
        check(t, "signals", "%s held-out MAE" % r.signal, r.MAE, source="phase3e_signal_accuracy.csv")
    check(t, "signals", "S2 held-out correlation",
          float(a[(a.signal == "S2 day-ahead") & (a.split == "held-out")]["corr"].iloc[0]),
          "%.3f", "phase3e_signal_accuracy.csv")

    u = pd.read_csv(os.path.join(OUT, "phase3c_utilisation.csv"))
    for _, r in u.iterrows():
        check(t, "headroom", "utilisation %.2f" % r.utilisation, r.saving_pct,
              source="phase3c_utilisation.csv")

    e = pd.read_csv(os.path.join(OUT, "phase3d_energy_bracket.csv"))
    for _, r in e[e.B_h == 24].iterrows():
        check(t, "energy", "facility saving at idle %.2f" % r.idle_fraction,
              r.facility_saving_pct, source="phase3d_energy_bracket.csv")

    f = pd.read_csv(os.path.join(OUT, "phase3c_flexibility.csv"))
    for _, r in f.iterrows():
        check(t, "fairness", "grant %.0f h saving" % r.grant_h, r.saving_pct,
              source="phase3c_flexibility.csv")
    check(t, "fairness", "worst tenant, no grant",
          float(f[f.grant_h == 0].worst_tenant_intensity.iloc[0]), source="phase3c_flexibility.csv")
    check(t, "fairness", "worst tenant, 24 h grant",
          float(f[f.grant_h == 24].worst_tenant_intensity.iloc[0]), source="phase3c_flexibility.csv")
    fr = pd.read_csv(os.path.join(OUT, "phase3c_fairness.csv"))
    check(t, "fairness", "Jain index under allocation sweep",
          float(fr.tenant_jain.iloc[0]), "%.5f", "phase3c_fairness.csv")

    for fn, tag in (("phase4_spatial_u85.csv", "0.85"), ("phase4_spatial_u45.csv", "0.45")):
        sp = pd.read_csv(os.path.join(OUT, fn))
        for _, r in sp[sp.saving_pct > 0].iterrows():
            check(t, "spatial", "%s %s cap=%s" % (tag, r["mode"], r.exposure_cap),
                  r.saving_pct, source=fn)
    st = pd.read_csv(os.path.join(OUT, "phase4_spatial_stress_u85.csv"))
    sel = st[(st["mode"] == "spatiotemporal") & (st.exposure_cap.isna())]
    check(t, "spatial", "stress test, uncapped spatiotemporal",
          float(sel.saving_pct.mean()), source="phase4_spatial_stress_u85.csv")

    cov = pd.read_csv(os.path.join(OUT, "audit_coverage.csv"))
    n = cov[cov.series == "national_intensity"].iloc[0]
    check(t, "data", "national periods retrieved", n.records, "%d", "audit_coverage.csv")
    check(t, "data", "national periods expected", n.expected_periods, "%d", "audit_coverage.csv")
    check(t, "data", "missing forecasts", n.missing_forecast, "%d", "audit_coverage.csv")
    rg = cov[cov.series == "regional_intensity"].iloc[0]
    check(t, "data", "regional records", rg.records, "%d", "audit_coverage.csv")

    fe = pd.read_csv(os.path.join(OUT, "audit_forecast_error.csv"))
    o = fe[fe.level == "all"].iloc[0]
    check(t, "data", "overall forecast MAE", o.MAE, "%.2f", "audit_forecast_error.csv")
    check(t, "data", "overall forecast RMSE", o.RMSE, "%.2f", "audit_forecast_error.csv")
    check(t, "data", "overall correlation", o["corr"], "%.3f", "audit_forecast_error.csv")

    ac = pd.read_csv(os.path.join(OUT, "audit_accounting.csv")).iloc[0]
    check(t, "data", "accounting correlation", ac.pearson_r, "%.3f", "audit_accounting.csv")
    check(t, "data", "accounting offset", ac.mean_signed_diff_gCO2_kWh, "%.1f", "audit_accounting.csv")

    w = pd.read_csv(os.path.join(OUT, "workload_class_summary.csv"))
    for _, r in w.iterrows():
        check(t, "workload", "%s core-hour share" % r.category, r.pct_of_core_hours,
              "%.2f", "workload_class_summary.csv")
        check(t, "workload", "%s machines" % r.category, r.n_vms, "%d", "workload_class_summary.csv")
    tc = pd.read_csv(os.path.join(OUT, "workload_tenant_concentration.csv")).iloc[0]
    check(t, "workload", "Gini of deferrable core hours", tc.gini_core_hours, "%.3f",
          "workload_tenant_concentration.csv")
    check(t, "workload", "top 1 per cent share", tc.top1pct_tenants_share_pct, "%.2f",
          "workload_tenant_concentration.csv")
    ls = pd.read_csv(os.path.join(OUT, "llm_inference_summary.csv"))
    for _, r in ls.iterrows():
        check(t, "workload", "%s requests" % r.workload, r.total_requests, "%d",
              "llm_inference_summary.csv")
        check(t, "workload", "%s peak-to-mean" % r.workload, r.peak_to_mean_ratio, "%.2f",
              "llm_inference_summary.csv")

    dp = pd.read_csv(os.path.join(OUT, "audit_deferral_potential.csv"))
    r24 = dp[dp.deferral_budget_h == 24].iloc[0]
    check(t, "context", "unconstrained single-job saving at 24 h",
          r24.saving_forecast_guided_pct, "%.2f", "audit_deferral_potential.csv")
    check(t, "context", "single-job share of foresight at 24 h",
          r24.realised_fraction_of_oracle_pct, "%.2f", "audit_deferral_potential.csv")

    m = pd.read_csv(os.path.join(OUT, "phase5_monthly.csv"))
    m = m[(m.policy == "horizon") & (m.B_h == 24)]
    piv = m.pivot_table(index="month", columns="signal", values="saving_pct")
    check(t, "monthly", "lowest month, published", piv.S1_published.min(), source="phase5_monthly.csv")
    check(t, "monthly", "highest month, published", piv.S1_published.max(), source="phase5_monthly.csv")
    check(t, "monthly", "lowest month, day-ahead", piv.S2_day_ahead.min(), source="phase5_monthly.csv")
    win = piv.loc[["2025-12", "2026-01", "2026-02"]].mean()
    sum_ = piv.loc[["2026-06", "2026-07", "2026-08"]].mean()
    check(t, "monthly", "winter mean, published", win.S1_published, source="phase5_monthly.csv")
    check(t, "monthly", "summer mean, published", sum_.S1_published, source="phase5_monthly.csv")
    check(t, "monthly", "winter mean, day-ahead", win.S2_day_ahead, source="phase5_monthly.csv")
    check(t, "monthly", "summer mean, day-ahead", sum_.S2_day_ahead, source="phase5_monthly.csv")
    dpp = pd.read_csv(os.path.join(OUT, "audit_deferral_potential.csv"))
    check(t, "context", "least work made worse", dpp.pct_jobs_worse_than_immediate.min(),
          source="audit_deferral_potential.csv")
    check(t, "context", "most work made worse", dpp.pct_jobs_worse_than_immediate.max(),
          source="audit_deferral_potential.csv")

    ai = pd.read_csv(os.path.join(OUT, "phase6_ai_workload.csv"))
    for _, r in ai[(ai.B_h == 24) & (ai.policy == "horizon")].iterrows():
        check(t, "sensitivity", "AI workload %s saving" % r.workload, r.saving_pct,
              source="phase6_ai_workload.csv")
    check(t, "sensitivity", "AI workload arrival cv",
          float(ai[ai.workload == "ai_training"].arrival_cv.iloc[0]), source="phase6_ai_workload.csv")
    check(t, "sensitivity", "AI workload violations",
          float(ai[(ai.workload == "ai_training") & (ai.B_h == 24) &
                   (ai.policy == "horizon")].deadline_violation_pct.iloc[0]),
          source="phase6_ai_workload.csv")
    ts = pd.read_csv(os.path.join(OUT, "phase6_tenant_sensitivity.csv"))
    # Only the extremes of the sweep are quoted in the text; the full sweep is in the figure.
    check(t, "sensitivity", "lowest mix saving", ts.saving_pct.min(),
          source="phase6_tenant_sensitivity.csv")
    check(t, "sensitivity", "highest mix saving", ts.saving_pct.max(),
          source="phase6_tenant_sensitivity.csv")
    check(t, "sensitivity", "tight mix after grant",
          float(ts[ts["mix"] == "tight"].saving_with_flexibility_grant.iloc[0]),
          source="phase6_tenant_sensitivity.csv")
    check(t, "sensitivity", "generous mix saving",
          float(ts[ts["mix"] == "generous"].saving_pct.iloc[0]),
          source="phase6_tenant_sensitivity.csv")
    check(t, "sensitivity", "bimodal spread before grant",
          float(ts[ts["mix"] == "bimodal"].spread.iloc[0]), "%.1f",
          "phase6_tenant_sensitivity.csv")
    hh = pd.read_csv(os.path.join(OUT, "phase5_held_out.csv"))
    cc = hh[(hh.signal == "S1_published") & (hh.policy == "capacitycurve")]
    for _, r in cc.iterrows():
        check(t, "held-out", "capacity curve B=%d" % r.B_h, r.mean_saving_pct,
              source="phase5_held_out.csv")

    mg = pd.read_csv(os.path.join(OUT, "phase7_marginal_sensitivity.csv"))
    for _, r in mg.iterrows():
        if r.policy != "agnostic":
            check(t, "marginal", "%s scored on %s" % (r.policy, r.scoring), r.saving_pct,
                  source="phase7_marginal_sensitivity.csv")
    mf = pd.read_csv(os.path.join(OUT, "phase7_marginal_factors.csv"))
    check(t, "marginal", "mean marginal factor", mf.mef_gCO2_kWh.mean(), "%.0f",
          "phase7_marginal_factors.csv")
    check(t, "marginal", "lowest marginal factor", mf.mef_gCO2_kWh.min(), "%.0f",
          "phase7_marginal_factors.csv")
    check(t, "marginal", "highest marginal factor", mf.mef_gCO2_kWh.max(), "%.0f",
          "phase7_marginal_factors.csv")
    pw = pd.read_csv(os.path.join(OUT, "phase7_pairwise_policies.csv"))
    s1 = pw[pw.signal == "S1_published"]
    check(t, "pairwise", "distinguishable pairs", s1.distinguishable_5pct.sum(), "%d",
          "phase7_pairwise_policies.csv")
    cc = s1[(s1.policy_a == "capacitycurve") & (s1.policy_b == "horizon")]
    for _, r in cc.iterrows():
        check(t, "pairwise", "capacity curve minus horizon at B=%d" % r.B_h,
              abs(r.mean_difference_pp), source="phase7_pairwise_policies.csv")

    # The pre-specified five-policy interval, quoted in the disclosure paragraph.
    check(t, "disclosure", "original five-policy interval low", 7.93, source="pre-amendment scoring")
    check(t, "disclosure", "original five-policy interval high", 12.16, source="pre-amendment scoring")
    check(t, "disclosure", "imports factor used in the marginal estimate", 328, "%d",
          "intensity_factors.json (mean of three import factors)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "consistency_audit.csv"), index=False)
    bad = df[df.status != "match"]
    print("checked %d values across %d groups" % (len(df), df.group.nunique()))
    print(df.groupby("group").status.value_counts().to_string())
    if len(bad):
        print("\nNOT FOUND:")
        print(bad.to_string(index=False))
    else:
        print("\nevery checked value appears verbatim in the manuscript")

    # Citation coverage: every key cited must exist in the bibliography, and every entry in
    # the bibliography must be cited at least once.
    cited = set()
    for grp in re.findall(r"\\cite\{([^}]+)\}", t):
        cited.update(k.strip() for k in grp.split(","))
    bib = open(os.path.join(ROOT, "submission", "references.bib")).read()
    listed = set(re.findall(r"@\w+\{([^,]+),", bib))
    print("\ncitation keys used: %d, bibliography entries: %d" % (len(cited), len(listed)))
    print("cited but missing from the bibliography: %s" % sorted(cited - listed))
    print("in the bibliography but never cited: %s" % sorted(listed - cited))
    years = [int(y) for y in re.findall(r"year\s*=\s*\{(\d{4})\}", bib)]
    recent = [y for y in years if y >= 2021]
    print("bibliography: %d entries, %d from 2021 or later (%.0f per cent)"
          % (len(years), len(recent), 100.0 * len(recent) / max(len(years), 1)))


if __name__ == "__main__":
    main()
