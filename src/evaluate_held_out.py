"""
Phase 5 - Held-out evaluation.

Months 25 to 36 (September 2025 to August 2026) are scored once, with every policy parameter
frozen at its design-period value:

    Lyapunov weight V = 20.0          selected on months 1-24
    percentile p = 30                 fixed in the analysis plan
    threshold = design-period median forecast intensity
    tenant profile seed = 20260913    fixed in the analysis plan
    S2 forecast model                 fitted on months 1-24 only

Inference uses the twelve held-out months as paired observations. For each policy the monthly
saving against the carbon-agnostic baseline gives twelve differences; the reported interval is
a percentile bootstrap over those months, and significance is assessed with a Wilcoxon signed
rank test corrected across policies by the Holm-Bonferroni procedure.

Every result is produced under both signals. A claim is made only where it holds under both.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_policies import (load_shapes, simulate, tenant_profile, PROC, OUT,  # noqa: E402
                               DESIGN_END)

HELD_OUT_START = pd.Timestamp(DESIGN_END, tz="UTC")
FROZEN = {"V": 20.0, "p": 30, "q": 50}  # q: daily capacity-curve quantile, fixed a priori
N_BOOT = 10000
RNG = np.random.default_rng(20260913)


def held_out_frames():
    d = pd.read_csv(os.path.join(PROC, "national_intensity_signals.csv"), parse_dates=["from"])
    d = d.dropna(subset=["actual", "forecast_published", "forecast_s2"])
    design_median = float(d.loc[d["from"] < HELD_OUT_START, "forecast_published"].median())
    d = d[d["from"] >= HELD_OUT_START].reset_index(drop=True)
    frames = {}
    for label, col in (("S1_published", "forecast_published"), ("S2_day_ahead", "forecast_s2")):
        frames[label] = d.rename(columns={col: "forecast"})[["from", "actual", "forecast"]].copy()
    return frames, design_median


def paired_stats(base_m, pol_m):
    months = sorted(set(base_m) & set(pol_m))
    diff = np.array([(base_m[m] - pol_m[m]) / base_m[m] * 100 for m in months])
    boot = np.array([RNG.choice(diff, len(diff), replace=True).mean() for _ in range(N_BOOT)])
    try:
        p = float(wilcoxon(diff, alternative="greater").pvalue)
    except ValueError:
        p = np.nan
    return {"n_months": len(months), "mean_saving_pct": float(diff.mean()),
            "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
            "min_month": float(diff.min()), "max_month": float(diff.max()),
            "months_positive": int((diff > 0).sum()), "p_wilcoxon": p}


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    t0 = time.time()
    frames, design_median = held_out_frames()
    shares, budgets, _ = tenant_profile()
    rows = []
    for label, g in frames.items():
        n, arr = load_shapes(g, 0.85)
        print("%s: %d held-out periods, %s to %s"
              % (label, len(g), g["from"].min().date(), g["from"].max().date()), flush=True)
        for B in (4, 12, 24):
            base = simulate("agnostic", g, n, arr, B, {}, shares, budgets, monthly=True)
            for pol, pr in [("threshold", {"thr": design_median}),
                            ("percentile", {"p": FROZEN["p"]}),
                            ("capacitycurve", {"q": FROZEN["q"]}),
                            ("horizon", {}),
                            ("lyapunov", {"V": FROZEN["V"], "ref": design_median}),
                            ("oracle", {})]:
                r = simulate(pol, g, n, arr, B, pr, shares, budgets, monthly=True)
                st = paired_stats(base["monthly"], r["monthly"])
                st.update({"signal": label, "B_h": B, "policy": pol,
                           "mean_delay_h": r["mean_delay_h"],
                           "violation_pct": r["deadline_violation_pct"],
                           "tenant_jain": r["tenant_jain"]})
                rows.append(st)
                print("  B=%2d %-11s %5.2f%% [%5.2f, %5.2f]  %2d/%d months positive  p=%.4g"
                      % (B, pol, st["mean_saving_pct"], st["ci_low"], st["ci_high"],
                         st["months_positive"], st["n_months"], st["p_wilcoxon"]), flush=True)
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for (sig, B), idx in df.groupby(["signal", "B_h"]).groups.items():
        sub = df.loc[idx]
        df.loc[idx, "p_holm"] = holm(sub["p_wilcoxon"].to_numpy())
    df["significant_5pct"] = df["p_holm"] < 0.05
    df.to_csv(os.path.join(OUT, "phase5_held_out.csv"), index=False)

    print("\nHELD-OUT SUMMARY (horizon policy, both signals)")
    h = df[df.policy == "horizon"][["signal", "B_h", "mean_saving_pct", "ci_low", "ci_high",
                                    "p_holm", "significant_5pct"]]
    print(h.round(4).to_string(index=False))
    print("\nelapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
