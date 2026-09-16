"""
Per-month held-out savings.

The held-out evaluation reports means and bootstrap intervals over the twelve paired months.
This script writes the underlying monthly series so that the month-to-month variation, which
the aggregate hides, can be shown directly. Parameters are the same frozen values used in the
scoring pass; nothing is re-tuned here.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_policies import load_shapes, simulate, tenant_profile, PROC, OUT, DESIGN_END  # noqa: E402

FROZEN_V = 20.0


def main():
    d = pd.read_csv(os.path.join(PROC, "national_intensity_signals.csv"), parse_dates=["from"])
    d = d.dropna(subset=["actual", "forecast_published", "forecast_s2"])
    design_median = float(d.loc[d["from"] < pd.Timestamp(DESIGN_END, tz="UTC"),
                                "forecast_published"].median())
    d = d[d["from"] >= pd.Timestamp(DESIGN_END, tz="UTC")].reset_index(drop=True)
    shares, budgets, _ = tenant_profile()

    rows = []
    for label, col in (("S1_published", "forecast_published"), ("S2_day_ahead", "forecast_s2")):
        g = d.rename(columns={col: "forecast"})[["from", "actual", "forecast"]]
        n, arr = load_shapes(g, 0.85)
        for B in (4, 12, 24):
            base = simulate("agnostic", g, n, arr, B, {}, shares, budgets, monthly=True)
            for pol, pr in [("threshold", {"thr": design_median}),
                            ("percentile", {"p": 30}),
                            ("capacitycurve", {"q": 50}),
                            ("lyapunov", {"V": FROZEN_V, "ref": design_median}),
                            ("horizon", {}), ("oracle", {})]:
                r = simulate(pol, g, n, arr, B, pr, shares, budgets, monthly=True)
                for m in sorted(set(base["monthly"]) & set(r["monthly"])):
                    b, v = base["monthly"][m], r["monthly"][m]
                    rows.append({"signal": label, "B_h": B, "policy": pol, "month": m,
                                 "baseline_intensity": b, "policy_intensity": v,
                                 "saving_pct": (b - v) / b * 100})
            print("%s B=%d done" % (label, B), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "phase5_monthly.csv"), index=False)
    s = out[(out.signal == "S1_published") & (out.policy == "horizon")]
    print("\nmonthly saving, published forecast, receding horizon")
    print(s.pivot_table(index="month", columns="B_h", values="saving_pct").round(2).to_string())


if __name__ == "__main__":
    main()
