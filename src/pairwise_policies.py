"""
Pairwise policy comparison.

The held-out evaluation tests each policy against the carbon-agnostic baseline. It does not
establish an ordering among the policies themselves, yet the manuscript compares them directly.
This script closes that gap: for every pair of policies, within a signal and a deferral budget,
it tests the twelve paired monthly differences with a two-sided Wilcoxon signed rank test and
corrects across all pairs in that stratum by the Holm-Bonferroni procedure.

A difference that does not survive the correction is reported as indistinguishable, and the
manuscript's language is required to follow.
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
N_BOOT = 10000
RNG = np.random.default_rng(20260913)
ORDER = ["percentile", "threshold", "lyapunov", "capacitycurve", "horizon", "oracle"]


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    m = pd.read_csv(os.path.join(OUT, "phase5_monthly.csv"))
    rows = []
    for (sig, B), g in m.groupby(["signal", "B_h"]):
        piv = g.pivot_table(index="month", columns="policy", values="saving_pct")
        pols = [p for p in ORDER if p in piv.columns]
        block = []
        for a, b in combinations(pols, 2):
            d = (piv[a] - piv[b]).dropna().to_numpy()
            boot = np.array([RNG.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)])
            try:
                p = float(wilcoxon(d, alternative="two-sided").pvalue)
            except ValueError:
                p = 1.0
            block.append({"signal": sig, "B_h": B, "policy_a": a, "policy_b": b,
                          "mean_difference_pp": float(d.mean()),
                          "ci_low": float(np.percentile(boot, 2.5)),
                          "ci_high": float(np.percentile(boot, 97.5)),
                          "months_a_better": int((d > 0).sum()), "n_months": len(d),
                          "p_wilcoxon": p})
        adj = holm([r["p_wilcoxon"] for r in block])
        for r, a in zip(block, adj):
            r["p_holm"] = float(a)
            r["distinguishable_5pct"] = bool(a < 0.05)
        rows.extend(block)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phase7_pairwise_policies.csv"), index=False)

    s1 = df[df.signal == "S1_published"]
    print("Pairwise comparisons, published forecast. Positive difference favours policy A.\n")
    for B in (4, 12, 24):
        sub = s1[s1.B_h == B]
        print("deferral budget %d h" % B)
        for _, r in sub.iterrows():
            print("  %-13s vs %-13s %+6.2f pp [%+5.2f, %+5.2f]  %2d/%d months  p_holm %.4f  %s"
                  % (r.policy_a, r.policy_b, r.mean_difference_pp, r.ci_low, r.ci_high,
                     r.months_a_better, r.n_months, r.p_holm,
                     "distinguishable" if r.distinguishable_5pct else "INDISTINGUISHABLE"))
        print()
    n = len(s1)
    print("published signal: %d of %d pairwise differences distinguishable after correction"
          % (int(s1.distinguishable_5pct.sum()), n))


if __name__ == "__main__":
    main()
