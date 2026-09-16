"""
Phase 7 - Manuscript figures.

Every figure is drawn from a stored result CSV, never from a value typed into this script.
A manifest records which file and which column each panel draws on, so the consistency audit
in the final phase can verify that the plotted numbers, the tables, and the manuscript text
all trace to the same source.

Figures are sized for a two-column journal page: 90 mm for single-column panels and 190 mm
for full-width ones, at 300 dots per inch, with a consistent palette and no decoration that
does not carry information.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

MM = 1 / 25.4
C = {"s1": "#2f5d8a", "s2": "#b0602f", "oracle": "#6b6b6b",
     "green": "#3d7a56", "grey": "#9a9a9a", "accent": "#8a2f4d"}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})

manifest = []


def save(fig, name, sources, note):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)))
    plt.close(fig)
    manifest.append({"figure": name, "sources": "; ".join(sources), "content": note})
    print("wrote %s" % name, flush=True)


# ---------------------------------------------------------------- Figure 1: signals
def fig_signals():
    a = pd.read_csv(os.path.join(OUT, "phase3e_signal_accuracy.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM))
    order = ["S1 published", "S2 day-ahead", "24 h persistence"]
    cols = [C["s1"], C["s2"], C["grey"]]
    x = np.arange(len(order))
    for i, split in enumerate(("design", "held-out")):
        d = a[a.split == split].set_index("signal").reindex(order)
        ax[0].bar(x + (i - 0.5) * 0.36, d.MAE, 0.34,
                  color=cols, alpha=1.0 if split == "held-out" else 0.55,
                  edgecolor="white", linewidth=0.6,
                  label="held-out" if split == "held-out" else "design")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(order)
    ax[0].set_ylabel("mean absolute error (gCO$_2$/kWh)")
    ax[0].set_title("(a) Forecast accuracy")
    ax[0].legend(frameon=False, loc="upper left")

    d = a[a.split == "held-out"].set_index("signal").reindex(order)
    ax[1].bar(x, d["corr"], 0.5, color=cols, edgecolor="white", linewidth=0.6)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(order)
    ax[1].set_ylabel("correlation with realised intensity")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("(b) Held-out correlation")
    save(fig, "fig1_signal_accuracy", ["phase3e_signal_accuracy.csv"],
         "MAE by signal and split; held-out correlation by signal")


# ------------------------------------------------- Figure 2: held-out saving and oracle share
def fig_held_out():
    d = pd.read_csv(os.path.join(OUT, "phase5_held_out.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(190 * MM, 72 * MM))
    for sig, col, mk in (("S1_published", C["s1"], "o"), ("S2_day_ahead", C["s2"], "s")):
        h = d[(d.signal == sig) & (d.policy == "horizon")].sort_values("B_h")
        ax[0].errorbar(h.B_h, h.mean_saving_pct,
                       yerr=[h.mean_saving_pct - h.ci_low, h.ci_high - h.mean_saving_pct],
                       fmt=mk + "-", color=col, capsize=2.5, lw=1.2, ms=4,
                       label="planner, %s" % ("published forecast" if sig == "S1_published" else "day-ahead forecast"))
        if sig == "S1_published":
            # Perfect foresight is scored on realised intensity and is therefore the same
            # curve under either forecast, so it is drawn once.
            o = d[(d.signal == sig) & (d.policy == "oracle")].sort_values("B_h")
            ax[0].plot(o.B_h, o.mean_saving_pct, ":", color=C["oracle"], lw=1.1,
                       label="perfect foresight")
        ax[1].plot(h.B_h, h.pct_of_oracle, mk + "-", color=col, lw=1.2, ms=4,
                   label="published" if sig == "S1_published" else "day-ahead")
    ax[0].set_xlabel("deferral budget (hours)")
    ax[0].set_ylabel("carbon saving against baseline (%)")
    ax[0].set_title("(a) Held-out saving, twelve paired months")
    # The legend sat over the interval whiskers at the twelve-hour point, so it is placed below
    # the axes where nothing is plotted.
    ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
                 fontsize=6.5, handlelength=2.2)
    ax[0].set_xticks([4, 12, 24])
    ax[1].set_xlabel("deferral budget (hours)")
    ax[1].set_ylabel("share of achievable saving realised (%)")
    ax[1].set_ylim(50, 102)
    ax[1].set_xticks([4, 12, 24])
    ax[1].set_title("(b) Share of perfect foresight captured")
    ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                 fontsize=6.5, handlelength=2.2)
    fig.subplots_adjust(bottom=0.34, wspace=0.3)
    save(fig, "fig2_held_out_saving", ["phase5_held_out.csv"],
         "mean saving with bootstrap intervals and share of oracle, horizon policy, both signals")


# ------------------------------------------------------------- Figure 3: policy comparison
def fig_policies():
    d = pd.read_csv(os.path.join(OUT, "phase5_held_out.csv"))
    d = d[d.signal == "S1_published"]
    pols = ["percentile", "threshold", "lyapunov", "capacitycurve", "horizon", "oracle"]
    labels = ["rolling\npercentile", "fixed\nthreshold", "drift-plus-\npenalty",
              "daily capacity\ncurve", "receding\nhorizon", "perfect\nforesight"]
    fig, ax = plt.subplots(figsize=(135 * MM, 64 * MM))
    width = 0.26
    for i, B in enumerate((4, 12, 24)):
        sub = d[d.B_h == B].set_index("policy").reindex(pols)
        ax.bar(np.arange(len(pols)) + (i - 1) * width, sub.mean_saving_pct, width,
               yerr=[sub.mean_saving_pct - sub.ci_low, sub.ci_high - sub.mean_saving_pct],
               capsize=2, color=[C["s1"], C["s1"], C["s1"], "#5b8c5a", C["green"], C["grey"]],
               alpha=0.45 + 0.275 * i, edgecolor="white", linewidth=0.5,
               label="%d h budget" % B)
    ax.set_xticks(np.arange(len(pols)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("carbon saving against baseline (%)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig3_policy_comparison", ["phase5_held_out.csv"],
         "held-out saving by policy and budget, published signal")


# ----------------------------------------------------- Figure 4: energy bracket and utilisation
def fig_energy_utilisation():
    e = pd.read_csv(os.path.join(OUT, "phase3d_energy_bracket.csv"))
    u = pd.read_csv(os.path.join(OUT, "phase3c_utilisation.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM))
    for B, col, mk in ((12, C["s2"], "s"), (24, C["s1"], "o")):
        s = e[e.B_h == B].sort_values("idle_fraction")
        ax[0].plot(s.idle_fraction, s.facility_saving_pct, mk + "-", color=col, lw=1.2, ms=4,
                   label="%d h budget" % B)
    ax[0].set_xlabel("idle power as a fraction of peak")
    ax[0].set_ylabel("facility-level carbon saving (%)")
    ax[0].set_title("(a) Saving against energy proportionality")
    ax[0].legend(frameon=False)
    ax[1].plot(u.utilisation, u.saving_pct, "o-", color=C["green"], lw=1.2, ms=4)
    ax[1].set_xlabel("mean utilisation")
    ax[1].set_ylabel("carbon saving against baseline (%)")
    ax[1].set_title("(b) Saving against headroom")
    save(fig, "fig4_energy_and_headroom",
         ["phase3d_energy_bracket.csv", "phase3c_utilisation.csv"],
         "facility saving against idle fraction; saving against mean utilisation")


# ----------------------------------------------------------- Figure 5: flexibility grant
def fig_flexibility():
    f = pd.read_csv(os.path.join(OUT, "phase3c_flexibility.csv")).sort_values("grant_h")
    lab = ["no grant", "8 h", "12 h", "24 h"]
    x = np.arange(len(f))
    fig, ax1 = plt.subplots(figsize=(120 * MM, 62 * MM))
    ax1.bar(x, f.saving_pct, 0.5, color=C["green"], edgecolor="white", linewidth=0.6)
    ax1.set_ylabel("carbon saving against baseline (%)", color=C["green"])
    ax1.tick_params(axis="y", labelcolor=C["green"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(lab)
    ax1.set_xlabel("deferral budget granted to the least flexible tenants")
    ax2 = ax1.twinx()
    ax2.plot(x, f.tenant_spread_gCO2_kWh, "o-", color=C["accent"], lw=1.2, ms=4)
    ax2.set_ylabel("spread between best and worst\nserved tenant (gCO$_2$/kWh)", color=C["accent"])
    ax2.tick_params(axis="y", labelcolor=C["accent"])
    ax2.spines["right"].set_visible(True)
    save(fig, "fig5_flexibility_grant", ["phase3c_flexibility.csv"],
         "saving and tenant intensity spread against granted deferral budget")


# --------------------------------------------------------------- Figure 6: spatial results
def fig_spatial():
    hi = pd.read_csv(os.path.join(OUT, "phase4_spatial_u85.csv"))
    lo = pd.read_csv(os.path.join(OUT, "phase4_spatial_u45.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM))
    modes = ["agnostic", "temporal", "spatial", "spatiotemporal"]
    lab = ["baseline", "temporal\nonly", "spatial\nonly", "spatial and\ntemporal"]
    sub = hi[hi.exposure_cap.isna()].set_index("mode").reindex(modes)
    ax[0].bar(np.arange(len(modes)), sub.saving_pct, 0.55,
              color=[C["grey"], C["s1"], C["s2"], C["green"]], edgecolor="white", linewidth=0.6)
    ax[0].set_xticks(np.arange(len(modes)))
    ax[0].set_xticklabels(lab)
    ax[0].set_ylabel("carbon saving on the forecast field (%)")
    ax[0].set_title("(a) Placement in time and space")
    for src, col, mk, name in ((lo, C["accent"], "o", "spare capacity (0.45)"),
                               (hi, C["s1"], "s", "busy fleet (0.85)")):
        s = src[(src["mode"] == "spatiotemporal")].copy()
        s["cap"] = s.exposure_cap.fillna(np.inf)
        s = s.sort_values("cap")
        ax[1].plot(range(len(s)), s.saving_pct, mk + "-", color=col, lw=1.2, ms=4, label=name)
        ticks = ["%.1f" % c if np.isfinite(c) else "none" for c in s.cap]
    ax[1].set_xticks(range(len(ticks)))
    ax[1].set_xticklabels(ticks)
    ax[1].set_xlabel("regional exposure cap (multiple of an equal share)")
    ax[1].set_ylabel("carbon saving on the forecast field (%)")
    ax[1].set_title("(b) Cost of spreading load across regions")
    ax[1].legend(frameon=False, loc="lower right")
    save(fig, "fig6_spatial_and_exposure",
         ["phase4_spatial_u85.csv", "phase4_spatial_u45.csv"],
         "saving by placement mode; saving against cumulative exposure cap at two utilisations")


def main():
    fig_signals()
    fig_held_out()
    fig_policies()
    fig_energy_utilisation()
    fig_flexibility()
    fig_spatial()
    pd.DataFrame(manifest).to_csv(os.path.join(OUT, "figure_manifest.csv"), index=False)
    print("\nmanifest written, %d figures" % len(manifest))


if __name__ == "__main__":
    main()
