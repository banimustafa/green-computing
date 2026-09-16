"""
Additional manuscript figures, 7 to 12.

As with the first six, every panel is drawn from a stored result file or a processed panel, and
the manifest records which. These figures cover the parts of the evidence that the original set
left as prose: the structure of the carbon signal, the structure of the workload, the shape of
the forecast error, the unconstrained deferral potential that frames the headline result, the
month-to-month variation the aggregate hides, and the trade between delay and carbon.
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
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(OUT, "figures")

MM = 1 / 25.4
C = {"s1": "#2f5d8a", "s2": "#b0602f", "oracle": "#6b6b6b", "green": "#3d7a56",
     "grey": "#9a9a9a", "accent": "#8a2f4d"}
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
    print("wrote", name, flush=True)


def fig7_grid_structure():
    d = pd.read_csv(os.path.join(PROC, "national_intensity.csv"), parse_dates=["from"])
    d = d.dropna(subset=["actual"])
    d["hour"] = d["from"].dt.hour
    d["month"] = d["from"].dt.month
    fig, ax = plt.subplots(1, 3, figsize=(190 * MM, 62 * MM))
    h = d.groupby("hour").actual.agg(["mean", "std"])
    ax[0].plot(h.index, h["mean"], color=C["green"], lw=1.3)
    ax[0].fill_between(h.index, h["mean"] - h["std"], h["mean"] + h["std"],
                       color=C["green"], alpha=0.22)
    ax[0].set_xlabel("hour of day (UTC)")
    ax[0].set_ylabel("realised intensity (gCO$_2$/kWh)")
    ax[0].set_title("(a) Diurnal shape")
    q = d.groupby("month").actual.quantile([0.1, 0.5, 0.9]).unstack()
    ax[1].plot(q.index, q[0.5], "o-", color=C["s1"], lw=1.2, ms=3.5)
    ax[1].fill_between(q.index, q[0.1], q[0.9], color=C["s1"], alpha=0.2)
    ax[1].set_xticks(range(1, 13))
    ax[1].set_xticklabels(list("JFMAMJJASOND"))
    ax[1].set_xlabel("month")
    ax[1].set_ylabel("")
    ax[1].set_title("(b) Seasonal shape")
    ax[2].hist(d.actual, bins=70, color=C["grey"], edgecolor="none")
    ax[2].axvline(d.actual.mean(), color=C["accent"], lw=1.1, ls="--",
                  label="mean %.0f" % d.actual.mean())
    ax[2].set_xlabel("realised intensity (gCO$_2$/kWh)")
    ax[2].set_ylabel("settlement periods")
    ax[2].legend(frameon=False, loc="upper right", fontsize=6.5)
    ax[2].set_title("(c) Distribution")
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig7_grid_structure", ["data/processed/national_intensity.csv"],
         "diurnal and seasonal shape and distribution of realised carbon intensity")


def fig8_workload():
    w = pd.read_csv(os.path.join(OUT, "workload_class_summary.csv"))
    vm = pd.read_csv(os.path.join(PROC, "workload_vms.csv.gz"),
                     usecols=["tenant", "vmcategory", "core_hours"])
    di = vm[vm.vmcategory == "Delay-insensitive"].groupby("tenant", observed=True).core_hours.sum()
    s = np.sort(di.to_numpy())
    lor = np.concatenate([[0], np.cumsum(s) / s.sum()])
    x = np.linspace(0, 1, len(lor))
    inf = pd.read_csv(os.path.join(OUT, "llm_inference_diurnal.csv"))

    fig, ax = plt.subplots(1, 3, figsize=(190 * MM, 66 * MM))
    b = w.set_index("category")["pct_of_core_hours"]
    ax[0].bar(range(len(b)), b.values, 0.55,
              color=[C["green"], C["s1"], C["grey"]], edgecolor="white", linewidth=0.6)
    ax[0].set_xticks(range(len(b)))
    ax[0].set_xticklabels(["delay-\ninsensitive", "interactive", "unlabelled"],
                          rotation=15, ha="right")
    ax[0].set_ylabel("share of core hours (%)")
    ax[0].set_title("(a) Where the compute sits")
    ax[1].plot(x, lor, color=C["accent"], lw=1.4)
    ax[1].plot([0, 1], [0, 1], ls=":", color=C["grey"], lw=1.0)
    ax[1].set_xlabel("cumulative share of tenants")
    ax[1].set_ylabel("cumulative core-hour share")
    ax[1].set_title("(b) Tenant concentration")
    for wl, col in (("code", C["s2"]), ("conversation", C["s1"])):
        g = inf[inf.workload == wl]
        ax[2].plot(g.hour, g.pct_of_daily_mean, "o-", color=col, lw=1.2, ms=3, label=wl)
    ax[2].axhline(100, color=C["grey"], ls=":", lw=0.8)
    ax[2].set_xlabel("hour of day (UTC)")
    ax[2].set_ylabel("% of daily mean")
    ax[2].set_title("(c) Inference demand")
    ax[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
                 fontsize=6.5)
    ax[0].set_title("(a) Where the compute sits")
    fig.subplots_adjust(bottom=0.33, wspace=0.42)
    save(fig, "fig8_workload_structure",
         ["workload_class_summary.csv", "data/processed/workload_vms.csv.gz",
          "llm_inference_diurnal.csv"],
         "core-hour shares by class, Lorenz curve of tenant concentration, inference demand shape")


def fig9_error_structure():
    d = pd.read_csv(os.path.join(PROC, "national_intensity.csv"), parse_dates=["from"])
    d = d.dropna(subset=["actual", "forecast"])
    d["err"] = d.forecast - d.actual
    fe = pd.read_csv(os.path.join(OUT, "audit_forecast_error.csv"))
    byh = fe[fe.level == "hour_utc"].astype({"stratum": int}).sort_values("stratum")
    bys = fe[fe.level == "season"]

    fig, ax = plt.subplots(1, 3, figsize=(190 * MM, 66 * MM))
    ax[0].hist(d.err, bins=80, color=C["s1"], edgecolor="none")
    ax[0].set_xlabel("forecast $-$ realised (gCO$_2$/kWh)")
    ax[0].set_ylabel("settlement periods")
    ax[0].set_title("(a) Error distribution")
    ax[0].set_xlabel("forecast $-$ realised (gCO$_2$/kWh)")
    ax[1].plot(byh.stratum, byh.MAE, "o-", color=C["s1"], lw=1.2, ms=3.5, label="MAE")
    ax[1].plot(byh.stratum, byh.RMSE, "s--", color=C["s2"], lw=1.2, ms=3.5, label="RMSE")
    ax[1].set_xlabel("hour of day (UTC)")
    ax[1].set_ylabel("gCO$_2$/kWh")
    ax[1].set_title("(b) By time of day")
    ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
                 fontsize=6.5)
    order = ["Winter", "Spring", "Summer", "Autumn"]
    ss = bys.set_index("stratum").reindex(order)
    ax2 = ax[2]
    ax2.bar(np.arange(4) - 0.18, ss.MAE, 0.34, color=C["s1"], label="MAE", edgecolor="white")
    ax2.bar(np.arange(4) + 0.18, ss.MAPE_pct, 0.34, color=C["accent"], label="MAPE (%)",
            edgecolor="white")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(order)
    ax2.set_ylabel("gCO$_2$/kWh, or per cent")
    ax2.set_title("(c) By season")
    ax2.set_xticklabels(order, rotation=20, ha="right")
    ax2.set_ylim(0, float(max(ss.MAE.max(), ss.MAPE_pct.max())) * 1.35)
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
               fontsize=6.5)
    fig.subplots_adjust(bottom=0.36, wspace=0.38)
    save(fig, "fig9_forecast_error_structure",
         ["data/processed/national_intensity.csv", "audit_forecast_error.csv"],
         "error distribution, error by hour, absolute and relative error by season")


def fig10_potential():
    d = pd.read_csv(os.path.join(OUT, "audit_deferral_potential.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(165 * MM, 72 * MM))
    ax[0].plot(d.deferral_budget_h, d.saving_oracle_pct, "s--", color=C["oracle"], lw=1.2, ms=4,
               label="perfect foresight")
    ax[0].plot(d.deferral_budget_h, d.saving_forecast_guided_pct, "o-", color=C["s1"], lw=1.3,
               ms=4, label="published forecast")
    ax[0].fill_between(d.deferral_budget_h, d.saving_forecast_guided_pct, d.saving_oracle_pct,
                       color=C["oracle"], alpha=0.15, label="price of uncertainty")
    ax[0].set_xlabel("deferral budget (hours)")
    ax[0].set_ylabel("carbon saving (%)")
    ax[0].set_title("(a) Unconstrained placement")
    ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
                 fontsize=6.5)
    ax[1].plot(d.deferral_budget_h, d.realised_fraction_of_oracle_pct, "o-", color=C["green"],
               lw=1.3, ms=4, label="share of foresight realised")
    ax[1].plot(d.deferral_budget_h, d.pct_jobs_worse_than_immediate, "^-", color=C["accent"],
               lw=1.2, ms=4, label="work finishing dirtier than immediate")
    ax[1].set_xlabel("deferral budget (hours)")
    ax[1].set_ylabel("per cent")
    ax[1].set_ylim(0, 100)
    ax[1].set_title("(b) What uncertainty costs")
    ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
                 fontsize=6.5)
    fig.subplots_adjust(bottom=0.34, wspace=0.28)
    save(fig, "fig10_deferral_potential", ["audit_deferral_potential.csv"],
         "oracle and forecast-guided saving for unconstrained placement, and the share of work made worse")


def fig11_monthly():
    m = pd.read_csv(os.path.join(OUT, "phase5_monthly.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(190 * MM, 60 * MM))
    for i, B in enumerate((12, 24)):
        for sig, col, mk in (("S1_published", C["s1"], "o"), ("S2_day_ahead", C["s2"], "s")):
            g = m[(m.signal == sig) & (m.B_h == B) & (m.policy == "horizon")].sort_values("month")
            ax[i].plot(range(len(g)), g.saving_pct, mk + "-", color=col, lw=1.2, ms=3.5,
                       label="published" if sig == "S1_published" else "day-ahead")
        o = m[(m.signal == "S1_published") & (m.B_h == B) & (m.policy == "oracle")].sort_values("month")
        ax[i].plot(range(len(o)), o.saving_pct, ":", color=C["oracle"], lw=1.1,
                   label="perfect foresight")
        ax[i].axhline(0, color=C["grey"], lw=0.8)
        ax[i].set_xticks(range(len(g)))
        ax[i].set_xticklabels([x[2:] for x in g.month], rotation=60, fontsize=6.5)
        ax[i].set_ylabel("carbon saving (%)")
        ax[i].set_title("(%s) %d-hour deferral budget" % ("ab"[i], B))
        ax[i].legend(frameon=False, loc="upper left", ncol=1)
    save(fig, "fig11_monthly_held_out", ["phase5_monthly.csv"],
         "saving in each of the twelve held-out months, both signals, two budgets")


def fig12_delay_trade():
    h = pd.read_csv(os.path.join(OUT, "phase5_held_out.csv"))
    h = h[h.signal == "S1_published"]
    fig, ax = plt.subplots(figsize=(120 * MM, 76 * MM))
    marks = {"percentile": "v", "threshold": "s", "lyapunov": "D", "capacitycurve": "P",
             "horizon": "o", "oracle": "*"}
    cols = {"percentile": C["grey"], "threshold": C["s2"], "lyapunov": C["accent"],
            "capacitycurve": "#5b8c5a", "horizon": C["green"], "oracle": C["oracle"]}
    for pol, g in h.groupby("policy"):
        g = g.sort_values("B_h")
        ax.plot(g.mean_delay_h, g.mean_saving_pct, marks[pol] + "-", color=cols[pol],
                lw=1.1, ms=5 if pol != "oracle" else 8, label={"lyapunov": "drift-plus-penalty", "horizon": "receding horizon",
                       "percentile": "rolling percentile", "threshold": "fixed threshold",
                       "capacitycurve": "daily capacity curve",
                       "oracle": "perfect foresight"}[pol])
        for _, r in g.iterrows():
            ax.annotate("%dh" % r.B_h, (r.mean_delay_h, r.mean_saving_pct),
                        textcoords="offset points", xytext=(4, -7), fontsize=6, color=cols[pol])
    ax.set_xlabel("mean delay imposed on deferrable work (hours)")
    ax.set_ylabel("carbon saving against baseline (%)")
    # Five series leave no clear interior space, so the legend goes below the axes.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              fontsize=6.5, handlelength=2.0)
    fig.subplots_adjust(bottom=0.30)
    save(fig, "fig12_delay_carbon_trade", ["phase5_held_out.csv"],
         "saving against mean delay for every policy and budget, published signal")


def main():
    fig7_grid_structure()
    fig8_workload()
    fig9_error_structure()
    fig10_potential()
    fig11_monthly()
    fig12_delay_trade()
    old = pd.read_csv(os.path.join(OUT, "figure_manifest.csv"))
    pd.concat([old, pd.DataFrame(manifest)], ignore_index=True).to_csv(
        os.path.join(OUT, "figure_manifest.csv"), index=False)
    print("manifest updated, %d new figures" % len(manifest))


if __name__ == "__main__":
    main()
