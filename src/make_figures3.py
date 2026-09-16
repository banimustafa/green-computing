"""Figures 13 and 14: workload-shape sensitivity, and tenant budget sensitivity."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); FIG = os.path.join(OUT, "figures")
MM = 1/25.4
C = {"s1":"#2f5d8a","s2":"#b0602f","oracle":"#6b6b6b","green":"#3d7a56","grey":"#9a9a9a","accent":"#8a2f4d"}
plt.rcParams.update({"font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,"legend.fontsize":7,
                     "xtick.labelsize":7.5,"ytick.labelsize":7.5,"axes.spines.top":False,
                     "axes.spines.right":False,"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight"})
man = []
def save(fig,name,src,note):
    for e in ("png","pdf"): fig.savefig(os.path.join(FIG,"%s.%s"%(name,e)))
    plt.close(fig); man.append({"figure":name,"sources":"; ".join(src),"content":note}); print("wrote",name)

def fig13():
    a = pd.read_csv(os.path.join(OUT,"phase6_ai_workload.csv"))
    fig, ax = plt.subplots(1,2, figsize=(160*MM,58*MM))
    pols = ["threshold","horizon","oracle"]
    lab = ["fixed\nthreshold","receding\nhorizon","perfect\nforesight"]
    for i,B in enumerate((12,24)):
        w = 0.36
        for j,(wl,col) in enumerate((("trace",C["s1"]),("ai_training",C["accent"]))):
            s = a[(a.B_h==B)&(a.workload==wl)].set_index("policy").reindex(pols)
            ax[i].bar(np.arange(3)+(j-0.5)*w, s.saving_pct, w, color=col, edgecolor="white",
                      linewidth=0.6, label="elastic services (trace)" if wl=="trace" else "training jobs (synthetic)")
        ax[i].set_xticks(range(3)); ax[i].set_xticklabels(lab)
        ax[i].set_ylabel("carbon saving against baseline (%)")
        ax[i].set_title("(%s) %d-hour deferral budget"%("ab"[i],B))
        if i==0: ax[i].legend(frameon=False, loc="upper left")
    save(fig,"fig13_workload_sensitivity",["phase6_ai_workload.csv"],
         "policy savings under trace-derived and synthetic training arrivals at matched mean load")

def fig14():
    t = pd.read_csv(os.path.join(OUT,"phase6_tenant_sensitivity.csv"))
    x = np.arange(len(t)); w = 0.27
    fig, ax = plt.subplots(1,2, figsize=(175*MM,72*MM))
    ax[0].bar(x-w, t.saving_pct, w, color=C["grey"], edgecolor="white", linewidth=.6, label="as scheduled")
    ax[0].bar(x, t.saving_with_allocation_lever, w, color=C["s2"], edgecolor="white", linewidth=.6, label="allocation lever")
    ax[0].bar(x+w, t.saving_with_flexibility_grant, w, color=C["green"], edgecolor="white", linewidth=.6, label="flexibility grant")
    ax[0].set_xticks(x); ax[0].set_xticklabels(t["mix"], rotation=18, ha="right")
    ax[0].set_ylabel("carbon saving (%)")
    ax[0].set_title("(a) Saving under five budget distributions")
    ax[0].set_ylim(0, float(t[["saving_pct","saving_with_allocation_lever",
                               "saving_with_flexibility_grant"]].to_numpy().max())*1.15)
    ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5,-0.20), ncol=3, fontsize=6.5)
    ax[1].bar(x-w/2, t.spread, w, color=C["grey"], edgecolor="white", linewidth=.6, label="as scheduled")
    ax[1].bar(x+w/2, t.spread_with_flexibility_grant, w, color=C["green"], edgecolor="white", linewidth=.6, label="flexibility grant")
    ax[1].set_xticks(x); ax[1].set_xticklabels(t["mix"], rotation=18, ha="right")
    ax[1].set_ylabel("spread between best and worst\nserved tenant (gCO$_2$/kWh)")
    ax[1].set_title("(b) Tenant inequality under the same variation")
    ax[1].set_ylim(0, float(t[["spread","spread_with_flexibility_grant"]].to_numpy().max())*1.15)
    ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5,-0.20), ncol=2, fontsize=6.5)
    fig.subplots_adjust(bottom=0.30, wspace=0.3)
    save(fig,"fig14_tenant_sensitivity",["phase6_tenant_sensitivity.csv"],
         "saving and tenant inequality under five deferral-budget distributions, with both fairness levers")

fig13(); fig14()
old = pd.read_csv(os.path.join(OUT,"figure_manifest.csv"))
pd.concat([old,pd.DataFrame(man)],ignore_index=True).drop_duplicates("figure",keep="last").to_csv(
    os.path.join(OUT,"figure_manifest.csv"), index=False)
print("manifest updated")


def fig15_delay_and_flexibility():
    """Figures 10 and 11 combined.

    The two panels answer the same question from opposite directions: what delay buys in carbon,
    and what granting the flexibility to be delayed buys. Presented together, the pairing is
    visible; presented apart, it has to be asserted in the text."""
    h = pd.read_csv(os.path.join(OUT, "phase5_held_out.csv"))
    h = h[h.signal == "S1_published"]
    f = pd.read_csv(os.path.join(OUT, "phase3c_flexibility.csv")).sort_values("grant_h")

    fig, ax = plt.subplots(1, 2, figsize=(190*MM, 78*MM))
    marks = {"percentile": "v", "threshold": "s", "lyapunov": "D", "capacitycurve": "P",
             "horizon": "o", "oracle": "*"}
    cols = {"percentile": C["grey"], "threshold": C["s2"], "lyapunov": C["accent"],
            "capacitycurve": "#5b8c5a", "horizon": C["green"], "oracle": C["oracle"]}
    names = {"percentile": "rolling percentile", "threshold": "fixed threshold",
             "lyapunov": "drift-plus-penalty", "capacitycurve": "daily capacity curve",
             "horizon": "receding horizon", "oracle": "perfect foresight"}
    for pol, g in h.groupby("policy"):
        g = g.sort_values("B_h")
        ax[0].plot(g.mean_delay_h, g.mean_saving_pct, marks[pol] + "-", color=cols[pol],
                   lw=1.1, ms=5 if pol != "oracle" else 8, label=names[pol])
        for _, r in g.iterrows():
            ax[0].annotate("%dh" % r.B_h, (r.mean_delay_h, r.mean_saving_pct),
                           textcoords="offset points", xytext=(4, -7), fontsize=6,
                           color=cols[pol])
    ax[0].set_xlabel("mean delay imposed on deferrable work (hours)")
    ax[0].set_ylabel("carbon saving against baseline (%)")
    ax[0].set_title("(a) What delay buys")
    ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
                 fontsize=6.5, handlelength=2.0)

    lab = ["no grant", "8 h", "12 h", "24 h"]
    x = np.arange(len(f))
    ax[1].bar(x, f.saving_pct, 0.5, color=C["green"], edgecolor="white", linewidth=0.6)
    ax[1].set_ylabel("carbon saving against baseline (%)", color=C["green"])
    ax[1].tick_params(axis="y", labelcolor=C["green"])
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab)
    ax[1].set_xlabel("deferral budget granted to the least flexible tenants")
    ax[1].set_title("(b) What granting flexibility buys")
    ax2 = ax[1].twinx()
    ax2.plot(x, f.tenant_spread_gCO2_kWh, "o-", color=C["accent"], lw=1.2, ms=4)
    ax2.set_ylabel("spread between best and worst\nserved tenant (gCO$_2$/kWh)", color=C["accent"])
    ax2.tick_params(axis="y", labelcolor=C["accent"])
    ax2.spines["right"].set_visible(True)
    fig.subplots_adjust(bottom=0.32, wspace=0.42)
    save(fig, "fig15_delay_and_flexibility",
         ["phase5_held_out.csv", "phase3c_flexibility.csv"],
         "carbon saving against mean delay for every policy and budget, and saving and tenant spread against the granted deferral budget")


if __name__ == "__main__":
    fig15_delay_and_flexibility()
    old = pd.read_csv(os.path.join(OUT, "figure_manifest.csv"))
    pd.concat([old, pd.DataFrame(man)], ignore_index=True).drop_duplicates(
        "figure", keep="last").to_csv(os.path.join(OUT, "figure_manifest.csv"), index=False)
