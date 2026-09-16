"""Assemble and execute the Phase 2 notebook: workload construction and characterisation."""
import os, nbformat as nbf
from nbclient import NotebookClient
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks", "phase2_workload_model.ipynb")
md = lambda t: nbf.v4.new_markdown_cell(t); code = lambda t: nbf.v4.new_code_cell(t)
cells = [
md("""# Phase 2 — Workload model

**Study:** Fair and carbon-aware spatiotemporal scheduling of AI workloads in data centres under forecast uncertainty

Phase 1 established that a carbon saving survives forecast error. Phase 2 builds the workload the
scheduler acts on, and records a change to the design that the data forced."""),
code("""import os, sys, time, platform
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd())=="notebooks" else os.getcwd()
PROC, OUT = os.path.join(ROOT,"data","processed"), os.path.join(ROOT,"outputs")
FIG = os.path.join(OUT,"figures"); os.makedirs(FIG, exist_ok=True); T0=time.time()
print(platform.platform(), "| python", platform.python_version(), "| cores", os.cpu_count())"""),
md("""## 1. Delay sensitivity is labelled by the operator, not assumed by us

The Azure 2019 VM trace assigns every virtual machine a category: Delay-insensitive, Interactive, or
Unknown. The deferrable and non-deferrable classes in this study inherit that label directly, which
removes a modelling assumption that a reviewer would otherwise be entitled to challenge."""),
code("""ch = pd.read_csv(os.path.join(OUT,"workload_class_summary.csv"))
display(ch.round(2))"""),
md("""## 2. What the trace showed, and how the design changed

The delay-insensitive machines are not job-shaped. Their median observed lifetime sits at the
boundary of the 30-day observation window and none runs for less than an hour, so they are
long-running elastic services rather than discrete jobs waiting to be shifted. They nonetheless carry
the majority of core hours.

The scheduling model was therefore changed. Deferrable work is treated as **capacity with a delivery
deadline and an adjustable rate**, not as jobs to be moved whole: total core hours must be delivered
within a horizon, and the rate within that horizon is what the scheduler controls. This is both what
the evidence supports and what carbon-aware operation looks like in practice.

The alternative would have been to keep the original model and force the trace to fit it. That is
recorded here as a rejected option."""),
code("""tc = pd.read_csv(os.path.join(OUT,"workload_tenant_concentration.csv"))
display(tc.round(3))
print("A Gini coefficient of %.3f on tenant core hours means the deferral burden can be concentrated"
      % tc.gini_core_hours[0])
print("on very few tenants. Fairness is therefore a live constraint on this trace, not a decorative one:")
print("the largest 1%% of tenants hold %.1f%% of deferrable core hours, and %d tenants hold half of them."
      % (tc.top1pct_tenants_share_pct[0], tc.tenants_for_50pct_core_hours[0]))"""),
md("""## 3. Non-deferrable inference load

The one-week Azure LLM inference traces supply the interactive class: 44.1 million requests with
prompt and generated token counts, aggregated to the half-hourly resolution of the grid series. This
load must be served on arrival, and its diurnal shape determines the residual capacity available for
shifting in each period."""),
code("""s = pd.read_csv(os.path.join(OUT,"llm_inference_summary.csv"))
display(s.round(2))
di = pd.read_csv(os.path.join(OUT,"llm_inference_diurnal.csv"))
ap = pd.read_csv(os.path.join(OUT,"workload_arrival_profile.csv"))

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
for w, c in [("code","#b05a3c"), ("conversation","#4a6fa5")]:
    g = di[di.workload==w]; ax[0].plot(g.hour, g.pct_of_daily_mean, "o-", color=c, label=w)
ax[0].axhline(100, color="grey", lw=.8, ls=":"); ax[0].set_xlabel("hour of day (UTC)")
ax[0].set_ylabel("% of daily mean requests"); ax[0].set_title("(a) Inference demand shape")
ax[0].legend(frameon=False, fontsize=8)
for cat, c in [("Delay-insensitive","#3d7a56"), ("Interactive","#4a6fa5")]:
    g = ap[ap.category==cat]; ax[1].plot(g.hour_of_day, g.pct_arrivals, "o-", color=c, label=cat)
ax[1].set_xlabel("hour of trace day"); ax[1].set_ylabel("% of arrivals")
ax[1].set_title("(b) VM arrivals by class"); ax[1].legend(frameon=False, fontsize=8)
b = ch.set_index("category")["pct_of_core_hours"]
ax[2].bar(range(len(b)), b.values, color=["#3d7a56","#4a6fa5","#999999"])
ax[2].set_xticks(range(len(b))); ax[2].set_xticklabels(b.index, rotation=20, ha="right", fontsize=8)
ax[2].set_ylabel("% of total core hours"); ax[2].set_title("(c) Where the compute sits")
for a in ax: a.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig3_workload_model.png"), dpi=200); plt.show()"""),
md("""## 4. Analysis plan frozen

Hypotheses, policies, metrics, fairness definitions, the forecast-signal bracket, and the held-out
period are fixed in `docs/analysis_plan_frozen.md` before any policy is scored. Months 1 to 24 are
available for design; months 25 to 36 are scored once."""),
code("""p = os.path.join(ROOT,"docs","analysis_plan_frozen.md")
print("frozen plan:", os.path.exists(p), "| %d words" % len(open(p).read().split()))
print("phase 2 notebook wall-clock: %.1f s" % (time.time()-T0))"""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name":"python3","display_name":"Python 3","language":"python"}
NotebookClient(nb, timeout=1200, kernel_name="python3",
               resources={"metadata":{"path": os.path.join(ROOT,"notebooks")}}).execute()
nbf.write(nb, NB); print("written:", NB)
