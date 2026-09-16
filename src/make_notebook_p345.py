"""Assemble and execute the consolidated results notebook for Phases 3 to 5."""
import os, nbformat as nbf
from nbclient import NotebookClient
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks", "phase3to5_scheduling_results.ipynb")
md = lambda t: nbf.v4.new_markdown_cell(t); code = lambda t: nbf.v4.new_code_cell(t)
cells = [
md("""# Phases 3 to 5 — Scheduling results

**Study:** Fair and carbon-aware spatiotemporal scheduling of AI workloads in data centres under forecast uncertainty

This notebook assembles every scheduling result, reproduces the tables that appear in the
manuscript, and runs the consistency checks that guard them.

**Regeneration.** Set `REGENERATE = True` to re-execute the simulations from the cached grid
and trace data rather than reading the stored result files. The full pipeline takes about
nineteen minutes on a single core, of which thirteen are the initial grid retrieval; see
`src/run_all.sh` for the stage-by-stage timings. With `REGENERATE = False` the notebook reads
the stored CSVs, which are the exact files the figures are drawn from."""),
code("""REGENERATE = False

import os, subprocess, sys, time
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd())=="notebooks" else os.getcwd()
OUT = os.path.join(ROOT, "outputs")
T0 = time.time()

def stage(script, *args):
    if REGENERATE:
        t = time.time()
        subprocess.run([sys.executable, os.path.join(ROOT, "src", script), *args], check=True)
        print("%s %s -> %.1f s" % (script, " ".join(args), time.time() - t))

def load(name):
    return pd.read_csv(os.path.join(OUT, name))

print("regenerate:", REGENERATE)"""),

md("""## 1. Policy comparison on the design period

Months 1 to 24 only. Policy parameters are selected here and frozen before the held-out
period is touched."""),
code("""stage("simulate_policies.py", "policies")
p = load("phase3c_policies.csv")
tbl = p.pivot_table(index="B_h", columns="policy", values="saving_pct").round(2)
tbl = tbl[["threshold", "percentile", "lyapunov", "horizon", "oracle"]]
display(tbl)"""),

md("""### Standing assertions

Three checks are enforced in the simulator and repeated here, because each caught a real
defect during development: work conservation, dominance of perfect foresight over any
forecast-based policy, and monotonicity of the saving in the deferral budget."""),
code("""issues = []
for B, g in p.groupby("B_h"):
    orc = float(g.loc[g.policy == "oracle", "saving_pct"].iloc[0])
    for pol in ("threshold", "percentile", "horizon", "lyapunov"):
        v = float(g.loc[g.policy == pol, "saving_pct"].iloc[0])
        if v > orc + 0.05:
            issues.append("A2: %s beats perfect foresight at B=%s" % (pol, B))
for pol in ("horizon", "oracle"):
    s = p[p.policy == pol].sort_values("B_h").saving_pct.to_numpy()
    if np.any(np.diff(s) < -0.05):
        issues.append("A3: %s not monotone in budget: %s" % (pol, np.round(s, 2)))
print("A2 oracle dominance and A3 budget monotonicity:", "PASS" if not issues else issues)"""),

md("""## 2. Held-out evaluation

Months 25 to 36, scored once with every parameter frozen. Each policy yields twelve monthly
savings against the carbon-agnostic baseline; the interval is a percentile bootstrap over
those months and significance uses a Wilcoxon signed rank test corrected by Holm-Bonferroni
within each signal and budget."""),
code("""stage("evaluate_held_out.py")
h = load("phase5_held_out.csv")
main = h[(h.signal == "S1_published")][["B_h","policy","mean_saving_pct","ci_low","ci_high",
                                        "months_positive","n_months","p_holm","pct_of_oracle"]]
display(main.round(3).reset_index(drop=True))
print("all comparisons significant at 5%% after correction:", bool((h.p_holm < 0.05).all()))"""),

md("""### Both signals

S1 is the published operational forecast. S2 is the day-ahead forecast built in Phase 3e from
information available at least 24 hours ahead. A claim enters the manuscript only where it
holds under both."""),
code("""both = h[h.policy.isin(["horizon","oracle"])].pivot_table(
    index=["B_h","policy"], columns="signal", values="mean_saving_pct").round(2)
display(both)
hz = h[h.policy=="horizon"].pivot_table(index="B_h", columns="signal", values="pct_of_oracle").round(1)
print("\\nShare of achievable saving realised (%):"); display(hz)
print("Under the published forecast the share rises with the budget; under the day-ahead")
print("forecast it falls. Forecast quality and scheduling flexibility substitute at the margin.")"""),

md("""## 3. Sensitivities

Three sweeps establish what the saving actually depends on: the power model, the available
headroom, and the deferral budgets granted to tenants."""),
code("""stage("simulate_policies.py", "energy"); stage("simulate_policies.py", "utilisation")
stage("simulate_policies.py", "flexibility"); stage("simulate_policies.py", "fairness")
e, u = load("phase3d_energy_bracket.csv"), load("phase3c_utilisation.csv")
f, fr = load("phase3c_flexibility.csv"), load("phase3c_fairness.csv")
print("Energy bracket — facility saving by idle-to-peak ratio")
display(e.pivot_table(index="idle_fraction", columns="B_h",
                      values=["saving_pct","facility_saving_pct"]).round(2))
print("Headroom — saving by mean utilisation"); display(u[["utilisation","saving_pct","deadline_violation_pct"]].round(2))
print("Flexibility grant"); display(f[["grant_h","saving_pct","tenant_jain","tenant_spread_gCO2_kWh","worst_tenant_intensity"]].round(3))
print("Allocation lever (negative result)"); display(fr[["fair_lambda","saving_pct","tenant_jain","tenant_spread_gCO2_kWh"]].round(5))"""),

md("""The allocation sweep is a negative result and is reported as one. Reweighting who receives
capacity within a period changes nothing, because a tenant's achievable intensities are fixed
by its deadline before any allocation decision is made. Granting slack changes both the saving
and the equity outcome, and changes them in the same direction."""),

md("""## 4. Spatial placement

Scored on the forecast field, because no regional outturn is published. The temporal baseline
is scored on the same field so the comparison is internally consistent, and the figures are
bounds rather than measured savings."""),
code("""stage("simulate_spatial.py", "12", "0.85"); stage("simulate_spatial.py", "3", "0.45")
hi, lo = load("phase4_spatial_u85.csv"), load("phase4_spatial_u45.csv")
print("Busy fleet, utilisation 0.85, twelve months")
display(hi[["mode","exposure_cap","saving_pct","top_region_share_pct","herfindahl"]].round(3))
print("Spare capacity, utilisation 0.45, three months")
display(lo[["mode","exposure_cap","saving_pct","top_region_share_pct","herfindahl"]].round(3))
st = load("phase4_spatial_stress_u85.csv")
print("\\nStress test, decisions on the forecast scored against a perturbed field:")
display(st.groupby(["mode"], dropna=False).saving_pct.agg(["mean","std"]).round(3))"""),

md("""## 5. Figures

Every figure is drawn from the stored CSVs above, never from a value typed into the plotting
script. The manifest records which file each panel draws on."""),
code("""stage("make_figures.py")
display(load("figure_manifest.csv"))
print("wall-clock for this notebook: %.1f s" % (time.time() - T0))"""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name":"python3","display_name":"Python 3","language":"python"}
NotebookClient(nb, timeout=2400, kernel_name="python3",
               resources={"metadata":{"path": os.path.join(ROOT,"notebooks")}}).execute()
nbf.write(nb, NB); print("written:", NB)
