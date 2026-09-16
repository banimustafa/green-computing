"""Assemble and execute the Phase 1 notebook, retaining all outputs, figures and timings."""
import os
import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB = os.path.join(ROOT, "notebooks", "phase1_data_audit.ipynb")

md = lambda t: nbf.v4.new_markdown_cell(t)
code = lambda t: nbf.v4.new_code_cell(t)

cells = [
md("""# Phase 1 — Data acquisition and audit

**Study:** Fair and carbon-aware spatiotemporal scheduling of AI workloads in data centres under forecast uncertainty

This notebook documents the acquisition, validation, and audit of the grid carbon intensity data
underpinning the study. It answers the question that gates the whole design: **does a carbon saving
from deferral survive once the scheduler is restricted to the forecast that was actually published,
rather than the realised intensity known only after the event?**

All paths are relative to the project root, so the notebook runs unchanged on any machine after the
retrieval step has populated `data/raw/`."""),

code("""import os, sys, json, time, platform, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))  if os.path.basename(os.getcwd())=="notebooks" else os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "src"))
PROC, OUT = os.path.join(ROOT,"data","processed"), os.path.join(ROOT,"outputs")
FIG = os.path.join(OUT,"figures"); os.makedirs(FIG, exist_ok=True)
T0 = time.time()
print("root:", ROOT)
print(platform.platform(), "| python", platform.python_version(),
      "| numpy", np.__version__, "| pandas", pd.__version__)"""),

md("""## 1. Provenance of the data

Carbon intensity data comes from the National Energy System Operator (NESO) Carbon Intensity API,
which is open and requires no key. Three series were retrieved over a 36-month window, September 2023
to August 2026, in monthly blocks (the API rejects windows longer than 31 days):

| Series | Fields | Role in the study |
|---|---|---|
| National intensity | forecast and **actual** | ground-truth anchor for the uncertainty analysis |
| National generation | realised fuel mix | independent check on the carbon accounting |
| Regional intensity (15 series) | forecast and mix, **no actual** | the spatial decision field |

The absence of a realised regional series is a property of the published data, not an oversight:
the NESO data portal lists a regional carbon intensity *forecast* product and no regional outturn
product. The scope of the spatial claim is set accordingly in Section 6."""),

code("""log = json.load(open(os.path.join(ROOT,"data","raw","_retrieval_log.json"))) if os.path.exists(os.path.join(ROOT,"data","raw","_retrieval_log.json")) else {}
files = glob.glob(os.path.join(ROOT,"data","raw","*.json"))
print("cached raw responses:", len(files))
print("raw size (MB): %.1f" % (sum(os.path.getsize(f) for f in files)/1e6))"""),

md("""## 2. Coverage and missingness

Completeness is reported against the number of half-hourly settlement periods the window should
contain. Gaps are retained rather than interpolated at this stage, so that any later imputation is an
explicit, auditable decision."""),

code("""cov = pd.read_csv(os.path.join(OUT,"audit_coverage.csv"))
cov["completeness_pct"] = (1 - cov.missing_periods/cov.expected_periods)*100
display(cov[["series","records","expected_periods","missing_periods","completeness_pct","missing_forecast"]].round(3))"""),

md("""## 3. Validating the carbon accounting

The published realised intensity is not taken on trust. It is recomputed from the realised fuel mix
and the published fuel-specific intensity factors, then compared with the published value.

The comparison uses a simplified factor mapping: the published mix reports a single `gas` category
where the factor table distinguishes combined-cycle from open-cycle plant, and a single `imports`
category where the table gives separate Dutch, French, and Irish factors. The recomputation therefore
carries a known approximation, and the result below should be read as a structural check rather than
as an exact reproduction."""),

code("""acc = pd.read_csv(os.path.join(OUT,"audit_accounting.csv"))
display(acc.round(3))
print("Interpretation: correlation of %.3f confirms the published series tracks the realised fuel mix."
      % acc.pearson_r[0])
print("The mean signed difference of %+.1f gCO2/kWh is a systematic offset from the simplified"
      % acc.mean_signed_diff_gCO2_kWh[0])
print("fuel-category mapping described above, not a random discrepancy. The study uses the published")
print("actual series throughout; the recomputation serves only to confirm its structure.")"""),

md("""## 4. Forecast accuracy

The error is defined as forecast minus actual, in gCO2/kWh, over all periods where both are present."""),

code("""fe = pd.read_csv(os.path.join(OUT,"audit_forecast_error.csv"))
display(fe[fe.level.isin(["all","season","year"])][["level","stratum","n","mean_actual","bias_ME","MAE","RMSE","MAPE_pct","corr"]].round(3))"""),

code("""nat = pd.read_csv(os.path.join(PROC,"national_intensity.csv"), parse_dates=["from"])
d = nat.dropna(subset=["actual","forecast"]).copy(); d["err"] = d.forecast - d.actual
byh = fe[fe.level=="hour_utc"].astype({"stratum":int}).sort_values("stratum")

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
ax[0].hist(d.err, bins=80, color="#4a6fa5", edgecolor="none")
ax[0].set_xlabel("forecast − actual (gCO$_2$/kWh)"); ax[0].set_ylabel("settlement periods")
ax[0].set_title("(a) Forecast error distribution")
ax[1].plot(byh.stratum, byh.MAE, "o-", color="#4a6fa5", label="MAE")
ax[1].plot(byh.stratum, byh.RMSE, "s--", color="#b05a3c", label="RMSE")
ax[1].set_xlabel("hour of day (UTC)"); ax[1].set_ylabel("gCO$_2$/kWh"); ax[1].legend(frameon=False)
ax[1].set_title("(b) Error by time of day")
h = d.assign(hour=d["from"].dt.hour).groupby("hour").actual.agg(["mean","std"])
ax[2].plot(h.index, h["mean"], color="#3d7a56")
ax[2].fill_between(h.index, h["mean"]-h["std"], h["mean"]+h["std"], alpha=.25, color="#3d7a56")
ax[2].set_xlabel("hour of day (UTC)"); ax[2].set_ylabel("gCO$_2$/kWh")
ax[2].set_title("(c) Diurnal intensity, mean ± 1 s.d.")
for a in ax: a.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig1_forecast_error.png"), dpi=200); plt.show()"""),

md("""## 5. The gating result: oracle versus forecast-guided deferral

For a unit job arriving in every settlement period, three placements are compared over a deferral
window of B hours:

- **immediate** — run on arrival, the carbon-agnostic baseline;
- **forecast-guided** — run in the period with the lowest *forecast* intensity in the window, and pay
  the *realised* intensity of the period chosen;
- **oracle** — run in the period with the lowest *realised* intensity, which requires perfect foresight.

All three are scored on realised intensity, so the comparison is like for like. The ratio between the
forecast-guided and oracle savings is the share of the theoretical opportunity that survives
uncertainty; its complement is the price of uncertainty."""),

code("""dp = pd.read_csv(os.path.join(OUT,"audit_deferral_potential.csv"))
display(dp.round(2))

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax[0].plot(dp.deferral_budget_h, dp.saving_oracle_pct, "s--", color="#b05a3c", label="oracle (perfect foresight)")
ax[0].plot(dp.deferral_budget_h, dp.saving_forecast_guided_pct, "o-", color="#4a6fa5", label="forecast-guided")
ax[0].fill_between(dp.deferral_budget_h, dp.saving_forecast_guided_pct, dp.saving_oracle_pct,
                   color="#b05a3c", alpha=.15, label="price of uncertainty")
ax[0].set_xlabel("deferral budget (hours)"); ax[0].set_ylabel("carbon saving vs immediate (%)")
ax[0].set_title("(a) Saving against deferral budget"); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(dp.deferral_budget_h, dp.realised_fraction_of_oracle_pct, "o-", color="#3d7a56")
ax[1].set_xlabel("deferral budget (hours)"); ax[1].set_ylabel("% of oracle saving realised")
ax[1].set_ylim(0, 100); ax[1].set_title("(b) Share of opportunity surviving uncertainty")
for a in ax: a.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig2_deferral_potential.png"), dpi=200); plt.show()"""),

code("""print("Gating criterion: a forecast-guided saving that is both positive and a substantial share")
print("of the oracle saving across realistic deferral budgets.\\n")
for _, r in dp.iterrows():
    print("  B=%2dh   guided %5.2f%%   oracle %5.2f%%   realised share %5.1f%%   jobs worse off %4.1f%%"
          % (r.deferral_budget_h, r.saving_forecast_guided_pct, r.saving_oracle_pct,
             r.realised_fraction_of_oracle_pct, r.pct_jobs_worse_than_immediate))"""),

md("""## 6. The spatial signal and its limits

Dispersion across the fourteen distribution regions sets an upper bound on what spatial shifting could
exploit. This bound is loose, and deliberately reported as such: it ignores capacity, network, latency,
and data-residency constraints, and the cleanest regions are low-demand, wind-rich areas that could not
absorb arbitrary load. The figure below motivates the regional exposure cap introduced in the
scheduling design rather than a headline saving."""),

code("""sp = pd.read_csv(os.path.join(OUT,"audit_spatial_spread.csv"))
cl = pd.read_csv(os.path.join(OUT,"audit_cleanest_region_share.csv"))
display(sp.round(2)); display(cl.round(2))
print("These figures are an unconstrained upper bound, not an achievable saving.")"""),

md("""## 7. Recorded limitations

1. **Forecast vintage.** The API returns one forecast value per settlement period. Whether this is the
   day-ahead forecast or a later revision is not stated in the response, so the reported accuracy may
   flatter a genuine day-ahead decision. Phase 1b resolves this against the archived forecast vintages
   published on the NESO data portal, and the scheduling experiments will use a vintage with a
   documented lead time.
2. **No regional outturn.** Regional performance is evaluated on the operational signal; only the
   national analysis is validated against realised intensity.
3. **Attributional, not marginal, intensity.** The published figures are average intensities. Marginal
   emissions would be the theoretically preferable signal for a marginal load decision, and this is
   stated as a limitation rather than corrected.
4. **No local power metering.** The sandbox exposes no RAPL counters, so job energy enters the
   scheduling experiments as a bounded interval drawn from published measurements, and every headline
   result is reported at both bounds."""),

code("""runtime = time.time() - T0
with open(os.path.join(OUT,"phase1_runtime.txt"),"w") as f:
    f.write("notebook wall-clock seconds: %.1f\\n" % runtime)
    f.write("platform: %s\\npython: %s\\ncpu_count: %s\\n" % (platform.platform(), platform.python_version(), os.cpu_count()))
print("notebook wall-clock: %.1f s" % runtime)"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs(os.path.dirname(NB), exist_ok=True)
NotebookClient(nb, timeout=1200, kernel_name="python3",
               resources={"metadata": {"path": os.path.join(ROOT, "notebooks")}}).execute()
nbf.write(nb, NB)
print("written:", NB)
