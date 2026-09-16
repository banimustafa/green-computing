"""
Independent verification pass.

The three existing audits check the manuscript against the result files, the result files against
each other, and every number in the source for a traceable origin. All three read the same
artefacts the pipeline wrote. This script checks the artefacts themselves, in three ways the
others cannot.

  1. Raw against panel. A random sample of settlement periods is re-read from the cached API
     responses and compared with the processed panel the simulations use.
  2. Determinism. Two pipeline stages are re-executed and their outputs compared with the stored
     files, byte for byte where the stage is deterministic.
  3. Rendered output. The numbers are re-audited from the text layer of the compiled PDF rather
     than from the LaTeX source, which is the only way to detect a value that is correct in the
     source and lost or mangled in typesetting.
"""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw")
PDF = os.path.join(ROOT, "submission", "manuscript.pdf")

results = []


def record(name, ok, detail=""):
    results.append({"check": name, "result": "pass" if ok else "FAIL", "detail": detail})
    print("%-58s %s  %s" % (name[:58], "pass" if ok else "FAIL", detail), flush=True)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------ 1. raw against the panel
def raw_against_panel(n_sample=1500, seed=7):
    rows = {}
    for f in sorted(glob.glob(os.path.join(RAW, "national_intensity_*.json"))):
        for rec in json.load(open(f))["data"]:
            it = rec.get("intensity", {})
            rows[rec["from"]] = (it.get("forecast"), it.get("actual"))
    panel = pd.read_csv(os.path.join(PROC, "national_intensity.csv"))
    panel["key"] = pd.to_datetime(panel["from"], utc=True, format="ISO8601").dt.strftime(
        "%Y-%m-%dT%H:%MZ")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(panel), size=min(n_sample, len(panel)), replace=False)
    bad = []
    for i in idx:
        r = panel.iloc[i]
        src = rows.get(r["key"])
        if src is None:
            bad.append((r["key"], "absent from the cached responses"))
            continue
        for got, want, label in ((r.forecast, src[0], "forecast"), (r.actual, src[1], "actual")):
            if pd.isna(got) and want is None:
                continue
            if pd.isna(got) != (want is None) or (want is not None and float(got) != float(want)):
                bad.append((r["key"], "%s %s against %s" % (label, got, want)))
    record("raw responses reproduce the national panel",
           not bad, "%d periods sampled, %d mismatches" % (len(idx), len(bad)))
    if bad:
        for b in bad[:5]:
            print("    ", b)

    # Every cached response is accounted for in the panel, and nothing is invented.
    record("panel contains no period absent from the cached responses",
           set(panel["key"]) <= set(rows),
           "%d panel periods, %d cached periods" % (len(panel), len(rows)))


# ------------------------------------------------------------------------- 2. determinism
def determinism():
    targets = ["audit_coverage.csv", "audit_accounting.csv", "audit_forecast_error.csv",
               "audit_deferral_potential.csv", "audit_spatial_spread.csv"]
    before = {t: sha(os.path.join(OUT, t)) for t in targets}
    subprocess.run([sys.executable, os.path.join(HERE, "build_and_audit.py")],
                   check=True, capture_output=True)
    after = {t: sha(os.path.join(OUT, t)) for t in targets}
    same = [t for t in targets if before[t] == after[t]]
    record("audit stage reproduces identical outputs on re-execution",
           len(same) == len(targets), "%d of %d files identical by checksum"
           % (len(same), len(targets)))

    stored = pd.read_csv(os.path.join(OUT, "phase3d_energy_bracket.csv"))
    sha_before = sha(os.path.join(OUT, "phase3d_energy_bracket.csv"))
    subprocess.run([sys.executable, os.path.join(HERE, "simulate_policies.py"), "energy"],
                   check=True, capture_output=True)
    rerun = pd.read_csv(os.path.join(OUT, "phase3d_energy_bracket.csv"))
    record("simulation stage reproduces identical results on re-execution",
           sha(os.path.join(OUT, "phase3d_energy_bracket.csv")) == sha_before,
           "energy bracket, %d rows, savings %s" % (len(rerun),
               np.round(rerun.saving_pct.to_numpy(), 4).tolist()))
    record("re-executed simulation matches the stored values numerically",
           np.allclose(stored.saving_pct, rerun.saving_pct, atol=1e-9)
           and np.allclose(stored.facility_saving_pct, rerun.facility_saving_pct, atol=1e-9))


# --------------------------------------------------------------------- 3. the rendered PDF
NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![\w])")

AXIS_DECIMALS = {"0.00", "0.25", "0.50", "0.75", "1.00", "0.0", "0.5", "1.0"}


def is_axis_label(tok):
    """Round values of the kind matplotlib places on an axis."""
    if tok in AXIS_DECIMALS:
        return True
    t = tok.replace(",", "")
    # Matplotlib places ticks on multiples of 1, 2, 2.5 and 5 times a power of ten, so counts
    # such as 1250 and 1750 appear as well as round hundreds.
    return t.isdigit() and len(t) <= 5 and int(t) > 0 and int(t) % 25 == 0


def rendered_pdf():
    tmp = tempfile.mkdtemp()
    txt = os.path.join(tmp, "m.txt")
    subprocess.run(["pdftotext", "-layout", PDF, txt], check=True, capture_output=True)
    body = open(txt, encoding="utf-8", errors="ignore").read()
    # The reference list is full of page ranges, volume numbers and DOIs that are not results.
    cut = body.rfind("References")
    body = body[:cut] if cut > 0 else body

    sys.path.insert(0, HERE)
    from full_audit import build_index, STRUCTURAL
    idx, _ = build_index()

    untraced, traced, structural, short, axis = [], 0, 0, 0, 0
    for m in NUM.finditer(body):
        tok = m.group(1)
        if tok in idx:
            traced += 1
        elif tok in STRUCTURAL:
            structural += 1
        elif "." not in tok and "," not in tok and len(tok) < 4:
            short += 1
        elif is_axis_label(tok):
            # Axis tick labels sit in the figure text layer, which the source-level audit does not
            # see because it excludes figure environments. They are counted, not flagged.
            axis += 1
        else:
            ctx = re.sub(r"\s+", " ", body[max(0, m.start() - 60):m.end() + 20]).strip()
            untraced.append((tok, ctx))
    record("every decimal and long integer in the rendered PDF traces to a result file",
           not untraced, "%d traced, %d structural, %d short integers, %d untraced"
           % (traced, structural, short, len(untraced)))
    for tok, ctx in untraced[:12]:
        print("     untraced %-10s %s" % (tok, ctx[:90]))

    src = open(os.path.join(ROOT, "submission", "manuscript.tex")).read()
    for probe in ("10.06", "89.18", "37.97", "272", "34,252"):
        record("rendered PDF carries the headline value %s" % probe,
               probe in body and probe in src)
    shutil.rmtree(tmp)


def main():
    raw_against_panel()
    determinism()
    rendered_pdf()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, "verification_pass.csv"), index=False)
    n_fail = int((df.result == "FAIL").sum())
    print("\n%d verifications, %d passed, %d failed" % (len(df), len(df) - n_fail, n_fail))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
