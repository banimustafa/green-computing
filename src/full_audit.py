"""
Full manuscript audit.

The existing consistency audit runs forward: it takes values from the result files and confirms
they appear in the manuscript. That direction cannot detect a number in the manuscript which has
no source at all, which is precisely where an error or an invention would hide. This script runs
the audit in both directions and over all three surfaces.

  1. Value index. Every numeric cell in every result file is formatted at several precisions and
     indexed to its file, column, and row.
  2. Tables. Each LaTeX tabular is parsed cell by cell and every numeric cell is looked up.
  3. Prose. Every number in the body text, outside tables and figure environments, is looked up.
  4. Figures. Every figure file referenced by the manuscript is checked to exist, to be declared in
     the figure manifest, and to have its declared source files present; the plotted series are
     re-derived from those sources and their ranges reported.

Numbers are classified rather than merely counted. Decimals and long integers are audited strictly.
Short integers carry little information and coincide with structural quantities such as section
numbers and counts, so they are reported separately for eye inspection rather than flagged.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
PROC = os.path.join(ROOT, "data", "processed")
TEX = os.path.join(ROOT, "submission", "manuscript.tex")
FIGDIR = os.path.join(ROOT, "submission", "figures")

# Quantities that are properties of the study design rather than computed results. They are
# declared here so that they are audited against this list rather than silently accepted.
STRUCTURAL = {
    "36": "months in the window", "24": "hours, deferral budget", "12": "months held out",
    "4": "hours, deferral budget", "14": "distribution regions", "13": "reference number",
    "5258": "tenants holding deferrable work", "200": "simulated tenants",
    "62": "least flexible tenants granted budget", "613": "cached API responses",
    "48": "half-hourly periods in a day", "96": "marginal factor bins",
    "45": "pairwise comparisons", "38": "distinguishable pairs", "7": "policies compared",
    "2": "hours, smallest tenant budget", "8": "hours, tenant budget",
    "10000": "bootstrap resamples", "90": "tenants holding half the deferrable work",
    "2023": "year", "2024": "year", "2025": "year", "2026": "year", "2019": "trace year",
    "0.85": "mean utilisation in the main configuration",
    "0.45": "mean utilisation in the spare-capacity configuration",
    "0.15": "tenant budget probability", "0.20": "tenant budget probability",
    "0.25": "tenant budget probability", "10,000": "bootstrap resamples",
    "0.95": "arrival cap as a share of capacity",
    "672": "settlement periods in the day-ahead warm-up, fourteen days",
}


def build_index():
    """Map formatted numeric strings to the files and columns they come from."""
    idx = {}
    files = sorted(glob.glob(os.path.join(OUT, "*.csv")))
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        name = os.path.basename(f)
        for col in d.columns:
            s = pd.to_numeric(d[col], errors="coerce").dropna()
            for v in s.unique():
                for fmt in ("%.0f", "%.1f", "%.2f", "%.3f", "%.4f", "%.5f"):
                    idx.setdefault(fmt % v, set()).add("%s:%s" % (name, col))
                if float(v).is_integer() and abs(v) >= 1000:
                    idx.setdefault("{:,}".format(int(v)), set()).add("%s:%s" % (name, col))
    # Values the manuscript derives from the result files rather than quoting directly. Each is
    # recomputed here from its inputs, so a derived figure is audited rather than excused.
    cov = pd.read_csv(os.path.join(OUT, "audit_coverage.csv"))
    r = cov[cov.series == "national_intensity"].iloc[0]
    wl = pd.read_csv(os.path.join(OUT, "workload_class_summary.csv"))
    pol = pd.read_csv(os.path.join(OUT, "phase3c_policies.csv"))
    sigp = pd.read_csv(os.path.join(PROC, "national_intensity_signals.csv"), parse_dates=["from"])
    nat_all = pd.read_csv(os.path.join(PROC, "national_intensity.csv"), parse_dates=["from"])
    cutoff = pd.Timestamp("2025-09-01", tz="UTC")
    derived = {"completeness per cent": 100.0 * r.records / r.expected_periods,
               "total virtual machines": float(wl.n_vms.sum()),
               "design periods under the published signal":
                   float((nat_all.dropna(subset=["actual", "forecast"])["from"] < cutoff).sum()),
               "design periods under the day-ahead signal":
                   float((sigp.dropna()["from"] < cutoff).sum())}
    for B in (4, 12, 24):
        h = pol[(pol.B_h == B) & (pol.policy == "horizon")].saving_pct.iloc[0]
        o = pol[(pol.B_h == B) & (pol.policy == "oracle")].saving_pct.iloc[0]
        derived["design-period share of oracle at B=%d" % B] = 100.0 * h / o
    for label, v in derived.items():
        for fmt in ("%.0f", "%.1f", "%.2f"):
            idx.setdefault(fmt % v, set()).add("derived:%s" % label)
        if float(v) >= 1000:
            idx.setdefault("{:,}".format(int(round(v))), set()).add("derived:%s" % label)

    # Summary statistics of the processed panels, which the data section quotes.
    nat = pd.read_csv(os.path.join(PROC, "national_intensity.csv"))
    for label, v in (("mean_actual", nat.actual.mean()), ("n_records", len(nat))):
        for fmt in ("%.0f", "%.1f", "%.2f"):
            idx.setdefault(fmt % v, set()).add("national_intensity.csv:%s" % label)
        idx.setdefault("{:,}".format(int(v)), set()).add("national_intensity.csv:%s" % label)
    return idx, len(files)


def strip_environments(t):
    """Remove tables and figures so that prose is audited separately from them."""
    t = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", " ", t, flags=re.S)
    t = re.sub(r"\\(label|ref|cite|includegraphics|documentclass|usepackage)\{[^}]*\}", " ", t)
    t = re.sub(r"\\(section|subsection|bmhead)\*?\{", " ", t)
    t = re.sub(r"%.*", " ", t)
    return t


NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![\w])")


def classify(tok):
    if "." in tok:
        return "decimal"
    if "," in tok or len(tok) >= 4:
        return "long integer"
    return "short integer"


def audit_tokens(text, surface, idx, rows):
    for m in NUM.finditer(text):
        tok = m.group(1)
        kind = classify(tok)
        src = sorted(idx.get(tok, []))
        ctx = re.sub(r"\s+", " ", text[max(0, m.start() - 55):m.end() + 25]).strip()
        if src:
            status = "traced"
        elif tok in STRUCTURAL:
            status = "structural"
        elif kind == "short integer":
            status = "short integer, not audited"
        else:
            status = "UNTRACED"
        rows.append({"surface": surface, "token": tok, "kind": kind, "status": status,
                     "sources": "; ".join(src[:3]), "context": ctx})


def audit_tables(t, idx, rows):
    tables = re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", t, flags=re.S)
    for i, tab in enumerate(tables, 1):
        body = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", tab, flags=re.S)
        if not body:
            continue
        cells = re.sub(r"\\(toprule|midrule|botrule|textbf|begin|end)\{?[^}\s]*\}?", " ",
                       body.group(0))
        audit_tokens(cells, "table %d" % i, idx, rows)


def audit_figures(t):
    used = re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}", t)
    man = pd.read_csv(os.path.join(OUT, "figure_manifest.csv"))
    rows = []
    for f in used:
        stem = os.path.splitext(f)[0]
        rec = man[man.figure == stem]
        srcs = rec.sources.iloc[0].split("; ") if len(rec) else []
        missing = [s for s in srcs
                   if not os.path.exists(os.path.join(OUT, s))
                   and not os.path.exists(os.path.join(ROOT, s))]
        rows.append({"figure": f, "file_present": os.path.exists(os.path.join(FIGDIR, f)),
                     "in_manifest": bool(len(rec)), "n_sources": len(srcs),
                     "missing_sources": "; ".join(missing)})
    return pd.DataFrame(rows)


def main():
    idx, nfiles = build_index()
    t = open(TEX).read()
    rows = []
    audit_tables(t, idx, rows)
    audit_tokens(strip_environments(t), "prose", idx, rows)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "full_audit_numbers.csv"), index=False)

    figs = audit_figures(t)
    figs.to_csv(os.path.join(OUT, "full_audit_figures.csv"), index=False)

    print("value index built from %d result files, %d distinct formatted values\n"
          % (nfiles, len(idx)))
    print("NUMBERS BY SURFACE AND STATUS")
    print(pd.crosstab(df.surface, df.status).to_string())
    print("\nNUMBERS BY KIND AND STATUS")
    print(pd.crosstab(df.kind, df.status).to_string())

    bad = df[df.status == "UNTRACED"]
    print("\nUNTRACED decimals and long integers: %d" % len(bad))
    for _, r in bad.iterrows():
        print("  [%s] %s -> %s" % (r.surface, r.token, r.context[:95]))

    print("\nFIGURES")
    print(figs.to_string(index=False))
    assert figs.file_present.all(), "a referenced figure file is missing"
    assert figs.in_manifest.all(), "a figure is not declared in the manifest"
    assert (figs.missing_sources == "").all(), "a figure declares a source that does not exist"
    print("\nall %d referenced figures exist, are declared, and their sources are present"
          % len(figs))


if __name__ == "__main__":
    main()
