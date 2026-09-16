"""
Phase 2 - Construction of the workload model.

The deferrable and non-deferrable classes are not assumed by us. The Azure 2019 VM trace
(AzurePublicDatasetV2) carries an operator-assigned category for every virtual machine,
taking the values Delay-insensitive, Interactive, and Unknown. The scheduling study takes
that label at face value:

  deferrable      <- Delay-insensitive virtual machines
  non-deferrable  <- Interactive virtual machines
  excluded        <- Unknown, reported but held out of the headline experiments

The subscription identifier gives the tenant, which is what the tenant-level fairness
constraint is defined over.

Schema (no header row in the source file), 11 columns:
  vmid, subscriptionid, deploymentid, vmcreated, vmdeleted, maxcpu, avgcpu, p95maxcpu,
  vmcategory, vmcorecountbucket, vmmemorybucket
Timestamps are seconds from the start of the 30-day observation window.
"""

import gzip
import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")

COLS = ["vmid", "subscriptionid", "deploymentid", "vmcreated", "vmdeleted",
        "maxcpu", "avgcpu", "p95maxcpu", "vmcategory", "cores", "memory_gb"]
KEEP = ["subscriptionid", "vmcreated", "vmdeleted", "avgcpu", "p95maxcpu",
        "vmcategory", "cores", "memory_gb"]


def short_hash(s):
    """Stable 8-hex-character tenant key, so that a 64-character encrypted identifier
    does not dominate memory. Collisions across 6,687 subscriptions are negligible."""
    return hashlib.blake2b(s.encode(), digest_size=4).hexdigest()


def parse_vmtable(path, chunksize=400_000):
    parts = []
    with gzip.open(path, "rt") as fh:
        for chunk in pd.read_csv(fh, header=None, names=COLS, usecols=KEEP,
                                 chunksize=chunksize,
                                 dtype={"vmcategory": "category", "cores": "str",
                                        "memory_gb": "str"}):
            chunk["tenant"] = chunk["subscriptionid"].map(short_hash)
            parts.append(chunk.drop(columns=["subscriptionid"]))
    df = pd.concat(parts, ignore_index=True)
    # Core and memory buckets are published as strings, occasionally as ranges.
    for c in ("cores", "memory_gb"):
        df[c] = pd.to_numeric(df[c].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    df["duration_s"] = df["vmdeleted"] - df["vmcreated"]
    df["core_hours"] = df["cores"] * df["duration_s"] / 3600.0
    df["tenant"] = df["tenant"].astype("category")
    return df


def characterise(df):
    rows = []
    for cat, g in df.groupby("vmcategory", observed=True):
        d = g["duration_s"] / 3600.0
        rows.append({
            "category": cat, "n_vms": len(g),
            "pct_left_censored": (g["vmcreated"] <= 0).mean() * 100,
            "pct_right_censored": (g["vmdeleted"] >= df["vmdeleted"].max() - 1).mean() * 100, "pct_of_vms": len(g) / len(df) * 100,
            "n_tenants": g["tenant"].nunique(),
            "total_core_hours": g["core_hours"].sum(),
            "pct_of_core_hours": np.nan,
            "median_duration_h": d.median(), "mean_duration_h": d.mean(),
            "p90_duration_h": d.quantile(0.90),
            "pct_shorter_than_1h": (d < 1).mean() * 100,
            "pct_longer_than_24h": (d > 24).mean() * 100,
            "median_cores": g["cores"].median(), "mean_avgcpu_pct": g["avgcpu"].mean()})
    out = pd.DataFrame(rows)
    out["pct_of_core_hours"] = out["total_core_hours"] / out["total_core_hours"].sum() * 100
    return out


def tenant_concentration(df):
    """Tenant structure of the deferrable class, which determines whether a tenant-level
    fairness constraint is measurable on this trace."""
    d = df[df.vmcategory == "Delay-insensitive"]
    by = d.groupby("tenant", observed=True)["core_hours"].sum().sort_values(ascending=False)
    share = by / by.sum()
    cum = share.cumsum()
    n = len(by)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(by.sort_values().to_numpy()) / (n * by.sum()))
    return pd.DataFrame([{
        "n_tenants_deferrable": n,
        "top1_share_pct": share.iloc[0] * 100,
        "top10_share_pct": share.iloc[:10].sum() * 100,
        "top1pct_tenants_share_pct": share.iloc[:max(1, n // 100)].sum() * 100,
        "tenants_for_50pct_core_hours": int((cum < 0.5).sum() + 1),
        "median_tenant_core_hours": float(by.median()),
        "gini_core_hours": gini}])


def arrival_profile(df):
    """Diurnal arrival shape of each class, expressed in trace-relative hours.

    Virtual machines already running when observation began appear with a creation
    timestamp of zero. Including them puts 86.5% of delay-insensitive arrivals in hour
    zero, which is an artefact of left censoring rather than a diurnal peak, so they are
    excluded here. The trace carries no absolute dates, so alignment to the grid series
    remains a modelling decision recorded in the analysis plan."""
    rows = []
    for cat in ("Delay-insensitive", "Interactive"):
        g = df[(df.vmcategory == cat) & (df.vmcreated > 0)]
        h = (g["vmcreated"] // 3600 % 24).value_counts().sort_index()
        h = (h / h.sum() * 100).rename("pct_arrivals")
        rows.append(h.rename_axis("hour_of_day").reset_index().assign(category=cat))
    return pd.concat(rows, ignore_index=True)


def main():
    t0 = time.time()
    df = parse_vmtable(os.path.join(RAW, "vmtable.csv.gz"))
    print("parsed %d virtual machines in %.1f s" % (len(df), time.time() - t0))

    ch = characterise(df)
    ch.to_csv(os.path.join(OUT, "workload_class_summary.csv"), index=False)
    tc = tenant_concentration(df)
    tc.to_csv(os.path.join(OUT, "workload_tenant_concentration.csv"), index=False)
    ap = arrival_profile(df)
    ap.to_csv(os.path.join(OUT, "workload_arrival_profile.csv"), index=False)

    # Compact panel retained for the scheduling experiments.
    keep = df[df.vmcategory.isin(["Delay-insensitive", "Interactive"])].copy()
    keep[["tenant", "vmcategory", "vmcreated", "vmdeleted", "duration_s",
          "cores", "memory_gb", "avgcpu", "core_hours"]].to_csv(
        os.path.join(PROC, "workload_vms.csv.gz"), index=False, compression="gzip")

    print("\nWORKLOAD CLASSES\n", ch.round(2).to_string(index=False))
    print("\nTENANT CONCENTRATION (deferrable class)\n", tc.round(3).to_string(index=False))
    print("\ntotal elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
