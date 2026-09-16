"""
Phase 2b - Aggregation of the Azure LLM inference traces (2024, one week) into the
half-hourly resolution of the grid series.

The raw traces hold 16.8 million code requests and a larger number of conversation
requests, so they are read in chunks and reduced on the fly; the full frames are never
held in memory. For each half-hour the aggregation records request count, prompt
(context) tokens, and generated tokens, which together drive the non-deferrable
inference load in the scheduling model.
"""

import os
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")

FILES = {"code": "AzureLLMInferenceTrace_code_1week.csv",
         "conversation": "AzureLLMInferenceTrace_conv_1week.csv"}


def aggregate(path, chunksize=2_000_000):
    acc = []
    for ch in pd.read_csv(path, chunksize=chunksize,
                          dtype={"ContextTokens": "int32", "GeneratedTokens": "int32"}):
        ch["TIMESTAMP"] = pd.to_datetime(ch["TIMESTAMP"], format="ISO8601", utc=True)
        g = ch.groupby(ch["TIMESTAMP"].dt.floor("30min")).agg(
            requests=("ContextTokens", "size"),
            context_tokens=("ContextTokens", "sum"),
            generated_tokens=("GeneratedTokens", "sum"))
        acc.append(g)
    out = pd.concat(acc).groupby(level=0).sum()
    out.index.name = "period"
    return out


def main():
    t0 = time.time()
    frames = {}
    for name, fn in FILES.items():
        p = os.path.join(RAW, fn)
        t1 = time.time()
        frames[name] = aggregate(p)
        print("%s: %d periods, %.0f M requests, %.1f s"
              % (name, len(frames[name]), frames[name].requests.sum() / 1e6, time.time() - t1),
              flush=True)

    panel = pd.concat(frames, names=["workload"]).reset_index()
    panel.to_csv(os.path.join(PROC, "llm_inference_halfhourly.csv"), index=False)

    summary = panel.groupby("workload").agg(
        periods=("requests", "size"),
        total_requests=("requests", "sum"),
        mean_requests_per_30min=("requests", "mean"),
        peak_requests_per_30min=("requests", "max"),
        peak_to_mean_ratio=("requests", lambda s: s.max() / s.mean()),
        total_context_tokens=("context_tokens", "sum"),
        total_generated_tokens=("generated_tokens", "sum")).reset_index()
    summary["mean_tokens_per_request"] = (
        (summary.total_context_tokens + summary.total_generated_tokens) / summary.total_requests)
    summary.to_csv(os.path.join(OUT, "llm_inference_summary.csv"), index=False)

    # Diurnal shape, which is what the scheduling model uses to reserve capacity.
    panel["hour"] = pd.to_datetime(panel["period"], utc=True).dt.hour
    diurnal = panel.groupby(["workload", "hour"]).requests.mean().reset_index()
    diurnal["pct_of_daily_mean"] = diurnal.groupby("workload").requests.transform(
        lambda s: s / s.mean() * 100)
    diurnal.to_csv(os.path.join(OUT, "llm_inference_diurnal.csv"), index=False)

    print("\nINFERENCE TRACE SUMMARY\n", summary.round(2).to_string(index=False))
    print("\ntotal elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
