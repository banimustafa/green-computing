"""
Phase 1 - Retrieval of GB carbon intensity data for the study:
"Fair and carbon-aware spatiotemporal scheduling of AI workloads in data centres
 under forecast uncertainty"

Source: National Energy System Operator (NESO) Carbon Intensity API,
        https://api.carbonintensity.org.uk  (open, no API key required)

Three series are retrieved over a 36-month window:
  (a) national half-hourly intensity, FORECAST and ACTUAL   -> ground truth anchor
  (b) national half-hourly realised generation mix          -> accounting validation
  (c) regional half-hourly intensity + mix, FORECAST only   -> spatial decision field

The API caps each request at 31 days, so the window is walked in monthly blocks.
Every raw JSON response is cached verbatim under data/raw/ so that the retrieval is
resumable, auditable, and reproducible without re-contacting the API.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

BASE = "https://api.carbonintensity.org.uk"
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HEADERS = {"Accept": "application/json", "User-Agent": "gcas-research/1.0"}

# 36-month window ending at the most recent complete month.
START = datetime(2023, 9, 1, tzinfo=timezone.utc)
END = datetime(2026, 9, 1, tzinfo=timezone.utc)

# DNO regions 1-14 are the addressable distribution regions; 18 is the GB aggregate,
# retained because it is the only regional series with a national actual to check against.
REGIONS = list(range(1, 15)) + [18]


def month_blocks(start, end):
    """Yield (from, to) pairs of at most one calendar month."""
    cur = start
    while cur < end:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cur, min(nxt, end)
        cur = nxt


def fmt(dt):
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def get(url, path, retries=4):
    """Fetch url to path, with caching and exponential back-off."""
    if os.path.exists(path) and os.path.getsize(path) > 200:
        return "cached"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            json.loads(body)  # reject malformed payloads before caching
            with open(path, "wb") as f:
                f.write(body)
            return "fetched"
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            if attempt == retries - 1:
                return "FAILED: %s" % exc
            time.sleep(2 ** attempt)
    return "FAILED"


def main():
    os.makedirs(RAW, exist_ok=True)
    log = []
    blocks = list(month_blocks(START, END))
    print("blocks: %d  regions: %d" % (len(blocks), len(REGIONS)), flush=True)

    for a, b in blocks:
        tag = a.strftime("%Y%m")

        # (a) national intensity: forecast and actual
        log.append(("national_intensity", tag, get(
            "%s/intensity/%s/%s" % (BASE, fmt(a), fmt(b)),
            os.path.join(RAW, "national_intensity_%s.json" % tag))))

        # (b) national realised generation mix
        log.append(("national_generation", tag, get(
            "%s/generation/%s/%s" % (BASE, fmt(a), fmt(b)),
            os.path.join(RAW, "national_generation_%s.json" % tag))))

        # (c) regional forecast intensity and mix
        for rid in REGIONS:
            log.append(("regional_%02d" % rid, tag, get(
                "%s/regional/intensity/%s/%s/regionid/%d" % (BASE, fmt(a), fmt(b), rid),
                os.path.join(RAW, "regional_%02d_%s.json" % (rid, tag)))))
        print("%s done (%d calls logged)" % (tag, len(log)), flush=True)

    # Static reference: published fuel-specific carbon intensity factors
    log.append(("intensity_factors", "static", get(
        "%s/intensity/factors" % BASE,
        os.path.join(RAW, "intensity_factors.json"))))

    fails = [r for r in log if str(r[2]).startswith("FAILED")]
    with open(os.path.join(RAW, "_retrieval_log.json"), "w") as f:
        json.dump({"retrieved_utc": datetime.now(timezone.utc).isoformat(),
                   "window": [fmt(START), fmt(END)],
                   "calls": len(log),
                   "failures": fails,
                   "log": log}, f, indent=1)
    print("TOTAL calls %d, failures %d" % (len(log), len(fails)), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
