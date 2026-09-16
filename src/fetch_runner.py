"""
Resumable, time-budgeted runner for the grid retrieval.

The sandbox suspends background work between invocations, so retrieval is performed
in the foreground in bounded slices. Every response is cached, so each invocation
resumes exactly where the previous one stopped.

Priority order reflects analytical dependency:
  1. national intensity  (forecast + actual)   -> the ground-truth anchor
  2. national generation (realised fuel mix)   -> accounting validation
  3. region 18 (GB)                            -> regional pipeline check
  4. regions 1-14                              -> spatial decision field

Usage:  python3 src/fetch_runner.py [minutes]
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

sys.path.insert(0, os.path.dirname(__file__))
from fetch_grid import BASE, RAW, START, END, month_blocks, fmt, get  # noqa: E402


def build_tasks():
    tasks = []
    blocks = [(a, b - timedelta(minutes=30)) for a, b in month_blocks(START, END)]
    for kind in ("national_intensity", "national_generation", "region18", "regions"):
        for a, b in blocks:
            tag = a.strftime("%Y%m")
            if kind == "national_intensity":
                tasks.append(("%s/intensity/%s/%s" % (BASE, fmt(a), fmt(b)),
                              os.path.join(RAW, "national_intensity_%s.json" % tag)))
            elif kind == "national_generation":
                tasks.append(("%s/generation/%s/%s" % (BASE, fmt(a), fmt(b)),
                              os.path.join(RAW, "national_generation_%s.json" % tag)))
            elif kind == "region18":
                tasks.append(("%s/regional/intensity/%s/%s/regionid/18" % (BASE, fmt(a), fmt(b)),
                              os.path.join(RAW, "regional_18_%s.json" % tag)))
            else:
                for rid in range(1, 15):
                    tasks.append(("%s/regional/intensity/%s/%s/regionid/%d" % (BASE, fmt(a), fmt(b), rid),
                                  os.path.join(RAW, "regional_%02d_%s.json" % (rid, tag)))) 
    tasks.append(("%s/intensity/factors" % BASE, os.path.join(RAW, "intensity_factors.json")))
    return tasks


def main():
    budget = float(sys.argv[1]) * 60 if len(sys.argv) > 1 else 300.0
    os.makedirs(RAW, exist_ok=True)
    tasks = build_tasks()
    todo = [t for t in tasks if not (os.path.exists(t[1]) and os.path.getsize(t[1]) > 200)]
    print("total %d, cached %d, outstanding %d" % (len(tasks), len(tasks) - len(todo), len(todo)), flush=True)

    # Retrieval is network-bound rather than compute-bound, so a small thread pool
    # raises throughput without loading the single available core. Concurrency is kept
    # modest out of courtesy to a public API that requires no key.
    t0, done, fail = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for url, path in todo:
            futures[pool.submit(get, url, path)] = path
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if str(r).startswith("FAILED"):
                fail += 1
                print("FAIL %s -> %s" % (os.path.basename(futures[fut]), r), flush=True)
            if done % 50 == 0:
                print("  %d fetched, %.0fs elapsed" % (done, time.time() - t0), flush=True)
            if time.time() - t0 > budget:
                print("budget reached", flush=True)
                break
        pool.shutdown(wait=False, cancel_futures=True)

    remaining = len([t for t in tasks if not (os.path.exists(t[1]) and os.path.getsize(t[1]) > 200)])
    print("fetched %d (failures %d) in %.0fs; remaining %d" % (done, fail, time.time() - t0, remaining), flush=True)


if __name__ == "__main__":
    main()
