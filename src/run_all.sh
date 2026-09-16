#!/usr/bin/env bash
# Full pipeline, in dependency order. Measured wall-clock times on the reference machine
# (single core, Intel Xeon 2.10 GHz, 3.9 GiB memory, no GPU) are given for each stage.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 src/fetch_runner.py 20          # grid retrieval, 613 responses      ~13 min
python3 src/build_and_audit.py          # parse, validate, Phase 1 audit     ~19 s
python3 src/build_workload.py           # Azure VM trace characterisation    ~44 s
python3 src/build_inference_load.py     # LLM inference aggregation          ~70 s
python3 src/build_dayahead_forecast.py  # S2 signal                          ~5 s
python3 src/simulate_policies.py policies      # design-period comparison    ~79 s
python3 src/simulate_policies.py fairness      # allocation lever            ~31 s
python3 src/simulate_policies.py flexibility   # flexibility grant           ~22 s
python3 src/simulate_policies.py utilisation   # headroom sweep              ~35 s
python3 src/simulate_policies.py energy        # idle-power bracket          ~51 s
python3 src/simulate_policies.py signals       # both signals                ~58 s
python3 src/simulate_spatial.py 12 0.85        # spatial, busy fleet         ~110 s
python3 src/simulate_spatial.py 3 0.45         # spatial, spare capacity     ~24 s
python3 src/simulate_policies.py aiworkload    # synthetic training workload ~60 s
python3 src/simulate_policies.py tenantsens    # tenant budget sensitivity  ~101 s
python3 src/evaluate_held_out.py               # held-out scoring, once      ~85 s
python3 src/monthly_held_out.py                # per-month series            ~90 s
python3 src/pairwise_policies.py               # pairwise policy tests       ~20 s
python3 src/marginal_sensitivity.py            # marginal-signal sensitivity ~40 s
                                               # requires data/raw/neso_historic_generation_mix.csv
python3 src/make_figures.py                    # figures 1-6                 ~30 s
python3 src/make_figures2.py                   # figures 7-12                ~40 s
python3 src/make_figures3.py                   # figures 13-14               ~15 s
python3 src/consistency_audit.py               # 118-value audit             ~5 s

echo "pipeline complete"
