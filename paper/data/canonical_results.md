# Canonical Results

Single source of truth for every number the paper may cite. Every value was
read directly from the raw files in `results/`, `results/sweep/` and
`profiling/` (branch `paper`, commit `007cce6`) and every derived value was
recomputed from those raw values, not copied from the prose write-ups.

## Conventions

- **Speedup "X over Y"** = `t_Y / t_X` (time on the slower backend divided by
  time on the faster one). A value > 1 means X is faster. The direction is
  spelled out in every speedup row.
- **Status** column:
  - **M**: measured. The value appears verbatim in a raw data file (JSON / CSV / log).
  - **D**: derived. Computed here from M values; the formula is in Notes.
  - **R**: reported only. The value appears in a `.md` write-up, with no raw data file in the repo behind it.
  - **I**: input assumption (e.g. cloud prices), not a measurement.
- Paths are relative to the repo root. `file → key` means a JSON key; `file:N` means a line number.
- Stdev = sample standard deviation (n − 1), which is what the source files use.
- Workload for every AF2 row unless noted: AlphaFold2 `model_3`, 118 residues,
  `num_recycle=0`, float32, random-init (untrained) weights, trivial single-sequence MSA.

## Master table

| ID | Experiment | Metric | Value | Status | Source | Notes |
|---|---|---|---|---|---|---|
| **HW** | **Hardware / setup** | | | | | |
| HW-01 | Setup | CPU backend | Google Colab Intel Xeon, 2 vCPU, 13.6 GB RAM | R | `results/comparison.md:9,17-18` | JSON records only `host_processor: x86_64` |
| HW-02 | Setup | GPU backend | Google Colab NVIDIA Tesla T4, 15 GB VRAM | R | `results/comparison.md:10,17-18` | JSON records `devices: ["cuda:0"]` |
| HW-03 | Setup | TPU backend | Stanford GKE TPU v5e-8, `tpu-v5-lite-podslice`, 2×4 | R | `results/comparison.md:11,17` | |
| HW-04 | Setup | TPU devices visible | 8 | M | `results/result_tpu-v5e-podslice.json → num_devices` | |
| HW-05 | Setup | TPU HBM limit per chip | 16,909,000,000 bytes (16.9 GB) | M | `results/sweep/chip_visibility_sweep.json → memory_stats_per_device[].bytes_limit` | |
| HW-06 | Setup | Sequence length (baseline) | 118 residues | M | `results/result_*.json → num_residues` | All three AF2 backend JSONs |
| HW-07 | Setup | AF2 output atom positions shape | [118, 37, 3] | M | `results/result_*.json → output_final_atom_positions_shape` | Identical on all three backends |
| **B** | **AF2 backend baseline (1 query)** | | | | | |
| B-01 | Baseline CPU | init_params | 41.99 s | M | `results/result_cpu-colab.json → init_params_seconds` | |
| B-02 | Baseline CPU | 1st predict (compile + run) | 271.98 s | M | `results/result_cpu-colab.json → first_predict_compile_and_run_seconds` | |
| B-03 | Baseline CPU | 2nd predict (steady state) | 212.113 s | M | `results/result_cpu-colab.json → second_predict_steady_state_seconds` | |
| B-04 | Baseline GPU | init_params | 109.16 s | M | `results/result_gpu-t4.json → init_params_seconds` | GPU init is 2.60× slower than CPU init |
| B-05 | Baseline GPU | 1st predict (compile + run) | 97.62 s | M | `results/result_gpu-t4.json → first_predict_compile_and_run_seconds` | |
| B-06 | Baseline GPU | 2nd predict (steady state) | 13.086 s | M | `results/result_gpu-t4.json → second_predict_steady_state_seconds` | See discrepancy 16 and "GPU session variability (Colab)": the 2026-09-15 rerun differs, gap unexplained |
| B-07 | Baseline TPU | init_params | 36.6 s | M | `results/result_tpu-v5e-podslice.json → init_params_seconds` | |
| B-08 | Baseline TPU | 1st predict (compile + run) | 27.78 s | M | `results/result_tpu-v5e-podslice.json → first_predict_compile_and_run_seconds` | |
| B-09 | Baseline TPU | 2nd predict (steady state) | 0.47 s | M | `results/result_tpu-v5e-podslice.json → second_predict_steady_state_seconds` | Runs on 1 of 8 chips, see CV-09 |
| B-10 | Baseline speedup | **GPU over CPU** (steady state) | **16.21×** | D | `t_CPU / t_GPU` = B-03 / B-06 = 212.113 / 13.086 | Matches `results/comparison.md:32` (16.2x). The GPU is 16.2× faster than the CPU. Uses B-06: see discrepancy 16 and "GPU session variability (Colab)". |
| B-11 | Baseline speedup | **TPU over GPU** (steady state) | **27.84×** | D | `t_GPU / t_TPU` = B-06 / B-09 = 13.086 / 0.47 | Matches `results/comparison.md:36` (27.8x). The TPU is 27.8× faster than the T4. Uses B-06: see discrepancy 16 and "GPU session variability (Colab)". |
| B-12 | Baseline speedup | **TPU over CPU** (steady state) | **451.3×** | D | `t_CPU / t_TPU` = B-03 / B-09 = 212.113 / 0.47 | Matches `results/comparison.md:33` (451x). Consistency check: 16.21 × 27.84 = 451.3 |
| B-13 | Baseline ratio | CPU 1st predict / steady state | 1.28× | D | B-02 / B-03 | `results/comparison.md:46` |
| B-14 | Baseline ratio | GPU 1st predict / steady state | 7.46× | D | B-05 / B-06 | `results/comparison.md:47`. Uses B-06: see discrepancy 16 and "GPU session variability (Colab)". |
| B-15 | Baseline ratio | TPU 1st predict / steady state | 59.1× | D | B-08 / B-09 | `results/comparison.md:48` |
| **C** | **Cost analysis** | | | | | |
| C-01 | Cost | TPU v5e price | $1.20 / chip-hour | I | `results/cost_analysis.md:5` | Google Cloud on-demand, Aug 2026 |
| C-02 | Cost | TPU full pod price (8 chips) | $9.60 / hour | I | `results/cost_analysis.md:8-9` | = 8 × C-01 |
| C-03 | Cost | GPU T4 price | $0.35 / hour | I | `results/cost_analysis.md:10-11` | |
| C-04 | Cost | CPU price | $0.19 / hour | I | `results/cost_analysis.md:12` | approx. n2-standard-4 |
| C-05 | Cost | CPU predictions / hour | 16.97 | D | 3600 / B-03 | md rounds to 17.0 (`cost_analysis.md:18`) |
| C-06 | Cost | GPU predictions / hour | 275.1 | D | 3600 / B-06 | `cost_analysis.md:19` |
| C-07 | Cost | TPU predictions / hour | 7,660 | D | 3600 / B-09 | `cost_analysis.md:20` |
| C-08 | Cost | CPU cost / 1,000 predictions | $11.19 | D | C-04 / C-05 × 1000 | `cost_analysis.md:18` |
| C-09 | Cost | GPU cost / 1,000 predictions | $1.27 | D | C-03 / C-06 × 1000 = 1.2722 | `cost_analysis.md:19` |
| C-10 | Cost | TPU full pod cost / 1,000 predictions | $1.25 | D | C-02 / C-07 × 1000 = 1.2533 | `cost_analysis.md:20` |
| C-11 | Cost | TPU hypothetical 1-chip cost / 1,000 | $0.157 | D | C-01 / C-07 × 1000 = 0.1567 | `cost_analysis.md:44` |
| C-12 | Cost | GPU cost / TPU 1-chip cost | 8.12× | D | C-09 / C-11 | md: "about 8x cheaper" (`cost_analysis.md:45`) |
| C-13 | Cost | Pod hourly price / GPU hourly price | 27.4× | D | C-02 / C-03 | md: "27x" (`cost_analysis.md:27`) |
| C-14 | Cost | Idle pod fraction (baseline workload) | 87.5% (7/8 chips) | D | from CV-10 | md: "~87%" (`cost_analysis.md:50`) |
| C-15 | Cost | Pod cost / 1,000 with pmap multi-query | $0.181 | D | C-02 / (MQ-05 × 3600) × 1000 | **Not stated in any source file** (`sharding.md:45-46` says only "roughly 7x" lower) |
| **SL** | **Sequence-length sweep (TPU, recycle=0)** | | | | | |
| SL-01 | Seq length 60 | init_params | 36.12 s | M | `results/sweep/sequence_length_sweep.json → [0].init_params_seconds` | |
| SL-02 | Seq length 60 | 1st predict | 25.95 s | M | `… → [0].first_predict_seconds` | |
| SL-03 | Seq length 60 | steady state | 0.208 s | M | `… → [0].steady_state_seconds` | |
| SL-04 | Seq length 120 | init_params | 36.50 s | M | `… → [1].init_params_seconds` | |
| SL-05 | Seq length 120 | 1st predict | 24.88 s | M | `… → [1].first_predict_seconds` | |
| SL-06 | Seq length 120 | steady state | 0.408 s | M | `… → [1].steady_state_seconds` | |
| SL-07 | Seq length 250 | init_params | 36.10 s | M | `… → [2].init_params_seconds` | |
| SL-08 | Seq length 250 | 1st predict | 29.37 s | M | `… → [2].first_predict_seconds` | |
| SL-09 | Seq length 250 | steady state | 1.170 s | M | `… → [2].steady_state_seconds` | |
| SL-10 | Seq length 500 | init_params | 35.49 s | M | `… → [3].init_params_seconds` | |
| SL-11 | Seq length 500 | 1st predict | 31.46 s | M | `… → [3].first_predict_seconds` | |
| SL-12 | Seq length 500 | steady state | 3.031 s | M | `… → [3].steady_state_seconds` | |
| SL-13 | Seq length | time factor 120 vs 60 | 1.96× (length 2.0×) | D | SL-06 / SL-03 | md rounds to 2.0x (`sweep/README.md:42`) |
| SL-14 | Seq length | time factor 250 vs 60 | 5.63× (length 4.17×) | D | SL-09 / SL-03 | md 5.6x / 4.2x |
| SL-15 | Seq length | time factor 500 vs 60 | 14.57× (length 8.33×) | D | SL-12 / SL-03 | md 14.6x / 8.3x |
| SL-16 | Seq length | time ∝ length^k, least-squares fit on 4 points | k = 1.28 | D | log-log fit of SL-03/06/09/12 | Endpoint-only estimate (60→500) gives k = 1.26. `scaling_law.md:35` quotes 1.28 |
| **RD** | **Recycle-depth sweep (TPU, 118 res)** | | | | | |
| RD-01 | Recycle 0 | init_params | 35.74 s | M | `results/sweep/recycle_depth_sweep.json → [0].init_params_seconds` | |
| RD-02 | Recycle 0 | 1st predict | 27.66 s | M | `… → [0].first_predict_seconds` | |
| RD-03 | Recycle 0 | steady state | 0.469 s | M | `… → [0].steady_state_seconds` | |
| RD-04 | Recycle 1 | init_params | 37.71 s | M | `… → [1].init_params_seconds` | |
| RD-05 | Recycle 1 | 1st predict | 55.46 s | M | `… → [1].first_predict_seconds` | |
| RD-06 | Recycle 1 | steady state | 0.933 s | M | `… → [1].steady_state_seconds` | |
| RD-07 | Recycle 3 | init_params | 40.31 s | M | `… → [2].init_params_seconds` | |
| RD-08 | Recycle 3 | 1st predict | 54.94 s | M | `… → [2].first_predict_seconds` | |
| RD-09 | Recycle 3 | steady state | 1.845 s | M | `… → [2].steady_state_seconds` | |
| RD-10 | Recycle | steady factor rec1 / rec0 | 1.99× | D | RD-06 / RD-03 | |
| RD-11 | Recycle | steady factor rec3 / rec0 | 3.93× | D | RD-09 / RD-03 | |
| RD-12 | Recycle | marginal time per extra recycle | 0.459 s | D | (RD-09 − RD-03) / 3 | 0→1 alone: 0.464 s |
| **CV** | **Chip visibility (TPU, single query)** | | | | | |
| CV-01 | 1 chip visible | init_params | 36.32 s | M | `results/sweep/chip_visibility_sweep.json → [0].init_params_seconds` | |
| CV-02 | 1 chip visible | 1st predict | 27.37 s | M | `… → [0].first_predict_seconds` | |
| CV-03 | 1 chip visible | steady state | 0.469 s | M | `… → [0].steady_state_seconds` | |
| CV-04 | 1 chip visible | HBM in use / peak (TPU_0) | 463 MB / 486 MB | M | `… → [0].memory_stats_per_device[0]` | bytes: 463,000,000 / 486,000,000 |
| CV-05 | 8 chips visible | init_params | 36.18 s | M | `… → [1].init_params_seconds` | |
| CV-06 | 8 chips visible | 1st predict | 27.43 s | M | `… → [1].first_predict_seconds` | |
| CV-07 | 8 chips visible | steady state | 0.470 s | M | `… → [1].steady_state_seconds` | |
| CV-08 | 8 chips visible | HBM in use / peak (TPU_0) | 463 MB / 486 MB | M | `… → [1].memory_stats_per_device[0]` | |
| CV-09 | 8 chips visible | HBM in use (TPU_1 … TPU_7) | 0 MB each | M | `… → [1].memory_stats_per_device[1..7]` | Single query uses exactly 1 chip |
| CV-10 | 8 chips visible | chips with nonzero HBM | 1 / 8 | D | count of CV-08/CV-09 | |
| CV-11 | 2- and 4-chip subsets | outcome | failed (topology mismatch; libtpu `SLICE_FAILURE_INIT_ERROR`) | R | `results/sweep/chip_visibility.md:32-46` | No numeric result |
| **PR** | **Precision (TPU)** | | | | | |
| PR-01 | float32 | init_params | 36.52 s | M | `results/sweep/precision_sweep.json → [0].init_params_seconds` | |
| PR-02 | float32 | 1st predict | 27.88 s | M | `… → [0].first_predict_seconds` | |
| PR-03 | float32 | steady state | 0.470 s | M | `… → [0].steady_state_seconds` | |
| PR-04 | float32 | HBM in use / peak | 463 MB / 486 MB | M | `… → [0].hbm_bytes_in_use`, `hbm_peak_bytes_in_use` | |
| PR-05 | bfloat16 | init_params | 37.73 s | M | `… → [1].init_params_seconds` | |
| PR-06 | bfloat16 | 1st predict | 26.39 s | M | `… → [1].first_predict_seconds` | |
| PR-07 | bfloat16 | steady state | 0.469 s | M | `… → [1].steady_state_seconds` | |
| PR-08 | bfloat16 | HBM in use / peak | 277 MB / 576 MB | M | `… → [1].hbm_bytes_in_use`, `hbm_peak_bytes_in_use` | |
| PR-09 | Precision | Δ 1st predict (bf16 vs fp32) | −5.3% | D | PR-06 / PR-02 − 1 | md: −5% |
| PR-10 | Precision | Δ steady state | −0.2% (≈ 0) | D | PR-07 / PR-03 − 1 | |
| PR-11 | Precision | Δ HBM in use | −40.2% | D | 277 / 463 − 1 | md: −40% |
| PR-12 | Precision | Δ HBM peak | +18.5% | D | 576 / 486 − 1 | md: +18% |
| PR-13 | Precision | bf16 parameter cast duration | ~2.3 s | R | `results/sweep/precision.md:34-35` | Log not in repo |
| **CC** | **Persistent compilation cache (TPU, two separate processes)** | | | | | |
| CC-01 | Cache cold | init_params | 37.68 s | M | `results/sweep/compilation_cache.json → cold.init_params_seconds` | |
| CC-02 | Cache cold | 1st predict | 28.80 s | M | `… → cold.first_predict_seconds` | |
| CC-03 | Cache cold | steady state | 0.470 s | M | `… → cold.steady_state_seconds` | |
| CC-04 | Cache warm | init_params | 5.53 s | M | `… → warm.init_params_seconds` | |
| CC-05 | Cache warm | 1st predict | 15.19 s | M | `… → warm.first_predict_seconds` | |
| CC-06 | Cache warm | steady state | 0.469 s | M | `… → warm.steady_state_seconds` | |
| CC-07 | Cache | init_params speedup (warm over cold) | 6.81× | M / D | `… → speedup_init_params`; = CC-01 / CC-04 | |
| CC-08 | Cache | 1st predict speedup (warm over cold) | 1.90× | M / D | `… → speedup_first_predict`; = CC-02 / CC-05 | |
| **TR** | **XLA profiler trace (TPU, first predict)** | | | | | |
| TR-01 | Trace | `PjitFunction(apply_fn)` wall duration | 16.56 s | R | `profiling/trace_analysis.md:30` | Raw `.trace.json.gz` not in repo |
| TR-02 | Trace | `PjitFunction(apply_fn)` self time | 440 ns | R | `profiling/trace_analysis.md:30` | |
| TR-03 | Trace | `pjit.py:250 cache_miss` wall duration | 16.56 s | R | `profiling/trace_analysis.md:31` | |
| TR-04 | Trace | `cache_miss` self time | 12.55 s | R | `profiling/trace_analysis.md:31` | |
| TR-05 | Trace | `cache_miss` self time share of call | 75.8% | D | TR-04 / TR-01 | md: ~76% |
| **RR** | **Repeated runs (TPU, n = 3)** | | | | | |
| RR-01 | Repeats | init_params, reps 1 / 2 / 3 | 37.54 / 37.02 / 37.01 s | M | `results/sweep/repeated_runs.json → runs[].init_params_seconds` | |
| RR-02 | Repeats | 1st predict, reps 1 / 2 / 3 | 27.56 / 27.96 / 27.43 s | M | `… → runs[].first_predict_seconds` | |
| RR-03 | Repeats | steady state, reps 1 / 2 / 3 | 0.470 / 0.469 / 0.469 s | M | `… → runs[].steady_state_seconds` | |
| RR-04 | Repeats | init_params mean ± sd (CV) | 37.19 ± 0.303 s (0.82%) | M | `… → init_params_seconds` | Recomputed: matches |
| RR-05 | Repeats | 1st predict mean ± sd (CV) | 27.65 ± 0.276 s (1.00%) | M | `… → first_predict_seconds` | Recomputed: matches. CV is 0.999%, right at the "under 1%" claim |
| RR-06 | Repeats | steady state mean ± sd (CV) | 0.4693 ± 0.0006 s (0.12%) | M | `… → steady_state_seconds` | Recomputed: matches. See discrepancy 2 |
| **MC** | **Model comparison (TPU)** | | | | | |
| MC-01 | model_3 | init / 1st predict / steady | 36.67 / 28.72 / 0.471 s | M | `results/sweep/model_comparison.json → [0]` | |
| MC-02 | model_4 | init / 1st predict / steady | 35.75 / 28.24 / 0.472 s | M | `… → [1]` | |
| MC-03 | model_5 | init / 1st predict / steady | 35.62 / 28.00 / 0.403 s | M | `… → [2]` | |
| MC-04 | Models | model_5 steady-state time vs model_3 | −14.4% | D | MC-03 / MC-01 − 1 | Equivalently model_5 over model_3 = 1.17×; md says "~15% faster" |
| **VM** | **Multi-query batching with `jax.vmap` (TPU, negative result)** | | | | | |
| VM-01 | vmap B=1 | steady / proteins·s⁻¹ / s per protein / HBM TPU_0 | 0.4875 s / 2.051 / 0.4875 s / 487 MB | M | `results/sweep/batching_sweep.json → [0]` | |
| VM-02 | vmap B=2 | steady / proteins·s⁻¹ / s per protein / HBM TPU_0 | 1.0374 s / 1.928 / 0.5187 s / 532 MB | M | `… → [1]` | |
| VM-03 | vmap B=4 | steady / proteins·s⁻¹ / s per protein / HBM TPU_0 | 2.6905 s / 1.487 / 0.6726 s / 630 MB | M | `… → [2]` | |
| VM-04 | vmap B=8 | steady / proteins·s⁻¹ / s per protein / HBM TPU_0 | 5.3053 s / 1.508 / 0.6632 s / 821 MB | M | `… → [3]` | |
| VM-05 | vmap | throughput B=8 relative to B=1 | 0.735× | D | VM-04 / VM-01 throughput | Batching lowers throughput. Only TPU_0 used (`batching.md:36-38`, R) |
| **MQ** | **Multi-chip experiment 1: `jax.pmap` MULTI-QUERY. 8 independent proteins, one per chip** | | | | | |
| MQ-01 | pmap multi-query | single-protein baseline, steady state | 0.470 s | M | `results/sweep/sharding.json → pmap_data_parallelism.single_protein_baseline_seconds` | Same value as CV-07; the file does not say which run it comes from |
| MQ-02 | pmap multi-query | 8 proteins / 8 chips, steady state (whole call) | 0.5435 s | M | `… → pmap_8chips_steady_state_seconds` | No compile time recorded for this experiment |
| MQ-03 | pmap multi-query | amortized time per protein | 0.0679 s | M / D | `… → per_protein_amortized_seconds`; = MQ-02 / 8 | |
| MQ-04 | pmap multi-query | baseline throughput | 2.128 proteins/s | D | 1 / MQ-01 | md: 2.13 |
| MQ-05 | pmap multi-query | throughput | 14.718 proteins/s | M / D | `… → throughput_proteins_per_sec`; = 8 / MQ-02 | md: 14.72 |
| MQ-06 | pmap multi-query | **throughput speedup (pmap 8 chips over 1-protein baseline)** | **6.92×** | M / D | `… → speedup_vs_single_protein_baseline`; = MQ-05 / MQ-04 = 6.918 | |
| MQ-07 | pmap multi-query | parallel efficiency | 86.5% | D | MQ-06 / 8 | Not stated in source |
| MQ-08 | pmap multi-query | HBM per chip, TPU_0 … TPU_7 | 644, 452, 445, 447, 447, 446, 454, 469 MB | M | `… → memory_per_chip_mb` | Range 445–644 MB (TPU_0 = 644). See discrepancy 1 |
| **GS** | **Multi-chip experiment 2: GSPMD auto-mesh, single protein (replication, not sharding)** | | | | | |
| GS-01 | Auto-mesh run 1 | init / 1st predict / steady | 37.56 / 14.76 / 0.472 s | M | `results/sweep/sharding.json → mesh_auto_sharding.run_1` | |
| GS-02 | Auto-mesh run 1 | HBM per chip | 463 MB | M | `… → run_1.memory_per_chip_mb` | = single-chip baseline (`single_chip_baseline_mb` = 463). One scalar per run; the per-device list was not saved. "All 8 identical", as earlier write-ups put it, was prose only (R): no per-chip list backs it |
| GS-03 | Auto-mesh run 2 | init / 1st predict / steady | 36.81 / 14.46 / 0.473 s | M | `… → mesh_auto_sharding.run_2` | |
| GS-04 | Auto-mesh run 2 | HBM per chip | 463 MB | M | `… → run_2.memory_per_chip_mb` | One scalar per run; the per-device list was not saved. "All 8 identical" was prose only (R) |
| **EN** | **Multi-chip experiment 3: `jax.pmap` + `jax.lax.pmean` ENSEMBLE. 8 ensemble members of ONE query, one per chip** | | | | | |
| EN-01 | pmap+pmean ensemble | query length / ensemble members / devices | 118 res / 8 / 8 | M | `results/sweep/ensemble_shard.json → num_residues, num_ensemble, num_devices` | |
| EN-02 | pmap+pmean ensemble | 1st call (compile + run) | 16.61 s | M | `… → first_compile_and_run_seconds` | |
| EN-03 | pmap+pmean ensemble | steady state (whole call, 1 query, 8 members) | 0.5381 s | M | `… → second_steady_state_seconds` | Output is **one** averaged prediction, not 8 proteins |
| EN-04 | pmap+pmean ensemble | chips with nonzero memory | 8 / 8 | M | `… → num_chips_with_nonzero_memory` | |
| EN-05 | pmap+pmean ensemble | `pmean` reduction consistent across chips | true | M | `… → pmean_reduction_consistent_across_chips` | |
| EN-06 | pmap+pmean ensemble | HBM in use / peak, TPU_0 | 624 / 731 MB | M | `… → memory_stats_per_device_mb[0]` | |
| EN-07 | pmap+pmean ensemble | HBM in use / peak, TPU_1 … TPU_5 | 427 / 450 MB each | M | `… → memory_stats_per_device_mb[1..5]` | |
| EN-08 | pmap+pmean ensemble | HBM in use / peak, TPU_6, TPU_7 | 450 / 450 MB each | M | `… → memory_stats_per_device_mb[6..7]` | |
| EN-09 | pmap+pmean ensemble | speedup | **not measured** | none | none | No sequential `num_ensemble=8` baseline exists in the data. Do not derive one. |
| **SC** | **Scaling law: pmap multi-query throughput(chips, length) (TPU)** | | | | | Uses the multi-query pmap script from experiment 1 (`scaling_law.md:5`), **not** the ensemble one |
| SC-01 | Grid chips=1 | throughput at length 100 / 250 / 500 / 1000 | 2.935 / 0.864 / 0.332 / 0.066 proteins/s | M | `results/sweep/scaling_law_data.json → grid[0..3]` | |
| SC-02 | Grid chips=2 | throughput at length 100 / 250 / 500 / 1000 | 5.509 / 1.694 / 0.659 / 0.131 proteins/s | M | `… → grid[4..7]` | |
| SC-03 | Grid chips=4 | throughput at length 100 / 250 / 500 / 1000 | 10.417 / 3.344 / 1.309 / 0.262 proteins/s | M | `… → grid[8..11]` | |
| SC-04 | Grid chips=8 | throughput at length 100 / 250 / 500 / 1000 | 19.153 / 6.450 / 2.575 / 0.522 proteins/s | M | `… → grid[12..15]` | |
| SC-05 | Fit | prefactor a | 4527.77 | M | `… → fitted_power_law.formula` | Refit by OLS in log space: 4527.77 (matches) |
| SC-06 | Fit | chip exponent b | 0.9631 | M | `… → fitted_power_law.chip_exponent` | Refit: 0.9631 |
| SC-07 | Fit | length exponent c | −1.5724 | M | `… → fitted_power_law.length_exponent` | Refit: −1.5724 |
| SC-08 | Fit | R² (log-log) | 0.9813 | M | `… → fitted_power_law.r_squared_log_log` | Refit: 0.9813 |
| SC-09 | Fit | fit / measured at length 100 (chips 1, 2, 4, 8) | 1.105, 1.148, 1.184, 1.255 | D | exp(fit) / SC-01..04 | Over-predicts 10–26% |
| SC-10 | Fit | fit / measured at length 250 | 0.889, 0.884, 0.873, 0.882 | D | same | Under-predicts 11–13% |
| SC-11 | Fit | fit / measured at length 500 | 0.778, 0.764, 0.750, 0.743 | D | same | Under-predicts 22–26% |
| SC-12 | Fit | fit / measured at length 1000 | 1.316, 1.293, 1.260, 1.233 | D | same | Over-predicts 23–32% |
| SC-13 | Measured speedup | 8 chips over 1 chip at length 100 / 250 / 500 / 1000 | 6.53× / 7.47× / 7.76× / 7.91× | D | SC-04 / SC-01 | Not stated in source |
| SC-14 | Measured speedup | 4 chips over 1 chip at length 100 / 250 / 500 / 1000 | 3.55× / 3.87× / 3.94× / 3.97× | D | SC-03 / SC-01 | Not stated in source |
| SC-15 | Measured speedup | 2 chips over 1 chip at length 100 / 250 / 500 / 1000 | 1.88× / 1.96× / 1.98× / 1.98× | D | SC-02 / SC-01 | Not stated in source |
| SC-16 | Cost | $/1000 predictions, length 100, chips 1 / 2 / 4 / 8 | $0.114 / $0.121 / $0.128 / $0.139 | D | chips × C-01 / (throughput × 3600) × 1000 on **measured** grid | Matches `scaling_law.md:48` |
| SC-17 | Cost | $/1000 predictions, length 250, chips 1 / 2 / 4 / 8 | $0.386 / $0.394 / $0.399 / $0.413 | D | same | `scaling_law.md:49` |
| SC-18 | Cost | $/1000 predictions, length 500, chips 1 / 2 / 4 / 8 | $1.004 / $1.012 / $1.019 / $1.036 | D | same | `scaling_law.md:50` |
| SC-19 | Cost | $/1000 predictions, length 1000, chips 1 / 2 / 4 / 8 | $5.051 / $5.089 / $5.089 / $5.109 | D | same | `scaling_law.md:51`. Low precision: throughput at 1 chip has 2 significant figures (0.066) |
| **A3** | **AlphaFold3 side-investigation (118 res, seed 1, 5 diffusion samples, real weights, empty MSA)** | | | | | |
| A3-01 | AF3 Colab CPU | process wall-clock total | 2453.99 s | M | `results/result_af3_cpu-colab.json → wall_clock_total_seconds` (formerly `total_inference_seconds`) | Includes start-up, model build, parameter loading, featurisation, JIT compilation, inference, extraction and output writing. Per sample: 490.80 s (`wall_clock_seconds_per_sample`), the previously reported value |
| A3-02 | AF3 Colab CPU | **seconds per sample (model inference)** | **480.28 s** | M / D | `… → seconds_per_sample` = A3-25 / 5 | **Still includes JIT compilation.** Replaces 490.80 s (wall-clock / 5) |
| A3-03 | AF3 Colab CPU | best ranking_score / ptm / fraction_disordered / has_clash | 0.41 / 0.23 / 0.37 / 0.0 | M | `… → best_ranking_score, ptm, fraction_disordered, has_clash` | Same values in `sweep/af3_toy_test_cpu-colab_summary_confidences.json` |
| A3-04 | AF3 Colab GPU T4 | process wall-clock total | 114.12 s | M | `results/result_af3_gpu-t4.json → wall_clock_total_seconds` (formerly `total_inference_seconds`) | Same scope as A3-01. Per sample: 22.82 s (`wall_clock_seconds_per_sample`), the previously reported value |
| A3-05 | AF3 Colab GPU T4 | **seconds per sample (model inference)** | **17.24 s** | M / D | `… → seconds_per_sample` = A3-28 / 5 | **Still includes JIT compilation.** Replaces 22.82 s (wall-clock / 5) |
| A3-06 | AF3 Colab GPU T4 | best ranking_score / ptm / fraction_disordered / has_clash | 0.33 / 0.23 / 0.21 / 0.0 | M | `… → best_ranking_score, ptm, fraction_disordered, has_clash` | Same values in `sweep/af3_toy_test_gpu-t4_summary_confidences.json` |
| A3-07 | AF3 Stanford CPU | ranking_score / ptm / fraction_disordered / has_clash / iptm | 0.41 / 0.23 / 0.37 / 0.0 / null | M | `results/sweep/af3_toy_test_summary_confidences.json` | `chain_pair_pae_min` = 0.76 in all three runs |
| A3-08 | AF3 Stanford CPU | seconds per sample | 78.43 s | R | `results/sweep/af3_comparison.md:176-177` | No timing JSON for this run in repo; hardware unspecified |
| A3-09 | AF3 Stanford CPU | featurising | 6.66 s | R | `results/sweep/af3_comparison.md:147` | |
| A3-10 | AF3 | **GPU over CPU** (Colab, model inference) | **27.86×** | D | `t_CPU / t_GPU` = A3-25 / A3-28 = 2401.38 / 86.21 | md: 27.9x. Previously 21.50× (wall-clock). Compilation is included on both sides |
| A3-11 | AF3 vs AF2 | AF3 per sample / AF2 steady state, Colab CPU | 2.26× (AF3 slower) | D | A3-02 / B-03 | md: 2.26x. Previously 2.31×. **Not interpretable; do not report.** AF3 includes compilation and runs 11 trunk passes (vs AF2's 1) shared across 5 samples. See discrepancies 6 and 15 |
| A3-12 | AF3 vs AF2 | AF3 per sample / AF2 steady state, Colab T4 | 1.32× (AF3 slower) | D | A3-05 / B-06 | md: 1.32x. Previously 1.74×. **Not interpretable; do not report.** AF3 includes compilation and runs 11 trunk passes (vs AF2's 1) shared across 5 samples. See discrepancies 6 and 15 |
| A3-13 | AF3 | Colab CPU per sample / Stanford CPU per sample | 6.26× (Colab slower) | D | wall-clock per sample (490.80) / A3-08 | Depends on an R value whose measurement method is not recorded. On the inference-only figure (A3-02): 6.12×. md: 6.3x |
| A3-14 | AF3 ranking | Stanford CPU, samples 0–4 | 0.26671 / 0.41295 / 0.38284 / 0.31773 / 0.32173 | M | `results/sweep/af3_toy_test_ranking_scores.csv` | Best = sample 1 |
| A3-15 | AF3 ranking | Colab CPU, samples 0–4 | 0.27116 / 0.41329 / 0.38312 / 0.31759 / 0.32182 | M | `results/sweep/af3_toy_test_cpu-colab_ranking_scores.csv` | Best = sample 1 |
| A3-16 | AF3 ranking | Colab GPU, samples 0–4 | 0.25646 / 0.33112 / 0.25924 / 0.31599 / 0.32586 | M | `results/sweep/af3_toy_test_gpu-t4_ranking_scores.csv` | Best = sample 1 |
| A3-17 | AF3 ranking | mean ± sd (CV), Stanford / Colab CPU / Colab GPU | 0.340 ± 0.058 (16.98%) / 0.341 ± 0.057 (16.57%) / 0.298 ± 0.037 (12.37%) | D | from A3-14..16, sample sd | Matches `af3_comparison.md:229-231` |
| A3-18 | AF3 reproducibility | Colab CPU vs Stanford CPU, per sample Δ | +1.67% / +0.08% / +0.07% / −0.04% / +0.03% | D | A3-15 / A3-14 − 1 | Same backend, different machine |
| A3-19 | AF3 reproducibility | Colab GPU vs Colab CPU, per sample Δ | −5.42% / −19.88% / −32.33% / −0.51% / +1.26% | D | A3-16 / A3-15 − 1 | Different backend |
| A3-20 | AF3 reproducibility | best ranking_score Δ, GPU vs CPU (summary JSON) | −19.5% | D | 0.33 / 0.41 − 1 | From rounded summary values; −19.9% from the unrounded CSVs |
| A3-21 | AF3 reproducibility | fraction_disordered Δ, GPU vs CPU | −43.2% | D | 0.21 / 0.37 − 1 | |
| A3-22 | AF3 | weights file size (decompressed) | 1,146,811,260 bytes (1.147 GB / 1.068 GiB) | M | `results/sweep/af3_tpu_attempt.log:18` | md: "~1.15 GB" |
| A3-23 | AF3 | weights size vs AF2 | 3.3× | R / D | A3-22 / 350 MB | AF2 "~350 MB" is R only (`af3_comparison.md:31`) |
| A3-24 | AF3 TPU attempt | outcome / wall time to failure | `--jax_backend=tpu` rejected at flag parsing / 5.588 s real (18.874 s user, 0.447 s sys) | M | `results/sweep/af3_tpu_attempt.log:32-37` | Valid values: cpu, gpu, mps |
| A3-25 | AF3 Colab CPU | model inference, seed 1, 5 samples | 2401.38 s | M | `results/result_af3_cpu-colab.json → model_inference_seconds`; AF3 log in `notebooks/af3_cpu_colab.ipynb` cell 14 | Includes JIT compilation |
| A3-26 | AF3 Colab CPU | featurisation / sample extraction | 12.92 s / 0.33 s | M | `… → featurisation_seconds, sample_extraction_seconds`; same log | |
| A3-27 | AF3 Colab CPU | non-inference overhead, total / per sample | 52.61 s / 10.52 s | D | A3-01 − A3-25; ÷ 5 | Start-up, model build, parameter load, featurisation, extraction, output writing. **Not** compilation |
| A3-28 | AF3 Colab GPU T4 | model inference, seed 1, 5 samples | 86.21 s | M | `results/result_af3_gpu-t4.json → model_inference_seconds`; AF3 log in `notebooks/af3_gpu_colab.ipynb` cell 14 | Includes JIT compilation |
| A3-29 | AF3 Colab GPU T4 | featurisation / sample extraction | 11.09 s / 0.18 s | M | `… → featurisation_seconds, sample_extraction_seconds`; same log | |
| A3-30 | AF3 Colab GPU T4 | non-inference overhead, total / per sample | 27.91 s / 5.58 s | D | A3-04 − A3-28; ÷ 5 | Same scope as A3-27. **Not** compilation |

## The multi-chip experiments are separate. Do not merge them.

There are three distinct multi-chip TPU results, plus the `vmap` negative
result (VM) that runs on one chip. Keep them apart in the paper.

- **Experiment 1: pmap multi-query (MQ-01…MQ-08, and SC which reuses its script).**
  Eight *different, independent* proteins, one per chip. The output is 8 structures.
  The metric is **throughput**: 14.72 proteins/s against 2.13 for the baseline,
  a **6.92×** speedup. HBM per chip is 445–644 MB. No `pmean` is involved.
- **Experiment 2: GSPMD auto-mesh (GS-01…GS-04).** One protein under an Auto mesh.
  The run records 463 MB per chip, the single-chip footprint (one figure; the per-device list was not saved), so this is **replication, not sharding**.
  Steady state (0.472 / 0.473 s) is unchanged from single-chip.
- **Experiment 3: pmap + pmean ensemble (EN-01…EN-09).** **One** query whose 8
  ensemble members (different `random_seed`s) run one per chip and are averaged
  with a real `jax.lax.pmean` collective. The output is **one** averaged prediction.
  The results are 8/8 chips used, a consistent `pmean`, 0.5381 s steady state and
  16.61 s first call, with HBM of 427–624 MB in use and 450–731 MB peak.
  **No speedup was measured**, because there is no sequential `num_ensemble=8` run.
  Its 0.5381 s must not be compared with, or averaged with, MQ-02's 0.5435 s: they
  are different workloads that produce different outputs.
- In addition, the ensemble members use a trivial single-sequence MSA, so they are
  near-identical inputs (`ensemble_shard.md:82-88`). The experiment demonstrates the
  mechanism, not a scientifically meaningful ensemble.

## Discrepancies and caveats found while cross-checking

1. **pmap multi-query memory range is misreported.** `sweep/sharding.md:12-15` (diagram)
   and `:38` say "445-469 MB across all 8 chips". `sharding.json → memory_per_chip_mb`
   has TPU_0 = **644 MB**. The 445–469 MB range only holds for TPU_1…TPU_7 (MQ-08).

   **RISOLTO:** fixed in `results/sweep/sharding.md`. The diagram (lines 10-15) now shows
   the real per-chip values (chip 0 = 644 MB, chip 1 = 452 MB, chips 2-6 = 445-454 MB,
   chip 7 = 469 MB) and labels each chip as running its own forward pass. Line 35 now says
   445-469 MB on chips 1-7 and 644 MB on `TPU_0`, stressing that the values differ per chip
   rather than matching GSPMD's single 463 MB per-chip figure. A new paragraph at line 37 states that
   the data does not establish why chip 0 holds more memory, and gives one plausible,
   unmeasured contributor (`init_params` and `jnp.stack` run outside `pmap` on the JAX
   default device). The line numbers in the problem text above refer to the pre-fix
   version at commit `96a2186`.
2. **Run-to-run noise in `model_comparison.md:24-25` is misquoted.** It says "~0.06%
   stdev", but the steady-state CV from `repeated_runs.json` is 0.0006 / 0.4693 = **0.12%**
   (RR-06). "0.0006" appears to be the absolute stdev in seconds, read as a percent.
   The conclusion still holds: model_5's −14.4% is far above the noise.

   **RISOLTO:** fixed in `results/sweep/model_comparison.md:24-25`, which now reads
   "~0.12% coefficient of variation, i.e. 0.0006s stdev on a 0.469s mean". No measured
   value was changed.
3. **The trace ↔ cache link in `profiling/trace_analysis.md:70-74` is not supported by
   the numbers.** The trace covers the first `predict()` (TR-01, 16.56 s). The 6.81×
   speedup (CC-07) is on `init_params` (37.68 → 5.53 s). 16.56 / 5.53 = 2.99, and the
   cache speedup on first predict is only 1.90× (CC-08). The "matches almost exactly"
   claim should not go into the paper. The trace's 16.56 s is also shorter than every
   measured cold first-predict (27.4–28.8 s), so it is a different run or timing window,
   and the raw trace file is not in the repo.

   **RISOLTO:** fixed in `profiling/trace_analysis.md`. The table header (line 28) now
   reads "% of traced `apply_fn` call" instead of "% of first predict() call", and lines
   37-38 speak of the traced `apply_fn` call rather than the entire first-call cost. The
   "matching the 6.8x measured speedup almost exactly" paragraph was replaced (now lines
   68-85). The new text:
   - compares the trace with the cache's first-`predict` result (1.90×, about 13.6 s saved);
   - says the 6.81× applies to `init_params`, which the trace does not cover;
   - notes that trace and cache are separate runs, and that the 16.56 s span is shorter
     than the 27.37–28.80 s cold first predicts.

   It also says, marked as plausible and not measured, that JAX's persistent cache stores
   XLA binaries while a fresh process still re-traces to a jaxpr. The raw trace file is
   still not in the repo. The line numbers in the problem text above (70-74) refer to the
   pre-fix version at commit `96a2186`.
4. **`sweep/sharding.md:74-77` calls auto-mesh timings "statistically indistinguishable"
   from baseline.** First predict was 14.76 / 14.46 s (GS-01/03), against ~27.4 s
   single-chip (CV-06), a 1.9× difference. The md attributes this to a warm XLA cache.
   That is a plausible explanation, not a measurement.
5. **Scaling-law residual ranges are approximate** in `scaling_law.md:21-23` ("over
   ~20–30% at extremes, under ~12–25% in the middle"). The recomputed ranges are:
   over-prediction of 10.5–25.5% at L=100 and 23.3–31.6% at L=1000; under-prediction of
   11.2–12.7% at L=250 and 22.2–25.7% at L=500 (SC-09…SC-12).
6. **The AF3-vs-AF2 ratios (A3-11, A3-12, and 21.5× vs 16.2×) mix definitions.**
   AF3 time is `total_inference_seconds / 5`, which per `af3_comparison.md:148` includes
   setup and compile on GPU. AF2 time is the warm second call. The ratios are real
   arithmetic but not like-for-like, so qualify them in the paper.

   **AGGIORNATO (partially resolved):** the AF3 per-sample figures now use AF3's own
   model-inference time / 5 (A3-02 = 480.28 s, A3-05 = 17.24 s).
   - **What changed:** the non-inference overhead (A3-27: 52.61 s; A3-30: 27.91 s)
     that the old wall-clock figures counted is now excluded. Updated in
     `results/result_af3_*.json` and `results/sweep/af3_comparison.md`.
   - **New values:** A3-11 = 2.26×, A3-12 = 1.32×, A3-10 = 27.9×.
   - **Still open:** the AF3 inference time **still includes JIT compilation**, because
     it is the only model call in the process and there is no warm-up. AF2 is a warm
     second call, so A3-11 and A3-12 remain not like-for-like and overstate AF3's
     steady-state gap by an unmeasured amount.
   - **Line references:** `af3_comparison.md:148` now describes the GPU featurising
     time. The problem text above refers to its content at commit `9f640ad`.
   - **See discrepancy 15.** Isolating compilation would not make A3-11 and A3-12
     usable. AF3 also ran 11 trunk passes (default `--num_recycles 10`) against AF2's 1,
     and its per-sample figure shares one trunk run across 5 diffusion samples.
     Correcting only the recycle mismatch flips the direction of the per-sample ratios.
     Treat both ratios as not interpretable and do not report them.
7. **The TPU baseline speedups (B-11, B-12) compare against an 8-chip slice label, but the
   work runs on one chip** (CV-09/CV-10). "TPU over GPU 27.8×" is effectively *one v5e
   chip* against one T4. State this explicitly.

   **RISOLTO:** fixed in `results/comparison.md`, with no change to any measured value
   or line numbering:
   - line 11: the TPU devices column reads "8 visible, 1 used" instead of "8 chips";
   - lines 33 and 36: the 451x and 27.8x bullets add "running on 1 of the slice's 8
     chips", with "(effectively one v5e chip vs one T4)" on the latter;
   - line 40: a new sentence says the single-query run uses only `TPU_0` (`TPU_1`-`TPU_7`
     at 0 MB), so both ratios compare one v5e chip against one T4 or 2 vCPUs, not the
     full slice;
   - line 67: "the TPU's 27.8x edge" became "the single TPU chip's 27.8x edge".

   The paper must still state this caveat wherever B-11 or B-12 are cited.
8. **`af3_comparison.md:264-267` lists the ubiquitin "AF2 exhibit" as using ESMFold
   weights, with 90.5 mean pLDDT.** That is not an AlphaFold2 result, and nothing in
   `results/` backs it. Keep it out of AF2 numbers.
9. **Numbers with no raw data in the repo (status R):** hardware specs (HW-01…03),
   trace timings (TR-01…04), the bf16 cast duration (PR-13), AF3 Stanford CPU timing and
   featurising (A3-08/09), and the AF2 weight size (A3-23). Cite them as reported, or
   add the raw artifacts.
10. **Minor:**
    - `comparison.md:73` quotes a "~27 to 110s compile cost", which mixes TPU first-predict
      (27.78 s) with GPU init_params (109.16 s).
    - `sweep/README.md` says "Twelve experiments", but the diagram numbers go to 12 while
      the section headings go to 11, with different numbering (e.g. pmap is node 10 but
      section 9). Refer to experiments by name, not number.
    - MQ-01's 0.470 s baseline does not say which run it comes from; it equals CV-07
      and CC-03.
11. **`jax.profiler.trace` inflates first-call times in the scripts that use it.**
    In `spike_tpu_forward_pass.py` and `spike_batch_forward_pass.py` the first call
    runs inside `jax.profiler.trace(...)`, and the timer stops only after the profiler
    context exits (`src/spike_tpu_forward_pass.py:195-200`). The Colab logs
    (`notebooks/alphafold_{cpu,gpu}_benchmark.ipynb` at commit `faeaa4b`, cell 10) show how much of the
    timed value comes after `predict()` has already returned (`model.py:183` exit log →
    script's "First predict() done" log):

    | Backend | Timed 1st predict | After `predict()` returned, still timed | Same interval, 2nd call (no profiler) |
    |---|---|---|---|
    | CPU (B-02) | 271.98 s | **36.00 s** (13%) | 0.0004 s |
    | GPU (B-05) | 97.62 s | **42.02 s** (43%) | 0.0003 s |

    The only code in that interval is `jax.block_until_ready` plus exiting the profiler
    context. `block_until_ready` takes under 1 ms in the second call, so the extra time
    is profiler-trace finalisation, not compilation. **43% of B-05 is profiler overhead.**

    *Script version.* The description above is the script as it produced the August
    and TPU results. Since commit `86fca03` the profiler is optional
    (`--profile_first_predict`, default `True`, so it is still on unless
    `--noprofile_first_predict` is passed) and the steady-state call repeats
    `--num_steady_state_runs` times (default `1`). The September reruns in
    `results/repro/` time 5 steady-state calls with the profiler off; each folder also
    holds one profiler-on run with a single steady-state call.

    Effect on the first-call / steady-state ratios:
    - **B-13 (CPU):** 1.28× → **~1.11×** ((271.98 − 36.00) / 212.113).
    - **B-14 (GPU):** 7.46× → **~4.25×** ((97.62 − 42.02) / 13.086).
    - **B-15 (TPU, 59.1×) and B-08:** the correction **cannot be verified**; no TPU run
      log is in the repo.

    Scripts without the profiler (`spike_pmap_forward_pass.py`,
    `spike_ensemble_shard_forward_pass.py`, `spike_meshshard_forward_pass.py`) time their
    first call without it. Their first-call numbers (GS-01/03: 14.76 / 14.46 s; EN-02:
    16.61 s) are therefore **not comparable** with the ~27–29 s first predicts of CV,
    PR, MC, RR, CC and B-08.
    - This offers an alternative to the "warm XLA cache" explanation in
      `sweep/sharding.md:74-77`: `configs/af_spike_sharding.yaml` runs the `pmap` and
      GSPMD scripts as separate processes without a cache directory.
    - It may also explain why the traced span TR-01 (16.56 s) is shorter than B-08
      (27.78 s).
    - Both are unverified on TPU.
    - CC-02 and CC-05 both include the profiler, so CC-08 compares like with like.
12. **The headline baselines were produced by script versions that were never
    committed.** Git has a single commit of `src/spike_tpu_forward_pass.py`
    (`2755250`, 2026-08-08, the same commit that added the results), so no history
    links a result to a script version.
    - **CPU and GPU (B-01…B-06, B-10, B-13, B-14):** they ran the copy embedded in the
      Colab notebooks (cell 8, identical in both; see the notebooks at commit `faeaa4b`,
      because the current notebooks have been rewritten). That copy differs from `src/`: it
      hard-codes `model_3`, recycle 0 and the toy sequence, and its log lines
      (`spike_tpu_forward_pass.py:111…153`, "Run tag: …") do not exist in `src/`. Its
      timing structure is the same (profiler on the 1st call, `block_until_ready` on
      both).
    - **TPU (B-07…B-09):** `results/result_tpu-v5e-podslice.json` has a field set that
      matches **neither** version. It lacks `host_processor`, which both write. The
      producing version is unknown.
    - **Consequence:** B-01…B-15, including the 16.2× / 27.8× / 451× headline ratios,
      **cannot be reproduced from the repository as it stands**. Running `src/` today
      would use a different script version and an environment whose dependencies are
      mostly unpinned (`paper/sections/methodology.md`, Sections 2 and 5.1).
13. **Slide 5's throughput chart shows values that exist in no data file.**
    The chart in `presentation/AlphaFold_on_Google_TPUs_Pazienza_Lorenzo_Ihab_El_Bani.pdf`
    (slide 5; identical copy in `website/public/presentation/`) plots x = TPU chips
    {1, 2, 4, 8}. Its y-axis is labelled "Throughput (predictions / second)" (log scale,
    10¹–10⁵) and its points **1,720 / 3,350 / 6,520 / 12,900**. Checked against
    `results/sweep/scaling_law_data.json`:
    - **Per second:** the highest measured throughput is **19.153 proteins/s** (8 chips,
      length 100; SC-04). No measured or fitted value comes within two orders of
      magnitude of the labels. The fit (SC-05…07) would reach 1,720/s on one chip only
      at length ≈ 1.9.
    - **Per minute or per hour:** no measured or fitted grid point matches either. The
      nearest candidates are 5–7% off and inconsistent with each other. Three are fit
      values at length 500 but for 2, 4 and 8 chips (not 1, 2, 4, 8), and the fourth is
      a measured 4-chip, length-250 value. The fit gives 1,720/h on one chip only at
      length ≈ 338, which was not tested.
    - **Ratios:** the labels scale as 1 : 1.95 : 3.79 : 7.50, close to the fitted chip
      term `chips^0.963` (1 : 1.95 : 3.80 : 7.41). The labels therefore look derived
      from the fit at an unstated length and unit, not measured.
    - **Dot positions:** in the rendered slide the dots sit visibly below their labels
      (e.g. the 1-chip dot is under 10³ while labelled 1,720).
    - **Measured values to use instead** at length 100: 2.94 / 5.51 / 10.42 / 19.15
      proteins/s (= 10,566 / 19,832 / 37,501 / 68,951 per hour). The measured
      8-over-1-chip speedup is 6.53× at that length (SC-13), not 7.5×. Any throughput
      figure needs its sequence length stated.
14. **Slide 3 attributes the profiler's 76% to the whole first prediction.**
    Slide 3 shows "Cold start 27 s → Steady state 0.47 s" and, under it, a bar labelled
    "FIRST PREDICTION" split "76% JAX/XLA compilation / 24% other". The callout reads
    "~76% of first prediction → JAX/XLA compilation". Against the data:
    - **The 76% is not a share of the first prediction.** It is TR-04 / TR-01 = 12.55 s
      of `cache_miss` self time over the **16.56 s traced `apply_fn` span**. That span
      comes from a separate trace-capture run (`configs/af_spike_trace_capture.yaml`),
      and its timer and window differ from B-08.
    - **The share of the 27.78 s timed first call cannot be obtained from the repo.**
      The runs differ, the timed call includes profiler finalisation (disc. 11) and no
      TPU logs exist. Placing the 76% under the 27 s cold start is unsupported.
    - **"24% other" is also mislabelled.** `cache_miss` has the same 16.56 s wall time as
      `apply_fn` (`profiling/trace_analysis.md:30-31`), so the whole span is inside
      `cache_miss`. The remaining 4.01 s (24.2%) is its child frames, `_infer_params` →
      `_trace_for_jit` → `trace_to_jaxpr` → Haiku `apply_fn`
      (`profiling/trace_analysis.md:41-51`). That is still JAX tracing, not other work.
      The device track is nearly idle over the span (`profiling/trace_analysis.md:53-58`).
    - **"27 s"** rounds B-08 (27.78 s) down and includes profiler overhead of unknown
      size on TPU (disc. 11).
    - **Supported wording:** "~76% of the traced `apply_fn` call (12.55 of 16.56 s) is
      self time in JAX's `cache_miss` tracing/compilation path; the rest is its child
      tracing frames."
    - **Related:** this is the slide-level form of discrepancy 3, and the same
      misattribution appears on the website and in the README
      (`paper/WEBSITE_UPDATES_NEEDED.md`, items 5.2–5.6).
15. **AF3 ran with 10 recycles and AF2 with 0, so A3-11 and A3-12 cannot be interpreted.**
    - **Settings.**
      - The AF3 Colab runs did not pass `--num_recycles` (`notebooks/af3_*_colab.ipynb`,
        cell 14), so AF3 used its default of **10** (`run_alphafold.py:372-377`, AF3
        commit `29596b970`). That means `num_recycles + 1` = **11 trunk passes**
        (`src/alphafold3/model/model.py:317-319`).
      - AF2 ran with `num_recycle = 0`: **1 Evoformer pass** (HW-05, RD-03).
      - The flag has `lower_bound=1` (`run_alphafold.py:376`), so AF3 cannot be run at
        AF2's recycle 0.
    - **Three confounds in A3-11 / A3-12, pulling in different directions:**
      1. **Recycles:** 11 trunk passes for AF3 against 1 for AF2. This inflates AF3.
      2. **Compilation:** included in AF3's model-inference time (A3-25, A3-28),
         excluded from AF2's steady state (B-03, B-06). This inflates AF3 (disc. 6).
      3. **Diffusion samples:** one AF3 call produces 5 samples, but the trunk and its
         recycles run once for all 5 (`model.py:317-321`). Dividing by 5 gives each
         sample only a fifth of the trunk cost. This deflates AF3. AF2 produces one
         structure per call.

      The net effect is unknown. Neither ratio says whether AF3 is faster or slower than
      AF2 on comparable work.
    - **Correcting only the recycle axis flips the direction.** Here AF2 is scaled to
      AF3's recycle count with the measured AF2 recycle sweep: RD-03/06/09, 0.469 →
      0.933 → 1.845 s, factors 1.99× and 3.93×, roughly proportional to recycles + 1.
      "Per sample" divides AF3 by 5 first (the A3-11 and A3-12 definition); "per call"
      uses A3-25 or A3-28 directly.

      | AF2 scaled to | AF2 factor | CPU per sample | GPU per sample | CPU per call | GPU per call |
      |---|---|---|---|---|---|
      | not scaled (current A3-11 / A3-12) | 1× | 2.26× | 1.32× | 11.32× | 6.59× |
      | r = 3, measured (no extrapolation; a lower bound for r = 10) | 3.93× | **0.58×** | **0.34×** | 2.88× | 1.68× |
      | r = 10, linear in the measured 0.4587 s per recycle | 10.78× | **0.21×** | **0.12×** | 1.05× | **0.61×** |
      | r = 10, time ∝ (r + 1) | 11× | **0.21×** | **0.12×** | 1.03× | **0.60×** |

      All values are AF3 ÷ scaled AF2; below 1 means AF3 is faster.
      - **Per sample (how A3-11 and A3-12 are defined):** both backends drop below 1 in
        every scaled row, already at the measured r = 3 point with no extrapolation.
        AF3 would be *faster* than AF2, the opposite of the current 2.26× / 1.32×.
      - **Per call:** GPU drops below 1 at r = 10, while CPU lands at about parity
        (1.03–1.05×). So even after this correction the result depends on the
        normalisation.
      - **Limits of this check:**
        - The recycle sweep was measured on the TPU, not on Colab CPU/GPU.
        - r = 10 lies beyond the measured 0–3 range.
        - AF3 still includes compilation, and the per-sample trunk sharing is not
          corrected.
        - The table shows that the sign is not robust. It does not estimate the true
          ratio.
    - **Consequence.** A3-11 and A3-12 must not be reported as AF3-vs-AF2 speed
      comparisons. The same applies to their predecessors (2.3× / 1.74×, and 2.31× in
      earlier notes) and to "AF3 gains more from the GPU (27.9×) than AF2 (16.2×)",
      which compares the same mismatched workloads. AF3's own GPU-over-CPU ratio (A3-10)
      is unaffected: both of its sides use the same AF3 settings.
      - A valid comparison would need matched trunk passes: AF2 measured at
        `num_recycle = 10`, or both models at 1. It would also need warm timing that
        excludes compilation (e.g. a second fold job in the same process) and a stated
        normalisation, per call or per structure.
16. **The Colab T4 steady state halved between the 2026-08-08 baseline and the 2026-09-15
    rerun, and nothing in the repo explains why.**
    - **The gap.** B-06 (2026-08-08) is 13.086 s, a single call. The 2026-09-15 rerun gives
      6.5672 ± 0.0574 s over 5 calls
      (`results/repro/2026-09-15_gpu-t4/result_gpu-t4-repro_model_3_len118_recycle0_float32_noprofile.json
      → steady_state_mean_seconds, steady_state_stdev_seconds`). 13.086 / 6.5672 = 1.99×.
      The same session's profiler-on run gives 6.9673 s (n = 1,
      `result_gpu-t4-repro_model_3_len118_recycle0_float32.json → steady_state_mean_seconds`),
      and the August steady-state call ran without the profiler (discrepancy 11), so the
      profiler setting does not account for the gap.
    - **What is recorded as the same.** Both runs report a Tesla T4 with 15,360 MiB,
      driver 580.82.07 and compute capability 7.5. August: `nvidia-smi` for GPU, memory and
      driver, the TensorFlow device log for compute capability
      (`paper/sections/methodology.md:36-38`, from the notebook at `faeaa4b`). September:
      `environment_gpu-t4-repro.json → hardware.nvidia_smi`. Configuration
      (`model_3`, 118 residues, recycle 0, float32) is the same.
    - **What is known to differ, with no measured effect.**
      - Script: August ran the uncommitted copy embedded in the notebook; September ran
        `src/spike_tpu_forward_pass.py` at `86fca03` (`environment_gpu-t4-repro.json →
        repo_commit`). Discrepancy 12 states the timing structure of the two is the same.
      - Software: August installed JAX unpinned and its resolved version is not in the repo
        (`paper/sections/methodology.md:97-98`); September used `jax` / `jax-cuda12-plugin`
        0.10.2 (`environment_gpu-t4-repro.json → package_versions`). `numpy` was 2.5.1 in
        August (`paper/sections/methodology.md:100`) and 2.1.3 in September (same key).
      - Host: the August host CPU and RAM were not recorded
        (`paper/sections/methodology.md:40`); September's is `Intel(R) Xeon(R) CPU @ 2.00GHz`,
        2 logical CPUs (`environment_gpu-t4-repro.json → hardware.cpu_model, logical_cpus`).
    - **Why it is open.** None of these differences has been isolated by a run that changes
      one of them. The repo therefore contains no verifiable cause, and it cannot say which
      of the two values is representative. August is n = 1; September is one session.
    - **Consequence.** B-06 stays the August baseline in the master table. B-10, B-11 and
      B-14 are computed from B-06 and carry this open gap; cite them with it.

    **APERTO (not resolved):** see "GPU session variability (Colab)" below.

## Speedup direction check (summary)

All speedups use time on the slower system divided by time on the faster system.

| Claim | Formula | Value | Correct wording |
|---|---|---|---|
| B-10 | 212.113 / 13.086 | 16.21× | GPU (T4) is 16.2× faster than CPU |
| B-11 | 13.086 / 0.47 | 27.84× | TPU is 27.8× faster than GPU (T4) |
| B-12 | 212.113 / 0.47 | 451.3× | TPU is 451× faster than CPU |
| A3-10 | 2401.38 / 86.21 | 27.86× | AF3: GPU is 27.9× faster than CPU (model inference, compilation included on both sides) |
| MQ-06 | 14.718 / 2.128 | 6.92× | pmap on 8 chips gives 6.92× the throughput of a single protein on 1 chip |
| CC-07 | 37.68 / 5.53 | 6.81× | init_params with a warm cache is 6.81× faster than cold |

The repo's other copies of these ratios (`README.md`, `website/index.html`,
`website/src/data/experiments.js`, `results/comparison.md`,
`results/sweep/af3_comparison.md`) were grepped. All of them use the directions
above; none has the GPU-over-CPU and TPU-over-GPU ratios swapped.

## B3 — CPU session variability (Colab)

Workload as in the master table (`model_3`, 118 residues, `num_recycle=0`, float32).
Rows 3 and 4 are the two B3 reruns, on different days, from
`notebooks/alphafold_cpu_benchmark.ipynb` with `REPO_REF` set to a commit hash. Each
run's files go in `results/repro/<YYYY-MM-DD>_cpu-colab/`. Fill the rows only from
those files: host CPU, cores and repo commit from `environment.json` (for 2026-09-15, from
`environment_cpu-colab-repro.json`, since that run has no `environment.json`), timings from
`result_cpu-colab-repro_model_3_len118_recycle0_float32_noprofile.json`.

| Run | Date | Time (UTC) | Colab tier | Host CPU | Cores | Repo commit | init_params (s) | First predict (s) | Steady state mean ± std (s) | n repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (original) | 2026-08-08 ¹ | | | not recorded ² | 2 vCPU (prose only, not a recorded field) ² | not recorded ³ | 41.99 (B-01) | 271.98 (B-02) ⁴ | 212.113 (B-03), no std | 1 |
| 2 (2026-09-15) | 2026-09-15 ⁵ | 18:04–18:40 ⁶ | ⁷ | Intel(R) Xeon(R) CPU @ 2.20GHz (family 6, model 79) ⁵ | 2 logical (1 core × 2 threads) ⁵ | `86fca03` ⁵ | 59.02 ⁸ | 385.48 ⁸ | 348.861 ± 4.6903 ⁸ | 5 ⁸ |
| 3 (B3 rerun 1) | 2026-09-16 ⁹ | 12:37–13:14 ¹⁰ | ¹¹ | Intel(R) Xeon(R) CPU @ 2.20GHz (family 6, model 79) ⁹ | 2 logical (1 core × 2 threads) ⁹ | `9e1a07e` ⁹ | 59.87 ¹² | 373.24 ¹² | 351.8907 ± 4.3025 ¹² | 5 ¹² |
| 4 (B3 rerun 2) | 2026-09-17 ¹³ | 08:52–09:30 ¹⁴ | ¹⁵ | Intel(R) Xeon(R) CPU @ 2.20GHz (family 6, model 79) ¹³ | 2 logical (1 core × 2 threads) ¹³ | `9e1a07e` ¹³ | 60.3 ¹⁶ | 394.69 ¹⁶ | 359.2726 ± 2.7874 ¹⁶ | 5 ¹⁶ |

The 2026-09-15 and 2026-09-16 runs used different commits (`86fca03`, `9e1a07e`; `src/` is identical at both, only the notebook differs) yet agree within 1% (351.8907 / 348.861 = 1.009); the three September sessions span 348.861–359.2726 s (359.2726 / 348.861 = 1.030, i.e. 3%), all 1.64–1.69× the August value (348.861 / 212.113 = 1.645; 359.2726 / 212.113 = 1.694).

1. The date of commit `2755250`, which added the results (discrepancy 12). The run date and time are not recorded.
2. HW-01, status R (`results/comparison.md:9,17-18`). The result JSON records only `host_processor: x86_64`.
3. The run used a copy of the script embedded in the notebook, not a committed script (discrepancy 12).
4. Includes `jax.profiler.trace` finalisation inside the timer, 36.00 s by the log (discrepancy 11). Not comparable with profiler-off first predicts.
5. `results/repro/2026-09-15_cpu-colab/environment_cpu-colab-repro.json`: `recorded_at_utc` (2026-09-15T18:54:38+00:00, written at step 7), `hardware.cpu_model`, `hardware.logical_cpus` (2), `hardware.lscpu` (`CPU family: 6`, `Model: 79`, `Socket(s): 1`, `Core(s) per socket: 1`, `Thread(s) per core: 2`), `repo_commit` (`86fca03bb0c9dde7f93b2c1e54a407d6dad6054e`). This run predates the notebook's `environment.json` / `pip_freeze.txt` cell, so those two files do not exist here.
6. First and last log lines of `log_cpu-colab-repro_noprofile.txt` (`I0915 18:04:00.904643` → `I0915 18:40:30.876429`), i.e. the timed profiler-off run. The log timestamps carry no timezone. They are read as UTC because the last line of `log_cpu-colab-repro_profile.txt` (18:54:33) falls 5 s before `recorded_at_utc` (18:54:38 UTC); this is an inference, not a recorded value.
7. Not recorded. Searched `environment_cpu-colab-repro.json`, both logs, `pip_freeze_cpu-colab-repro.txt`, both result JSONs and `alphafold_cpu_benchmark.executed.ipynb` (including its `metadata`) for "tier", "Colab Pro", "compute unit", "High-RAM", "machine_shape". The only match is the notebook's own title, "(free Colab)", which describes the intended runtime, not the one used.
8. `results/repro/2026-09-15_cpu-colab/result_cpu-colab-repro_model_3_len118_recycle0_float32_noprofile.json` → `init_params_seconds`, `first_predict_compile_and_run_seconds`, `steady_state_mean_seconds`, `steady_state_stdev_seconds`, `num_steady_state_runs`; `profile_first_predict` is `false`. The profiler-on run in the same folder (`result_cpu-colab-repro_model_3_len118_recycle0_float32.json`) is not used.
9. `results/repro/2026-09-16_cpu-colab/environment.json`, written by the notebook's step 4b right after the clone: `recorded_at_utc` (2026-09-16T12:37:30+00:00), `cpu_model`, `lscpu` (`CPU family: 6`, `Model: 79`), `logical_cpus` (2), `sockets` (1), `cores_per_socket` (1), `threads_per_core` (2), `git_rev_parse_head` (`9e1a07e453df4d77af15ee695c2cadc1cca73f07`, equal to `repo_ref`, as the step 4b assert requires).
10. First and last log lines of `log_cpu-colab-repro_noprofile.txt` (`I0916 12:37:45.799718` → `I0916 13:14:19.248941`), i.e. the timed profiler-off run. The log timestamps carry no timezone. They are read as UTC because the first line falls 15 s after `environment.json → recorded_at_utc` (12:37:30 UTC), and the last line of `log_cpu-colab-repro_profile.txt` (13:28:22) falls 3 s before `environment_cpu-colab-repro.json → recorded_at_utc` (13:28:25 UTC); this is an inference, not a recorded value.
11. Not recorded. Searched `environment.json`, `environment_cpu-colab-repro.json`, both logs, `pip_freeze.txt`, `pip_freeze_cpu-colab-repro.txt`, both result JSONs and `2026-09-16_alphafold_cpu_benchmark.ipynb` (including its `metadata`) for "tier", "Colab Pro", "compute unit", "High-RAM", "machine_shape". The only match is the notebook's own title, "(free Colab)", which describes the intended runtime, not the one used.
12. `results/repro/2026-09-16_cpu-colab/result_cpu-colab-repro_model_3_len118_recycle0_float32_noprofile.json` → `init_params_seconds`, `first_predict_compile_and_run_seconds`, `steady_state_mean_seconds`, `steady_state_stdev_seconds`, `num_steady_state_runs`; `profile_first_predict` is `false`. The profiler-on run in the same folder (`result_cpu-colab-repro_model_3_len118_recycle0_float32.json`) is not used.
13. `results/repro/2026-09-17_cpu-colab/environment.json`, written by the notebook's step 4b right after the clone: `recorded_at_utc` (2026-09-17T08:52:27+00:00), `cpu_model`, `lscpu` (`CPU family: 6`, `Model: 79`), `logical_cpus` (2), `sockets` (1), `cores_per_socket` (1), `threads_per_core` (2), `git_rev_parse_head` (`9e1a07e453df4d77af15ee695c2cadc1cca73f07`, equal to `repo_ref`, as the step 4b assert requires).
14. First and last log lines of `log_cpu-colab-repro_noprofile.txt` (`I0917 08:52:44.195650` → `I0917 09:30:16.468240`), i.e. the timed profiler-off run. The log timestamps carry no timezone. They are read as UTC because the first line falls 17 s after `environment.json → recorded_at_utc` (08:52:27 UTC), and the last line of `log_cpu-colab-repro_profile.txt` (09:44:29) falls 3 s before `environment_cpu-colab-repro.json → recorded_at_utc` (09:44:32 UTC); this is an inference, not a recorded value.
15. Not recorded. Searched `environment.json`, `environment_cpu-colab-repro.json`, both logs, `pip_freeze.txt`, `pip_freeze_cpu-colab-repro.txt`, both result JSONs and `2026-09-17_alphafold_cpu_benchmark.ipynb` (including its `metadata`) for "tier", "Colab Pro", "compute unit", "High-RAM", "machine_shape". The only match is the notebook's own title, "(free Colab)", which describes the intended runtime, not the one used.
16. `results/repro/2026-09-17_cpu-colab/result_cpu-colab-repro_model_3_len118_recycle0_float32_noprofile.json` → `init_params_seconds`, `first_predict_compile_and_run_seconds`, `steady_state_mean_seconds`, `steady_state_stdev_seconds`, `num_steady_state_runs`; `profile_first_predict` is `false`. The profiler-on run in the same folder (`result_cpu-colab-repro_model_3_len118_recycle0_float32.json`) is not used.

## GPU session variability (Colab)

Workload as in the master table (`model_3`, 118 residues, `num_recycle=0`, float32), on
the Colab T4 runtime. Row 1 is the August baseline from the master table. Row 2 is the
2026-09-15 rerun from `notebooks/alphafold_gpu_benchmark.ipynb`; its files are in
`results/repro/2026-09-15_gpu-t4/`. Fill further rows only from each run's files: GPU,
host and commit from the environment file, timings from
`result_gpu-t4-repro_model_3_len118_recycle0_float32_noprofile.json`.

| Run | Date | Time (UTC) | Colab tier | GPU | Host CPU / cores | Commit | init_params (s) | First predict (s) | Steady state mean ± std (s) | n repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (original) | 2026-08-08 ¹ | | | NVIDIA Tesla T4, 15 GB VRAM ² | ³ | not recorded ⁴ | 109.16 (B-04) | 97.62 (B-05) ⁵ | 13.086 (B-06), no std | 1 |
| 2 (2026-09-15) | 2026-09-15 ⁶ | 19:01–19:04 ⁷ | ⁸ | Tesla T4, 15360 MiB, driver 580.82.07, compute capability 7.5 ⁶ | Intel(R) Xeon(R) CPU @ 2.00GHz, 2 logical (1 core × 2 threads) ⁶ | `86fca03` ⁶ | 102.6 ⁹ | 39.59 ⁹ | 6.5672 ± 0.0574 ⁹ | 5 ⁹ |

1. The date of commit `2755250`, which added the results (discrepancy 12). No time with a timezone is recorded in this file; `paper/sections/methodology.md:41` gives a log-clock window without timezone.
2. HW-02, status R (`results/comparison.md:10,17-18`). The result JSON records `devices: ["cuda:0"]`.
3. Not recorded (`paper/sections/methodology.md:40`).
4. The run used a copy of the script embedded in the notebook, not a committed script (discrepancy 12).
5. Includes `jax.profiler.trace` finalisation inside the timer, 42.02 s by the log (discrepancy 11). Not comparable with profiler-off first predicts.
6. `results/repro/2026-09-15_gpu-t4/environment_gpu-t4-repro.json`: `recorded_at_utc` (2026-09-15T19:07:25+00:00, written at step 7), `hardware.nvidia_smi`, `hardware.cpu_model`, `hardware.logical_cpus` (2), `hardware.lscpu` (`Socket(s): 1`, `Core(s) per socket: 1`, `Thread(s) per core: 2`), `repo_commit` (`86fca03bb0c9dde7f93b2c1e54a407d6dad6054e`). This run predates the notebook's `environment.json` / `pip_freeze.txt` cell, so those two files do not exist here.
7. First and last log lines of `log_gpu-t4-repro_noprofile.txt` (`I0915 19:01:18.881451` → `I0915 19:04:14.812504`), i.e. the timed profiler-off run. The log timestamps carry no timezone. They are read as UTC because the last line of `log_gpu-t4-repro_profile.txt` (19:07:22) falls 3 s before `recorded_at_utc` (19:07:25 UTC); this is an inference, not a recorded value.
8. Not recorded. Searched `environment_gpu-t4-repro.json`, both logs, `pip_freeze_gpu-t4-repro.txt`, both result JSONs and `alphafold_gpu_benchmark.executed.ipynb` (including its `metadata`) for "tier", "Colab Pro", "compute unit", "High-RAM", "machine_shape". The notebook metadata records only `accelerator: GPU` and `gpuType: T4`; the title's "(free Colab)" describes the intended runtime, not the one used.
9. `results/repro/2026-09-15_gpu-t4/result_gpu-t4-repro_model_3_len118_recycle0_float32_noprofile.json` → `init_params_seconds`, `first_predict_compile_and_run_seconds`, `steady_state_mean_seconds`, `steady_state_stdev_seconds`, `num_steady_state_runs`; `profile_first_predict` is `false`. The profiler-on run in the same folder (`result_gpu-t4-repro_model_3_len118_recycle0_float32.json`) is not used.
10. **Steady-state gap:** row 1 is 13.086 s and row 2 is 6.5672 ± 0.0574 s, a factor of 13.086 / 6.5672 = 1.99×, i.e. almost 2×.
11. **The gap is not explained.** The repo contains no verifiable cause: both runs report the same GPU, memory, driver and compute capability (discrepancy 16 lists the sources), and no run isolates any of the known differences (script copy vs `86fca03`, unrecorded August JAX version, unrecorded August host). Discrepancy 16 tracks it as open.
