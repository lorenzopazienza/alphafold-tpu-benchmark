# Methodology

This section documents how every number in `paper/data/canonical_results.md`
was produced. Every statement below is backed by a file in this repository; the
source is cited inline. Anything the repository does not record is marked
**[NOT IN REPO]** instead of being filled in. Section 8 lists all of those gaps.

Evidence rules used here:
- **Code or config:** what a script or Job file says it does (`file:line`).
- **Log:** what a run actually printed. The only raw run logs in the repo are
  the Colab notebook outputs and `results/sweep/af3_tpu_attempt.log`.
- **Prose:** claims that appear only in a `.md` write-up. These are flagged
  whenever they are the only source.

## 1. Hardware

### 1.1 CPU baseline (Google Colab)

| Item | Value | Source |
|---|---|---|
| Platform | Google Colab, CPU runtime (no accelerator) | `notebooks/alphafold_cpu_benchmark.ipynb` cell 0; the `nvidia-smi` check in cell 2 finds no GPU (`nvidia-smi: command not found`, printed by the AF3 CPU notebook's version of the same check) |
| JAX devices | `[CpuDevice(id=0)]`, backend `cpu` | Log, `alphafold_cpu_benchmark.ipynb` cell 10 |
| Architecture | `x86_64` | `results/result_cpu-colab.json → host_processor, host_machine` |
| CPU model, vCPU count, RAM | **[NOT IN REPO]** as a measurement | "Intel Xeon, 2 vCPU, 13.6 GB RAM" appears only in prose (`results/comparison.md:9,17-18`). No notebook runs `lscpu`, `nproc` or `free`. |
| Run date | 2026-08-08, ~05:19–05:28 (log clock) | Log timestamps, cell 10 |

### 1.2 GPU baseline (Google Colab)

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA Tesla T4, 15,360 MiB | `nvidia-smi` output, `alphafold_gpu_benchmark.ipynb` cell 2 |
| Compute capability | 7.5 | TensorFlow device log, cell 10 (13,749 MB made available to TF) |
| NVIDIA driver / CUDA version reported by the driver | 580.82.07 / 13.0 | `nvidia-smi` output, cell 2 |
| JAX devices | `[CudaDevice(id=0)]`, backend `gpu` | Log, cell 10 |
| Host CPU model, vCPU count, host RAM | **[NOT IN REPO]** | Not printed by any notebook |
| Run date | 2026-08-08, ~04:49–04:54 (log clock) | `nvidia-smi` timestamp and log, cells 2 and 10 |

### 1.3 TPU (Stanford GKE)

| Item | Value | Source |
|---|---|---|
| Cluster / region / project | `class-tpu-cluster-west4` / `us-west4` / `soe-hpccenter` | `scripts/run_spike.sh:6`, `README.md:173-174` |
| Accelerator | `tpu-v5-lite-podslice`, topology `2x4` | `nodeSelector` in every Job, e.g. `configs/af_spike_job.yaml:13-14` |
| Chips requested | 8 (`google.com/tpu: "8"`, request and limit) | e.g. `configs/af_spike_job.yaml:40,43` |
| Container memory | 32 Gi request, 64 Gi limit | e.g. `configs/af_spike_job.yaml:41,44` |
| Scheduling | Kubernetes Job, admitted via Kueue queue `student-queue`, `backoffLimit: 0` | Label in every Job file; `README.md:176` |
| JAX devices | 8 devices, `TPU_0` … `TPU_7`, with 2×4 coordinates | `results/result_tpu-v5e-podslice.json → devices` |
| HBM limit per chip | 16,909,000,000 bytes | `results/sweep/chip_visibility_sweep.json → bytes_limit` |
| Host CPU model, host vCPU, host RAM, node machine type | **[NOT IN REPO]** | |
| libtpu / TPU runtime / firmware version | **[NOT IN REPO]** | |
| Whether all Jobs landed on the same physical node | **[NOT IN REPO]** | Node names were not recorded |
| Run dates | **[NOT IN REPO]** | Git only gives upper bounds (see Section 7) |

Only one chip does work in the default single-query path: `TPU_1`–`TPU_7`
report 0 bytes in use (`chip_visibility_sweep.json`). Single-query TPU timings
therefore measure one v5e chip, even though 8 are allocated.

### 1.4 AlphaFold3 runs

| Run | Hardware | Source |
|---|---|---|
| AF3, Colab CPU | Colab CPU runtime, `CpuDevice(id=0)` | `notebooks/af3_cpu_colab.ipynb` cells 2 and 14 |
| AF3, Colab GPU | Tesla T4, 15,360 MiB, driver 580.82.07; run on 2026-08-12 | `notebooks/af3_gpu_colab.ipynb` cells 2 and 14 |
| AF3, Stanford CPU | **[NOT IN REPO]**. No Job file, log or hardware description. Prose says it ran "inside a Kubernetes Job on `hpcc-cluster-41`'s CPU allocation" (`results/sweep/af3_comparison.md:72-75`). | |
| AF3, TPU attempt | Same TPU `nodeSelector` as above | `configs/af_spike_af3_tpu.yaml:22-23` |

The two Colab AF2 runs (2026-08-08) and the two Colab AF3 runs (2026-08-12)
used the same Colab runtime *types*. Nothing in the repo shows they ran on the
same physical machines.

## 2. Software versions

The repository has **no `requirements.txt`, lockfile or container digest**.
Versions were recovered from Job files, notebook cells and notebook outputs.

### 2.1 AF2 on TPU (all Jobs in `configs/`, except the AF3 attempt)

| Component | Version | Source |
|---|---|---|
| Base image | `python:3.12-slim` (tag only, no digest) | e.g. `configs/af_spike_job.yaml:17` |
| Python patch version | **[NOT IN REPO]** | The tag floats |
| `jax[tpu]` | `==0.10.2` (pinned), from `libtpu_releases.html` | e.g. `configs/af_spike_job.yaml:30` |
| `dm-haiku`, `ml_collections`, `absl-py`, `biopython`, `numpy` | **unpinned**. Resolved versions **[NOT IN REPO]** | e.g. `configs/af_spike_job.yaml:31` |
| `tensorflow-cpu` | `>=2.18`. Resolved version **[NOT IN REPO]** | e.g. `configs/af_spike_job.yaml:31` |
| `uv` | unpinned (`pip install uv`). Version **[NOT IN REPO]** | e.g. `configs/af_spike_job.yaml` install step |
| AlphaFold2 source | `git clone --depth 1` of the default branch of `google-deepmind/alphafold`. Commit **[NOT IN REPO]** | e.g. `configs/af_spike_job.yaml:34` |

### 2.2 AF2 on Colab CPU and GPU

| Component | CPU notebook | GPU notebook | Source |
|---|---|---|---|
| JAX | `pip install -q -U jax` (unpinned) | `pip install -q -U "jax[cuda12]"` (unpinned) | cell 4 of each |
| Resolved JAX / jaxlib version | **[NOT IN REPO]**: `-q` suppresses it | **[NOT IN REPO]** | |
| `dm-haiku`, `ml_collections`, `absl-py`, `biopython` | unpinned, versions **[NOT IN REPO]** | same | cell 4 |
| `numpy` | 2.5.1 (from a pip dependency-conflict message) | 2.5.1 (same message) | cell 4 output |
| TensorFlow | Colab's preinstalled build, version **[NOT IN REPO]** | same; `tensorflow-cpu` deliberately not installed | cell 4 |
| CUDA / cuDNN libraries used by JAX | n/a | **[NOT IN REPO]** | |
| Python version | **[NOT IN REPO]** | **[NOT IN REPO]** | Four days later the AF3 notebooks report CPython 3.12.13; this is indirect evidence only |
| AlphaFold2 source | `git clone --depth 1`, commit **[NOT IN REPO]** | same | cell 6 |

### 2.3 AF3 on Colab CPU and GPU

`uv sync` printed the full resolved environment: 70 packages, identical in the
CPU and GPU notebooks (`notebooks/af3_*_colab.ipynb`, cell 8 output). The
relevant versions:

| Component | Version |
|---|---|
| `alphafold3` | `3.0.3.dev1+g29596b970` (source commit `29596b970`) |
| `jax`, `jaxlib`, `jax-cuda12-pjrt`, `jax-cuda12-plugin` | 0.10.2 |
| `dm-haiku` | 0.0.16 |
| `numpy` | 2.4.1 |
| `tokamax` | 0.0.12 |
| `chex` / `flax` / `optax` | 0.1.91 / 0.12.2 / 0.2.6 |
| `nvidia-cudnn-cu12` / `nvidia-cublas-cu12` / `nvidia-cuda-runtime-cu12` / `nvidia-nccl-cu12` | 9.17.1.4 / 12.9.1.4 / 12.9.79 / 2.29.2 |
| `rdkit` | 2025.9.4 |
| Python | CPython 3.12.13 (`uv sync` output) |
| `uv` | 0.12.3 (installer output, cell 6) |
| Weights | `af3.bin.zst`, 1,146,811,260 bytes (cell 10 output) |

The CUDA wheels are installed in the CPU notebook as well; the lockfile does
not distinguish backends. In both runs AF3 logs a `tokamax` gated-linear-unit
kernel failing ("Not supported on cpu" / "Not supported on Tesla T4") and
falling back to another implementation (cell 14 output).

### 2.4 AF3 on TPU (attempt)

| Component | Version | Source |
|---|---|---|
| AF3 source | `main` branch tarball, commit **[NOT IN REPO]** | `configs/af_spike_af3_tpu.yaml:47` |
| `jax[tpu]` | `==0.10.2` | `configs/af_spike_af3_tpu.yaml:57` |
| Everything else | resolved by `uv sync`, versions **[NOT IN REPO]** | |

### 2.5 The Dockerfile was not used for any reported result

`Dockerfile` defines a `python:3.12-slim` image with `jax[...]` unpinned,
`dm-haiku==0.0.12` and `tensorflow-cpu==2.16.1` (`Dockerfile:12,28-35,41-47`).
Neither the TPU Jobs nor the Colab notebooks use it:
- the Jobs run the stock `python:3.12-slim` image and install packages at
  container start, with different pins (`jax[tpu]==0.10.2`, `tensorflow-cpu>=2.18`,
  unpinned `dm-haiku`);
- the notebooks `pip install` directly.

`Dockerfile:49` says AlphaFold is "pinned to a fixed commit for
reproducibility", but line 50 is `git clone --depth 1` with no commit. The
Dockerfile therefore documents neither the environment that produced the
results nor a pinned one.

### 2.6 Compatibility patch applied to AlphaFold2

Every AF2 script monkey-patches `jax.numpy.clip` so that the `a_min=`/`a_max=`
keyword arguments used by AlphaFold2's source map to the current `min=`/`max=`
(`src/spike_tpu_forward_pass.py:46-54`). AlphaFold2's source itself is not modified.

## 3. Workload and inputs

### 3.1 AlphaFold2 model configuration

- **Model config:** `model_3` by default; `model_4` and `model_5` only in the
  model comparison (`configs/af_spike_combined.yaml:35-38`). These three configs
  have `use_templates=False`; the script refuses `model_1`/`model_2`
  (`src/spike_tpu_forward_pass.py:141-144`).
- **Ensembling:** `num_ensemble = 1` in every script (e.g.
  `src/spike_tpu_forward_pass.py:147`). The ensemble experiment builds its 8
  members outside AlphaFold instead (Section 3.4).
- **Recycling:** `num_recycle = 0` except in the recycle sweep (1 and 3)
  (`configs/af_spike_job_sweep.yaml:52-53`).
- **Precision:** float32 by default. The bfloat16 run casts every float32
  parameter leaf after `init_params` (`src/spike_tpu_forward_pass.py:166-171`).
  Input features are not cast.

### 3.2 Sequences

- **118 residues** in the single-query scripts: a fixed toy sequence
  (`TOY_SEQUENCE_118`, `src/spike_tpu_forward_pass.py:91-94`), also used by the
  ensemble and GSPMD scripts and by the AF3 input (`src/make_af3_input.py:30`).
  It is synthetic, not a real protein.
- **Any other length:** the 20-letter alphabet `ACDEFGHIKLMNPQRSTVWY` repeated
  and truncated (`src/spike_tpu_forward_pass.py:97-101`).
  - The sequence-length sweep (60, 120, 250, 500) therefore uses only these
    synthetic sequences; none of its points is the 118-residue toy sequence.
- **`vmap` batching, `pmap` multi-query and the scaling grid:** every item uses
  the repeated alphabet rotated by its batch index
  (`src/spike_batch_forward_pass.py:53-58`, `src/spike_pmap_forward_pass.py:50-52`),
  **including at 118 residues**. These experiments do not fold the toy sequence.

### 3.3 MSA and features

- **MSA:** a single-sequence MSA containing only the query, built with
  AlphaFold's own `parsers.Msa` and `pipeline.make_msa_features` (e.g.
  `src/spike_tpu_forward_pass.py:111-118`). There is no genetic database search
  and there are no templates.
- **Feature processing:** `af_features.np_example_to_features(..., random_seed=0)`
  (`src/spike_tpu_forward_pass.py:153-155`). This step runs before any timer
  starts, so it is not part of any reported time.
- **Padded shapes:** the logged model inputs are `msa_feat (1, 512, 118, 49)` and
  `extra_msa (1, 5120, 118)` (log, `alphafold_*_benchmark.ipynb` cell 10). The
  MSA tensors are padded to 512 cluster rows and 5,120 extra rows, so the trivial
  MSA does not shrink the MSA dimensions of the compiled graph.

### 3.4 Seeds and experiment-specific settings

| Setting | Value | Source |
|---|---|---|
| Feature seed | 0 (ensemble: member index `i`, 0–7) | `src/spike_tpu_forward_pass.py:154`; `src/spike_ensemble_shard_forward_pass.py:103` |
| Parameter-init seed | 0 | `src/spike_tpu_forward_pass.py:162` |
| `predict` seed / `apply` PRNG key | 0 / `PRNGKey(0)` | `src/spike_tpu_forward_pass.py:176,183`; `src/spike_pmap_forward_pass.py:100` |
| Compilation cache | `jax_compilation_cache_dir=/tmp/jax_cache`, min entry size −1, min compile time 0; two separate processes | `src/spike_tpu_forward_pass.py:122-126`; `configs/af_spike_combined.yaml:57-64` |
| GSPMD mesh | `jax.make_mesh((8, 1), ("fsdp", "tp"))`, both axes `AxisType.Auto`, under `jax.set_mesh` | `src/spike_meshshard_forward_pass.py:85-86` |
| `pmap` multi-query | `jax.pmap(runner.apply, in_axes=(None, None, 0))` | `src/spike_pmap_forward_pass.py:101` |
| `pmap` + `pmean` ensemble | `jax.pmap` over members, `jax.lax.pmean` of `predicted_lddt` logits on axis `ensemble` | `src/spike_ensemble_shard_forward_pass.py:122,125` |
| `vmap` batching | `jax.vmap(runner.apply, in_axes=(None, None, 0))` on one device | `src/spike_batch_forward_pass.py:117` |
| Chip visibility | `TPU_VISIBLE_CHIPS` set to 1, 2, 4 or 8 chips; the 2- and 4-chip runs failed, and so did a retry with `LIBTPU_INIT_ARGS` | `configs/af_spike_chipcount.yaml:39-43`; `configs/af_spike_chips_retry.yaml` |

### 3.5 AlphaFold3 input

- **Input JSON:** one protein chain `A` with the same 118-residue toy sequence,
  `unpairedMsa` and `pairedMsa` empty, no templates, `modelSeeds: [1]`
  (`src/make_af3_input.py:36-59`; the same payload is written inline in each
  Colab notebook, cell 12).
- **Command:** `run_alphafold.py --norun_data_pipeline
  --flash_attention_implementation=xla --jax_backend={cpu,gpu}` (cell 14).
- **GPU flag:** the GPU run also sets
  `XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter`.
- **Samples:** the number of diffusion samples is not set on the command line.
  The run produced 5 (log: "Extracting 5 inference samples"; 5 rows in each
  `ranking_scores.csv`).
- **Weights:** AF3 uses the real, trained AF3 weights, unlike AF2 in this study
  (Section 4).

## 4. Why AlphaFold2 uses random weights

AF2 is run with `RunModel(cfg, params=None)`, so Haiku initializes the
parameters randomly (`src/spike_tpu_forward_pass.py:157-158`; log line
`model.py:120] Initialized parameters randomly`). The repository gives this
rationale (`README.md:180-183`, and the docstring of the script embedded in
`alphafold_*_benchmark.ipynb` cell 8):

1. The study asks a systems question: where time goes during compilation and
   execution on each backend. It does not ask about prediction quality.
2. Trained weights (about 350 MB per model) are not needed to answer it.
   Random init exercises "the exact same JIT-compiled graph shapes and
   accelerator ops as a real forward pass".
3. For the same reason the genetic MSA search is replaced by a single-sequence MSA.

Consequences and limits:
- **Outputs are meaningless.** AF2 structures and pLDDT from these runs must not
  be interpreted biologically.
- **Timing equivalence is asserted, not tested.** No run with trained AF2
  weights exists in the repo **[NOT IN REPO]**. Graph shapes do not depend on
  weight values, but value-dependent effects on runtime (for example numerical
  special cases on CPU) were not checked.
- **AF3 is not random-init.** AF2-vs-AF3 comparisons differ in weights as well as
  in model.
- **The "real protein" exhibit is not AF2.** It uses ESMFold with trained weights
  on human ubiquitin, run locally on Apple Silicon (`mps`)
  (`notebooks/real_protein_fold_visualization.ipynb`, cells 0, 4, 8). It is not
  part of the benchmark.

## 5. Timing methodology

### 5.1 Single-query AF2 scripts (`spike_tpu_forward_pass.py`, `spike_meshshard_forward_pass.py`)

All timers use host wall-clock `time.time()`. Each process measures three
consecutive regions:

| Region | What is timed | Synchronization | Source |
|---|---|---|---|
| `init_params` | `runner.init_params(...)`: Haiku `init`, including its own JIT trace, compilation and execution | none explicit | `src/spike_tpu_forward_pass.py:160-164` |
| 1st `predict` (compile + run) | `runner.predict(...)`, **inside `jax.profiler.trace(...)`** in `spike_tpu_forward_pass.py`, without a profiler in the GSPMD script | `jax.block_until_ready(result)` | `src/spike_tpu_forward_pass.py:173-179`; `src/spike_meshshard_forward_pass.py:108-113` |
| 2nd `predict` (steady state) | the identical call again, no profiler | `jax.block_until_ready(result2)` | `src/spike_tpu_forward_pass.py:181-186` |

- **Warm-up:** exactly one call. The first `predict` serves as warm-up and also
  carries compilation. The steady-state figure is **one** subsequent call.
- **No averaging:** there are no additional warm-up iterations, no repeated
  steady-state calls within a process, no averaging and no outlier rejection.
- **Rounding in JSON:** 2 decimals for `init_params` and first predict, 3 for
  steady state (`src/spike_tpu_forward_pass.py:219-221`).
- **`predict` includes host-side work.** AlphaFold2's `RunModel.predict` also
  computes confidence metrics after `apply`. This comes from upstream
  `alphafold/model/model.py`, which is not vendored here, so its commit is
  **[NOT IN REPO]**. The logs are consistent with it: `model.py:170` logs at entry
  and `model.py:183` at exit, before the timer stops.

**Version caveat for the CPU/GPU baselines.** The script in `src/` has a single
commit (`2755250`, which also added all results), so git cannot show which
version produced which result.
- **CPU/GPU:** the Colab notebooks embed an older, uncommitted version (cell 8,
  byte-identical in both notebooks). Their logs cite line numbers and messages
  that exist only in that version, e.g. `spike_tpu_forward_pass.py:146` and
  "Run tag: …". It hard-codes `model_3`, recycle 0 and the toy sequence, but its
  timing structure is the same as above: profiler around the first call,
  `block_until_ready` on both calls, `time.time()`.
- **TPU:** the baseline JSON (`results/result_tpu-v5e-podslice.json`) has a field
  set that matches **neither** surviving version. It lacks `host_processor`, which
  both versions write. The script version that produced it is **[NOT IN REPO]**.

### 5.2 Confound: the profiler adds time to the first `predict`

The Colab logs make it possible to split the first-call time. The measured
intervals are:

| Backend | Timed first predict | Inside `predict()` (entry log → exit log) | After `predict()` returns, still inside the timer | Same interval, 2nd call (no profiler) |
|---|---|---|---|---|
| CPU | 271.98 s | 235.98 s | **36.00 s** (13% of the timed value) | 0.0004 s |
| GPU (T4) | 97.62 s | 55.57 s | **42.02 s** (43% of the timed value) | 0.0003 s |

The only code between `predict()` returning and the timer stopping is
`jax.block_until_ready` followed by exiting the `jax.profiler.trace` context. The
same `block_until_ready` takes under 1 ms in the second call, so the extra
36–42 s is attributable to finalizing the profiler trace. (Timestamps are from
`alphafold_cpu_benchmark.ipynb` and `alphafold_gpu_benchmark.ipynb`, cell 10.)

Implications:
- **Scope.** First-call times ("compile + run") from `spike_tpu_forward_pass.py`
  and `spike_batch_forward_pass.py` include profiler overhead. On CPU/GPU they
  are not pure compile + execute times. Steady-state times are unaffected.
- **TPU magnitude unknown.** No TPU run log is in the repo **[NOT IN REPO]**.
- **Non-comparable first calls across scripts.** The `pmap`, ensemble and GSPMD
  scripts time their first call without a profiler. Their first-call numbers
  (e.g. GSPMD 14.76 / 14.46 s, ensemble 16.61 s) are not comparable with the
  ~27–29 s first predicts from `spike_tpu_forward_pass.py`.
- **Alternative explanation for GSPMD.** `results/sweep/sharding.md` attributes
  the shorter GSPMD first call to a warm XLA cache "carried over from earlier in
  the same pod's Python process lifetime". But `configs/af_spike_sharding.yaml:36-44`
  launches the `pmap` and GSPMD scripts as separate `python3` processes, without
  `--cache_dir`. The missing profiler is a simpler explanation consistent with the
  CPU/GPU logs, but it is **not verified on TPU**.
- **Trace window.** The same effect may explain why the profiler's traced
  `apply_fn` span (16.56 s, `profiling/trace_analysis.md`) is shorter than the
  measured TPU first predict (27.78 s). This is also not verified.

### 5.3 Multi-chip and batching scripts

| Script | 1st call timed | 2nd call timed | Profiler on 1st call | Derived metrics | Source |
|---|---|---|---|---|---|
| `spike_batch_forward_pass.py` (`vmap`) | yes | yes, `block_until_ready` | **yes** | throughput = B / t₂; per protein = t₂ / B | `src/spike_batch_forward_pass.py:119-136` |
| `spike_pmap_forward_pass.py` | yes | yes, `block_until_ready` | no | same definitions | `src/spike_pmap_forward_pass.py:103-118` |
| `spike_ensemble_shard_forward_pass.py` | yes | yes, `block_until_ready` | no | none (one query) | `src/spike_ensemble_shard_forward_pass.py:128-140` |

Neither script times `init_params` or the `vmap` batching setup. Steady state is
again a single call, rounded to 4 decimals. In both `pmap` scripts the
parameters are passed with `in_axes=None` on every call. Whether their
replication to the 8 devices is inside the timed region was not measured
separately **[NOT IN REPO]**.

### 5.4 Memory

Per-device `bytes_in_use` and `peak_bytes_in_use` come from JAX's
`device.memory_stats()`, read once after the second call (e.g.
`src/spike_tpu_forward_pass.py:192-204`).
- **Peak:** the process-lifetime peak, not the peak of the steady-state call.
- **`tpu-info`:** the sweep Jobs also ran `tpu-info` every 3 s
  (`configs/af_spike_job_sweep.yaml:34-44`). It reported all metrics as N/A; the
  write-up gives the missing runtime metrics ports as the cause
  (`results/sweep/README.md:76-92`). No duty-cycle data exists.

### 5.5 AlphaFold3

Two timers are recorded in `results/result_af3_*.json`:

| Timer | JSON field | How it is measured | What it includes |
|---|---|---|---|
| **Process wall-clock** | `wall_clock_total_seconds` | `time.time()` around `subprocess.run(["uv", "run", "python3", "run_alphafold.py", ...])` (`af3_*_colab.ipynb` cell 14) | `uv` start-up, model construction, parameter loading, featurisation, JIT compilation, inference for all samples, sample extraction, output writing |
| **Model inference** | `model_inference_seconds` | AF3's own log line "Running model inference with seed 1 took … seconds", printed in cell 14 and parsed by cell 16 | JIT compilation and inference for all samples. Compilation cannot be separated: it is the only model call in the process and there is no warm-up |

- **Per-sample figure:** `seconds_per_sample` = model inference ÷ number of
  samples, where the sample count (5) is also parsed from AF3's log (cell 16).
- **Old per-sample figure:** wall-clock ÷ 5 is kept as `wall_clock_seconds_per_sample`.
  Until 2026-09-15 that wall-clock figure *was* the published `seconds_per_sample`,
  next to `total_inference_seconds`, the old name of `wall_clock_total_seconds`.
- **Notebook state:**
  - **Code:** cell 16 was rewritten on 2026-09-15 to produce the current schema and
    was not re-executed.
  - **Saved outputs:** those of cells 16 and 18 still come from the original runs
    (2026-08-12). They show the old schema, and their confidence fields are `null`
    because the old code searched for `summary_confidences.json` instead of AF3's
    `<job>_summary_confidences.json`.
  - **Check:** the new log parsing was run against the saved cell 14 output and
    reproduces every timing field in the committed JSON files.
- **Warm-up and repeats:** none. There is no warm-up call and one run per backend.
- **Validity guards:** cell 14 asserts exit code 0 and a wall-clock above 30 s;
  cell 16 asserts that the three AF3 timing log lines are present.

Breakdown from AF3's log (cell 14 output). Every row except "everything else" is a
field in `results/result_af3_*.json`:

| | Colab CPU | Colab GPU (T4) |
|---|---|---|
| Process wall-clock (`wall_clock_total_seconds`) | 2,453.99 s | 114.12 s |
| Featurisation, 1 seed (`featurisation_seconds`) | 12.92 s | 11.09 s |
| Model inference, seed 1, 5 samples, incl. JIT compilation (`model_inference_seconds`) | 2,401.38 s | 86.21 s |
| Sample extraction (`sample_extraction_seconds`) | 0.33 s | 0.18 s |
| Everything else: start-up, model build, parameter load, output writing (derived here) | 39.36 s | 16.64 s |
| Non-inference overhead, total / per sample (`non_inference_overhead_*`) | 52.61 s / 10.52 s | 27.91 s / 5.58 s |
| **`seconds_per_sample`** (model inference ÷ 5) | **480.28 s** | **17.24 s** |
| `wall_clock_seconds_per_sample` (published until 2026-09-15) | 490.80 s | 22.82 s |

- **Not like-for-like with AF2.** AF3's `seconds_per_sample` excludes process
  overhead but still includes JIT compilation; AF2 steady-state times exclude
  compilation.
  - **Effect:** the AF3/AF2 ratios (2.26× on CPU, 1.32× on GPU) overstate AF3's
    steady-state gap by an unmeasured amount.
  - **See:** `paper/data/canonical_results.md`, discrepancy 6.
- **Stanford CPU run:** only prose reports it (78.43 s/sample,
  `results/sweep/af3_comparison.md:176-177`); no log or config is in the repo
  **[NOT IN REPO]**. The notebooks' guard message mentions an existing CPU result
  of "~392s", which is consistent with 5 × 78.43 s.
- **TPU attempt:** it stopped at flag validation after 5.588 s real time
  (`results/sweep/af3_tpu_attempt.log:32-37`), so no inference was timed.

## 6. Repetitions per experiment

"Processes" is the number of independent `python3` launches per configuration
point. Every process times exactly one steady-state call.

| Experiment | Launcher | Script | Processes per point | Notes |
|---|---|---|---|---|
| Backend baseline, CPU | `alphafold_cpu_benchmark.ipynb` | notebook-embedded version | 1 | |
| Backend baseline, GPU | `alphafold_gpu_benchmark.ipynb` | notebook-embedded version | 1 | |
| Backend baseline, TPU | `configs/af_spike_job.yaml:37` | unknown version (Section 5.1) | 1 | |
| Sequence length (60, 120, 250, 500) | `configs/af_spike_job_sweep.yaml:47-48` | `spike_tpu_forward_pass.py` | 1 | |
| Recycle depth (0, 1, 3) | `configs/af_spike_job_sweep.yaml:52-53` | same | 1 | Recycle 3 was also launched alone (`configs/af_spike_recycle3_only.yaml:40`). Which of the two runs produced the reported value is **[NOT IN REPO]**. |
| Chip visibility (1, 8; 2 and 4 failed) | `configs/af_spike_chipcount.yaml:39-43`, `af_spike_chips_retry.yaml` | same | 1 | |
| Precision (float32, bfloat16) | `configs/af_spike_precision.yaml:32-35` | same | 1 | |
| Model comparison (model_3/4/5) | `configs/af_spike_combined.yaml:35-38` | same | 1 | |
| **Repeated runs** | `configs/af_spike_combined.yaml:45-48` | same | **3** | The only experiment with repeats; all 3 ran sequentially in one pod |
| Compilation cache | `configs/af_spike_combined.yaml:57-64` | same | 1 cold + 1 warm | Same pod as the two rows above |
| Profiler trace capture | `configs/af_spike_trace_capture.yaml:32-33` | same | 1 | Raw trace not in repo |
| `vmap` batching (B = 1, 2, 4, 8) | `configs/af_spike_batch.yaml:32-35` | `spike_batch_forward_pass.py` | 1 | |
| `pmap` multi-query (8 proteins) | `configs/af_spike_sharding.yaml:36-37` | `spike_pmap_forward_pass.py` | 1 | |
| GSPMD auto-mesh | `configs/af_spike_sharding.yaml:43-44` | `spike_meshshard_forward_pass.py` | **2** | `sharding.json` has `run_1` and `run_2`, but the Job launches the script once. How the second run was launched is **[NOT IN REPO]**. |
| `pmap` + `pmean` ensemble | `configs/af_spike_ensemble_shard.yaml:32-33` | `spike_ensemble_shard_forward_pass.py` | 1 | |
| Scaling grid (4 × 4 points) | `configs/af_spike_scaling_grid.yaml:34-41` | `spike_pmap_forward_pass.py` | 1 | 16 processes in total, one per point |
| AF3, Colab CPU / Colab GPU | `af3_*_colab.ipynb` | `run_alphafold.py` | 1 each | |
| AF3, Stanford CPU | **[NOT IN REPO]** | `run_alphafold.py` | 1 (per prose) | |
| AF3, TPU | `configs/af_spike_af3_tpu.yaml` | `run_alphafold.py` | 1 (failed) | |

What this implies for the paper:
- **Noise estimate.** The only one is from the 3 repeated TPU runs: CV 0.12%
  steady state, 0.82% `init_params`, 1.00% first predict. It applies to the TPU
  setup only; there is no noise estimate for the shared Colab CPU/GPU runtimes.
- **n = 1.** Every other comparison, including the headline CPU/GPU/TPU
  speedups, rests on a single process with a single steady-state call.
- **Uncontrolled ordering.** Experiments sharing a Job run back to back in one
  pod (e.g. model comparison → repeats → cache). Their order was not randomized,
  and different Jobs may have run on different nodes **[NOT IN REPO]**.

## 7. Data provenance

- **Per-run JSONs.** The scripts write one `result_<run_id>.json` per process
  (e.g. `src/spike_tpu_forward_pass.py:227-229`). Apart from the three baseline
  files and the two AF3 Colab files, those per-run JSONs are **[NOT IN REPO]**.
- **Sweep files.** `results/sweep/*.json` use a different, aggregated schema.
  The aggregation step (script or manual) is **[NOT IN REPO]**.
- **Logs.** No TPU Job logs are in the repo **[NOT IN REPO]**. The only raw TPU
  artifact is `results/sweep/af3_tpu_attempt.log`.
- **Dates.** Git cannot date the runs. Most results first appear in commit
  `2755250` (2026-08-08); the pieces added later are:
  - `scaling_law_data.json` in `cccd446` (2026-08-08);
  - `ensemble_shard.json` in `625c5dd` (2026-08-11);
  - `result_af3_cpu-colab.json` in `9c11e5b` (2026-08-12).

  These are upper bounds on the TPU run dates. The Colab dates come from the
  notebook logs.

## 8. Information not found in the repository

1. CPU model, vCPU count and RAM of the Colab CPU runtime (only prose).
2. Host CPU and RAM of the Colab GPU runtime.
3. TPU node: host CPU, vCPU, RAM, machine type, node identity; libtpu / TPU runtime version.
4. Hardware and launcher of the AF3 Stanford CPU run.
5. Resolved versions of all unpinned packages on TPU (`dm-haiku`, `numpy`, `absl-py`, `biopython`, `ml_collections`, `tensorflow-cpu`, `uv`) and the Python patch version.
6. Resolved JAX / jaxlib / CUDA library versions, TensorFlow version and Python version for the AF2 Colab runs.
7. AlphaFold2 commit (all runs) and AlphaFold3 commit for the TPU attempt.
8. The script version that produced `results/result_tpu-v5e-podslice.json`.
9. Raw per-run JSONs and Job logs for every TPU experiment, and how they were aggregated into `results/sweep/*.json`.
10. Which run supplied the recycle = 3 value (sweep Job or the recycle-3-only Job).
11. How the second GSPMD run was launched.
12. Size of the profiler overhead on TPU first-call timings.
13. Whether AF2 timings with trained weights match the random-init timings.
14. Run dates of all TPU experiments.
15. The raw profiler trace analysed in `profiling/trace_analysis.md`.
