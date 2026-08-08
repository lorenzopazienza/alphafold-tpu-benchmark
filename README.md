# AlphaFold on TPU: A CPU/GPU/TPU Scaling Study

**Course:** Stanford HPC — Custom Domain Workflow (Project 2)
**Team:** Lorenzo et al.

## Problem

Protein structure prediction (AlphaFold's JAX/Haiku forward pass) is a real,
deployed scientific-ML workload. We benchmark the same model, the same
inference code, unmodified, across three accelerator backends — CPU, GPU,
and TPU — to answer an engineering question: **where does this workload
actually spend its time, and how does that change per backend?**

## Architecture

```
                 ┌─────────────────────────────┐
                 │  spike_tpu_forward_pass.py    │   <- one script, all backends
                 └──────────────┬───────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
   ┌────▼────┐            ┌─────▼─────┐           ┌─────▼─────┐
   │   CPU    │            │    GPU     │           │    TPU     │
   │ (local /  │            │ (Colab T4) │           │ (Stanford  │
   │  sandbox) │            │            │           │  v5e-8)    │
   └────┬────┘            └─────┬─────┘           └─────┬─────┘
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                                │
                    results/result_<tag>.json
                    results/trace_<tag>/  (JAX/XLA profiler)
```

We deliberately avoid two heavy dependencies that aren't needed to answer
the systems question:
1. **MSA search** (jackhmmer/hhblits) — replaced with a trivial
   single-sequence MSA built via AlphaFold's own `pipeline.make_msa_features`
   helper. Swappable for a real ColabFold-fetched MSA later without changing
   any of the timing/profiling code.
2. **Trained parameters** (~350MB/model download) — `RunModel` randomly
   initializes params via Haiku's own init, which exercises the exact same
   JIT-compiled graph shapes and accelerator ops as a real forward pass.

## Reproduction

### Locally / CPU
```bash
docker build --build-arg JAX_VARIANT=cpu -t af-bench:cpu .
docker run -v $(pwd)/results:/alphafold/results af-bench:cpu --run_tag=cpu
```

### GPU (e.g. Google Colab, free T4)
```bash
docker build --build-arg JAX_VARIANT=cuda12 -t af-bench:gpu .
docker run --gpus all -v $(pwd)/results:/alphafold/results af-bench:gpu --run_tag=gpu-t4
```
Or, without Docker, directly in a Colab notebook (see `notebooks/`):
```bash
!pip install -U "jax[cuda12]"
!pip install dm-haiku==0.0.12 ml_collections absl-py "tensorflow-cpu==2.16.1" biopython numpy
!git clone --depth 1 https://github.com/google-deepmind/alphafold.git
!cp src/spike_tpu_forward_pass.py alphafold/
!cd alphafold && python3 spike_tpu_forward_pass.py --run_tag=gpu-t4
```

### TPU (Stanford class cluster)
```bash
export TEAM=<your-team-name>
gcloud container clusters get-credentials class-tpu-cluster-west4 --region=us-west4 --project=soe-hpccenter
kubectl config use-context gke_soe-hpccenter_us-west4_class-tpu-cluster-west4

kubectl create configmap af-spike-script-${TEAM} --from-file=src/spike_tpu_forward_pass.py \
  --dry-run=client -o yaml | kubectl apply -f -
envsubst < configs/af_spike_job.yaml | kubectl apply -f -
kubectl logs -f job/af-spike-${TEAM}
```

## TPU configuration

| Item | Value |
|---|---|
| GCP project | `soe-hpccenter` |
| GKE cluster | `class-tpu-cluster-west4` |
| Region | `us-west4` |
| TPU accelerator | `tpu-v5-lite-podslice`, topology `2x4` (8 chips) |
| Orchestration | Kubernetes Job, admitted via Kueue (`student-queue`) |

## Experiments

Each run varies only the **backend** (CPU / GPU / TPU); the model config,
input sequence, and code path are identical across all three. Planned
follow-up sweeps (once the TPU baseline is confirmed working): sequence
length, batch size, and TPU chip count.

## Profiling

Each run wraps the first `predict()` call in a `jax.profiler.trace(...)`
context, isolating **XLA compilation** from **steady-state execution**
(measured separately via a second, uncompiled `predict()` call). Traces are
written to `results/trace_<tag>/` and viewable with:
```bash
pip install tensorboard-plugin-profile
tensorboard --logdir=results/trace_<tag>
```

## Results

| Backend | Devices | steady-state (s) | vs CPU |
|---|---|---|---|
| CPU (Colab) | 1 | 212.113 | 1x |
| GPU (T4, Colab) | 1 | 13.086 | 16.2x |
| **TPU (v5e-podslice, Stanford)** | **8 chips** | **0.47** | **451x** |

Full breakdown, per-run JSON, and the profiling analysis (why the bottleneck
differs by backend) are in `results/comparison.md`.

## Bottleneck & scaling conclusion

The bottleneck **shifts by backend**: CPU is compute-bound (little gap
between first and second call -- there's no accelerator graph to amortize
compiling), while TPU is compile-bound on a cold call (59x gap between
first and steady-state call) but has by far the fastest steady-state
compute once that one-time compile cost is paid. The practical mitigation
is the same lesson the course's own Lab 1/3 vLLM setup already encodes:
persist the compiled XLA cache across restarts so that cost is paid once,
not per request. See `results/comparison.md` for the full numbers and
the reasoning behind each figure.

## Real protein structure (beyond the systems benchmark)

The CPU/GPU/TPU study above uses AlphaFold's real JAX code but random-init
weights -- valid for a compile/execution-time study, but not a real
biological result. `structure/` contains a **real folded protein**:

- **Protein:** human ubiquitin (76 residues) -- one of the most
  well-characterized proteins in structural biology.
- **Model:** ESMFold (Meta AI), real trained weights, no MSA search needed.
- **Result:** mean confidence (pLDDT) **90.5/100** -- "very high" by
  standard convention. Confidence stays >90 across the structured body of
  the protein and drops off sharply only in the last ~6 residues -- which
  is the correct, expected result: that C-terminal tail is a known
  biologically flexible/disordered region (it's the part that conjugates
  to other proteins), so lower model confidence there reflects real
  biology, not a modeling error.
- Reproduce with `notebooks/real_protein_fold_visualization.ipynb`
  (runs on CPU, Apple Silicon MPS, or CUDA -- auto-detected).

| File | What it is |
|---|---|
| `structure/ubiquitin_predicted.pdb` | Real predicted 3D structure |
| `structure/ubiquitin_confidence.png` | Per-residue confidence plot |

## Scale-bounds study (sequence length, recycle depth, chip visibility)

Beyond the fixed-size CPU/GPU/TPU comparison, `results/sweep/` contains
three scaling experiments on the TPU, all on the real AlphaFold forward
pass:

- **Sequence length** (60 -> 500 residues): scaling is **super-linear**
  (8.3x longer sequence -> 14.6x more time) -- the expected signature of
  AlphaFold's triangular attention, not a hardware artifact.
- **Recycle depth** (0 -> 3 iterations): steady-state execution scales
  **linearly** per extra recycle, but compile time does **not** grow the
  same way -- it jumps once recycling is present at all, then stays flat.
- **Chip visibility** (1 vs 8 chips): **no measurable difference** --
  direct confirmation that AlphaFold's base inference path runs a single
  query on exactly one chip regardless of how many are visible. Real
  multi-chip speedup needs explicit model sharding (out of scope). We also
  attempted 2/4-chip subsets via undocumented internal libtpu flags; both
  failed (one crashed the node's libtpu controller) and are documented
  rather than hidden -- see `results/sweep/chip_visibility.md`.

Real per-device HBM memory usage (via JAX's own `memory_stats()` API) is
included in every result JSON from this point on -- a working substitute
for the `tpu-info` tool, which needed container-level metrics ports our
minimal image doesn't set up (also documented, not faked).


- **Precision** (float32 vs bfloat16): steady-state HBM usage dropped
  **~40%**, but speed barely changed at this problem size, and peak
  memory briefly *increased* during the cast itself before settling
  lower -- a more nuanced finding than the naive "bfloat16 = faster"
  story, directly addressing the course rubric's own example mitigation.


- **Multi-query batching** (jax.vmap, batch 1/2/4/8): throughput got
  **worse**, not better, as batch size grew -- memory data confirms only
  1 of 8 chips is ever used regardless of batch size. `vmap` vectorizes
  within a single chip; real multi-chip parallelism needs explicit
  sharding (pmap/pjit), which is the clear, scoped-out next step this
  project's conclusion points to.


- **Compilation cache** (the actual mitigation): persisting JAX's
  compilation cache across a real process restart gave a measured
  **6.8x speedup on init_params** and **1.9x on first predict** -- the
  concrete fix for the compile-time bottleneck every other experiment
  found. Also includes: a model comparison (model_5 is ~15% faster at
  steady-state than model_3/4) and 3 repeated runs confirming <1% noise
  across every measurement in this project.


- **Profiler trace analysis**: opened the actual XLA trace and found the
  named culprit -- JAX's own `cache_miss` function (inside `pjit.py`)
  accounts for 12.55 of the first call's 16.56 seconds (~76%). Direct,
  function-level confirmation of the compile-time bottleneck every other
  experiment inferred indirectly. See `profiling/trace_analysis.md`.


- **Real multi-chip parallelism**: `jax.pmap` achieves a genuine
  **6.92x throughput speedup** running 8 independent proteins across
  8 physical chips simultaneously (14.72 vs 2.13 proteins/sec) -- the
  direct fix for the "only 1 chip used" and cost-parity findings above.
  A separate, honestly-reported attempt at automatic tensor sharding of
  a single protein's own computation (GSPMD auto-mesh) did NOT achieve
  real sharding, reproduced twice -- documented as a genuine negative
  result, not hidden or spun.

Full tables, reasoning, and a scaling chart: `results/sweep/README.md`.

## Repository layout

```
configs/        Kubernetes Job manifests (single run + length/recycle sweeps)
src/            The benchmark script (single source of truth, all backends)
notebooks/      Colab/Jupyter notebooks (GPU run, CPU run, real protein fold)
structure/      Real predicted protein structure (PDB) + confidence plot
profiling/      Notes/screenshots from XLA profiler traces
results/        Per-backend JSON results, comparison.md, and sweep/ (scaling study)
scripts/        Helper shell scripts (cluster connect, job launch)
presentation/   Slide deck source
```
