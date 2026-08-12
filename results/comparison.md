# CPU / GPU / TPU Comparison

All three runs use the **identical script** (`src/spike_tpu_forward_pass.py`),
the same 118-residue toy sequence, and the same model config (`model_3`, 0
recycles, random-init params). Only the backend changes.

| Backend | Devices | init_params (s) | 1st predict: compile+run (s) | 2nd predict: steady-state (s) |
|---|---|---|---|---|
| CPU (Colab) | 1 | 41.99 | 271.98 | 212.113 |
| GPU (T4, Colab) | 1 | 109.16 | 97.62 | 13.086 |
| **TPU (v5e-podslice, Stanford)** | **8 chips** | **36.6** | **27.78** | **0.47** |

**Hardware specifications:**

| | CPU (Colab) | GPU (Colab) | TPU (Stanford, not Colab) |
|---|---|---|---|
| Device | Intel Xeon (2 vCPU) | NVIDIA Tesla T4 | Google TPU v5e (`tpu-v5-lite-podslice`) |
| Memory | 13.6 GB RAM | 15 GB VRAM | 8 chips, topology 2x4, HBM per chip |
| Backend | `cpu` | `gpu` | `tpu` |
| Where it ran | Free-tier Colab runtime | Free-tier Colab runtime (T4) | Stanford GKE cluster (`class-tpu-cluster-west4`), via Kubernetes Job + Kueue, **not** Colab's own TPU offering |

The CPU and GPU specs above are Colab's standard free-tier allocation for
those runtime types. The TPU row is deliberately **not** Colab's TPU
runtime (which offers a single v5e-1 device) — every TPU result in this
project comes from the real 8-chip `v5e-lite-podslice` on Stanford's
class cluster, reached via `gcloud`/`kubectl`, not from selecting "TPU"
in Colab's runtime menu.

## Headline numbers

**Steady-state speedup vs CPU:**
- GPU (T4): **16.2x** faster than CPU
- TPU (v5e): **451x** faster than CPU

**Steady-state speedup vs GPU:**
- TPU (v5e): **27.8x** faster than a single T4

The TPU number is the strongest single result: identical 118-residue
AlphaFold forward pass, same code, three backends, a >400x gap between the
slowest and fastest.

## Where the time actually goes

| Backend | 1st predict / steady-state ratio | Interpretation |
|---|---|---|
| CPU | 1.28x | Compute-bound: most of the cost is the actual tensor math, not compilation |
| GPU (T4) | 7.46x | Mixed: compilation is a real but not dominant cost |
| TPU (v5e) | 59.1x | Compile-bound: the first call is almost entirely XLA compilation overhead |

This is the core profiling finding: **the bottleneck shifts backend to
backend.** On CPU, you're paying for raw compute the whole time, since
there's no accelerator-specific graph compilation to amortize. On TPU,
compute itself is so fast (0.47s) that a single cold call is completely
dominated by XLA compiling the Evoformer graph for the MXU. This is exactly
why production TPU serving (see the course's own Lab 1/3 vLLM setup)
persists a compiled XLA cache across restarts, paying that ~27s compile
cost once, not per request.

## Bottleneck diagnosis and engineering conclusion

1. **For a single cold inference**, TPU's advantage is real but partly
   masked by compile overhead (59x jump to steady-state). A one-shot script
   like this benchmark doesn't fully capture TPU's advantage unless you
   measure past the first call, which is exactly why we report both
   numbers rather than just wall-clock time for one run.
2. **For sustained/production workloads** (many structures processed by a
   long-lived, already-compiled process), the TPU's 27.8x edge over GPU and
   451x edge over CPU is the number that matters, and it's a direct,
   measured result, not extrapolated.
3. **Practical mitigation**, regardless of backend: batch multiple
   sequences of the same padded shape through one compiled graph, and keep
   the serving process warm (persist the XLA cache, as the course's own
   Lab 3 `VLLM_XLA_CACHE_PATH` does) so the ~27 to 110s compile cost is
   paid once, not per request.
4. **Natural next experiment** (out of scope for the 3-day spike, but a
   clear extension): does the TPU's relative advantage grow or shrink as
   sequence length and batch size increase, and how does it scale across
   the full 8-chip slice vs a single chip? Answered in `sweep/`.
