# CPU / GPU / TPU Comparison

> Note (November 2026): this file is part of the record of version 1 of the paper. A revised version of the paper
> corrects some statements below.
>
> - This file says that the three runs used the identical script. The T4 and the CPU runtime ran a copy of the
>   script inside the notebooks. The notebooks in commit `2755250` hold this copy. We cannot identify the script
>   that ran the TPU baseline. AlphaFold2's source ran unmodified, but the scripts differed.
> - The ratios of 451, 27.8, and 16.2 compare platforms. These ratios divide times from different sessions and
>   different copies of the script. In September 2026, the T4 took a mean of 6.57 s per steady-state call instead of
>   13.086 s. The CPU runtime took means of 348.9 s to 359.3 s instead of 212.113 s. The revised version gives only
>   the smallest ratio over all sessions. In every session, the mean steady-state call took more than 13.9 times as
>   long on the T4 as the recorded call on one TPU chip, and more than 450 times as long on the CPU runtime. The
>   revised version gives no ratio between the T4 and the CPU runtime.
> - The ratios of 1.28, 7.46, and 59.1 compare the call that compiles the model with the steady-state call. The call
>   that compiles the model includes the time the profiler takes to close inside the timer. The TPU runs did not
>   record this time. The revised version drops these ratios. It gives the TPU call of 27.78 s with this condition.
>   For the T4 and the CPU runtime, it uses the September runs without the profiler.

All three runs use the **identical script** (`src/spike_tpu_forward_pass.py`),
the same 118-residue toy sequence, and the same model config (`model_3`, 0
recycles, random-init params). Only the backend changes.

| Backend | Devices | init_params (s) | 1st predict: compile+run (s) | 2nd predict: steady-state (s) |
|---|---|---|---|---|
| CPU (Google Colab Intel Xeon, 2 vCPU) | 1 | 41.99 | 271.98 | 212.113 |
| GPU (Google Colab NVIDIA Tesla T4) | 1 | 109.16 | 97.62 | 13.086 |
| **TPU (Stanford GKE v5e-8, 2×4 lite)** | **8 visible, 1 used** | **36.6** | **27.78** | **0.47** |

**Hardware specifications:**

| | CPU (Google Colab Intel Xeon, 2 vCPU) | GPU (Google Colab NVIDIA Tesla T4) | TPU (Stanford GKE v5e-8, 2×4 lite, not Google Colab) |
|---|---|---|---|
| Device | Intel Xeon (2 vCPU) | NVIDIA Tesla T4 | Google TPU v5e (`tpu-v5-lite-podslice`) |
| Memory | 13.6 GB RAM | 15 GB VRAM | 8 chips, topology 2x4, HBM per chip |
| Backend | `cpu` | `gpu` | `tpu` |
| Where it ran | Free-tier Google Colab runtime | Free-tier Google Colab runtime (NVIDIA Tesla T4) | Stanford GKE cluster (`class-tpu-cluster-west4`), via Kubernetes Job + Kueue, **not** Google Colab's own TPU offering (not used here) |

The CPU and GPU specs above are Google Colab's standard free-tier allocation for
those runtime types. The TPU row is deliberately **not** Google Colab's TPU
runtime (which offers a single v5e-1 device) - every TPU result in this
project comes from the real 8-chip `v5e-lite-podslice` on Stanford's
class cluster, reached via `gcloud`/`kubectl`, not from selecting "TPU"
in Google Colab's runtime menu.

## Headline numbers

**Steady-state speedup vs CPU:**
- GPU (Google Colab NVIDIA Tesla T4): **16.2x** faster than CPU
- TPU (Stanford GKE v5e-8, 2×4 lite): **451x** faster than CPU, running on 1 of the slice's 8 chips

**Steady-state speedup vs GPU:**
- TPU (Stanford GKE v5e-8, 2×4 lite): **27.8x** faster than a single Google Colab NVIDIA Tesla T4, running on 1 of the slice's 8 chips (effectively one v5e chip vs one T4)

The TPU number is the strongest single result: identical 118-residue
AlphaFold forward pass, same code, three backends, a >400x gap between the
slowest and fastest. The slice exposes 8 chips, but this single-query run uses only `TPU_0` (`TPU_1`-`TPU_7` hold 0 MB, see `sweep/chip_visibility.md`), so both TPU ratios compare one v5e chip against one T4 or 2 vCPUs, not the full 8-chip slice.

## Where the time actually goes

| Backend | 1st predict / steady-state ratio | Interpretation |
|---|---|---|
| CPU | 1.28x | Compute-bound: most of the cost is the actual tensor math, not compilation |
| GPU (Google Colab NVIDIA Tesla T4) | 7.46x | Mixed: compilation is a real but not dominant cost |
| TPU (Stanford GKE v5e-8, 2×4 lite) | 59.1x | Compile-bound: the first call is almost entirely XLA compilation overhead |

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
   long-lived, already-compiled process), the single TPU chip's 27.8x edge over GPU and
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
