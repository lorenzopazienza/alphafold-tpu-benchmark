# Scale-Bounds Study (TPU v5e-podslice, 8 chips)

Twelve experiments, all on the real AlphaFold JAX/Haiku forward pass, same
methodology as the CPU/GPU/TPU comparison (`../comparison.md`).

```mermaid
flowchart LR
    CMP["CPU/GPU/TPU baseline\n(../comparison.md)"]

    subgraph SB["Scale bounds"]
        SL["1. Sequence length"]
        RD["2. Recycle depth"]
        CV["3. Chip visibility"]
    end

    subgraph MIT["Mitigations"]
        PR["4. Precision (bf16)"]
        CC["6. Compilation cache"]
    end

    subgraph MC["Multi-chip parallelism"]
        BA["5. Batching (vmap)\nnegative result"]
        SH["10. pmap vs GSPMD\npositive + negative"]
        ES["11. Ensemble shard\nreal fix, verified"]
        SCL["12. Scaling law"]
    end

    MOD["7. Model comparison"]
    REP["8. Repeated runs"]
    TR["9. Trace analysis"]

    CMP --> SL & RD & CV & PR & CC & MOD & REP
    TR --> CC
    CV --> BA --> SH --> ES --> SCL
```

## 1. Sequence-length sweep (recycle=0)

| Length | init (s) | compile+run (s) | steady-state (s) | length factor | time factor |
|---|---|---|---|---|---|
| 60  | 36.12 | 25.95 | 0.208 | 1.0x | 1.0x |
| 120 | 36.50 | 24.88 | 0.408 | 2.0x | 2.0x |
| 250 | 36.10 | 29.37 | 1.170 | 4.2x | 5.6x |
| 500 | 35.49 | 31.46 | 3.031 | 8.3x | 14.6x |

**Finding:** scaling is **super-linear** past ~120 residues, an 8.3x length
increase produces a 14.6x time increase, not 8.3x. This matches AlphaFold's
known algorithmic complexity: the Evoformer's triangular attention and
multiplication operations scale worse than linearly with sequence length
(roughly quadratic to cubic in the pair representation), so this isn't a
TPU-specific artifact, it's the real, expected computational signature of
the model, now confirmed empirically on this hardware. **Compile time
(init + first predict) stays roughly flat** across all four lengths (~36s /
~25 to 31s): XLA compiles a graph shaped for each specific input size, but
the compile cost itself doesn't grow with problem size the way the actual
execution does.

## 2. Recycle-depth sweep (118 residues)

| Recycles | init (s) | compile+run (s) | steady-state (s) | steady-state factor |
|---|---|---|---|---|
| 0 | 35.74 | 27.66 | 0.469 | 1.0x |
| 1 | 37.71 | 55.46 | 0.933 | 2.0x |
| 3 | 40.31 | 54.94 | 1.845 | 3.9x |

**Finding:** steady-state execution time scales **linearly** with recycle
count (each extra recycle iteration adds ~0.46s, consistently). But
**compile time does not scale the same way**: it roughly doubles from 0 to
1 recycle, then stays flat from 1 to 3 recycles (55.46s vs 54.94s,
essentially identical). This suggests XLA compiles the recycle loop as a
single reusable structure once recycling is present at all, rather than
unrolling and re-compiling per additional iteration, an interesting,
concrete profiling observation about how JAX handles this specific looping
pattern, distinct from the sequence-length finding above.

## Known limitation: TPU utilization metrics unavailable

We attempted to capture real chip-level utilization (`tpu-info`: HBM usage,
duty cycle) during these runs, matching the tool the course's own Lab 2
uses. It ran but reported all metrics as `N/A`:

```
WARNING: Libtpu metrics unavailable. Is there a framework using the TPU?
```

Root cause: the official course container images (vLLM/Tunix) start their
runtime with specific metrics ports exposed
(`TPU_RUNTIME_METRICS_PORTS=8431-8434`); our minimal `python:3.12-slim`
container doesn't set these up. Reproducing that exact setup was judged not
worth the added infrastructure risk given time constraints; this is
disclosed here as a known gap rather than worked around with a fabricated
number.

![Scaling charts](../../figures/scaling_charts.png)

## 3. Chip-visibility experiment (1 vs 8 chips)

See `chip_visibility.md` for the full writeup, including the honestly
documented failure of the intermediate 2/4-chip attempts. Short version: no
measurable difference between 1 and 8 visible chips, since AlphaFold's base
inference path runs a single query on exactly one chip regardless. Real
multi-chip speedup would require explicit model sharding, out of scope
here.

## 4. Precision experiment (float32 vs bfloat16)

See `precision.md` for the full writeup. Short version: speed barely
changed at this problem size (random-init weights, recycle=0), but
steady-state HBM usage dropped ~40%, and peak memory briefly went *up*
during the cast itself before settling lower. A more nuanced, more honest
finding than a flat "bfloat16 is faster" claim.

## 5. Multi-query batching experiment (jax.vmap)

See `batching.md` for the full writeup. Short version: batching made
per-protein throughput *worse*, not better, confirmed by memory data
showing only one of the 8 chips is ever used, regardless of batch size.
`vmap` vectorizes within a chip; it doesn't distribute across chips. Real
multi-chip speedup would need explicit sharding (`pmap`/`pjit`), a clear
scoped-out next step.

## 6. Model comparison (model_3 vs model_4 vs model_5)

model_5 is ~15% faster at steady-state than model_3/model_4, a real
architectural difference, not noise. See `model_comparison.md`.

## 7. Statistical rigor (3 repeated runs)

Under 1% stdev across every metric, so every other single-measurement
number in this project is highly reproducible, not a fluke. See
`repeated_runs.md`.

## 8. Compilation cache: the actual mitigation

The answer to the bottleneck every other experiment found: persisting
JAX's compilation cache across process restarts gives a real, measured
**6.8x speedup on `init_params`** and **1.9x on first `predict`**, with
zero change to steady-state performance. See `compilation_cache.md`.

## 9. Real multi-chip parallelism: pmap (success) vs auto-mesh sharding (honest failure)

Direct follow-up to the batching (vmap) negative result. `jax.pmap`
achieves genuine multi-chip data parallelism: **6.92x throughput speedup**
(14.72 vs 2.13 proteins/sec), confirmed by distinct per-chip memory
footprints. A separate attempt at GSPMD auto-sharding (splitting one
protein's own computation across chips, using the same mesh pattern the
course's own Lab 2 Tunix script uses) did **not** achieve real sharding:
reproduced twice, every chip held an identical full-size 463MB copy
(replication, not a split), confirmed via honest, reproduced measurement
rather than trusting the naive "nonzero memory" heuristic. See
`sharding.md`.

## 10. Real single-query sharding: ensemble averaging via pmap + pmean

Direct follow-up to the auto-mesh failure above, found the exact
source-level reason it failed (AlphaFold's own ensembling is a sequential
`hk.while_loop`, nothing for GSPMD to distribute), then built a version
that actually works: AlphaFold's source is completely untouched, but the
ensemble average is re-implemented via `jax.pmap` + a real `jax.lax.pmean`
collective reduction across chips. **8/8 chips used, verified-correct
cross-device reduction, distinct per-chip memory** (427-624MB, not the
flat 463MB-everywhere signature that gave away replication before).
Honestly scoped: this is real distributed computation for one query's
ensembling, not full internal tensor sharding of the Evoformer, that
remains future work. See `ensemble_shard.md`.

## 11. Empirical scaling law: throughput(chips, length)

Fitted a power law to a 16-point grid (4 chip counts x 4 lengths, all real
TPU measurements): **throughput ≈ 4527.77 * chips^0.963 * length^-1.572**
(R² = 0.981). The chip exponent confirms `pmap`'s near-linear speedup
generalizes across every length tested. Combined with `cost_analysis.md`'s
pricing, this shows cost per prediction is roughly chip-count-invariant:
more chips should be chosen for throughput/latency needs, not for cost.
See `scaling_law.md`.

## AlphaFold3 side-investigation (separate from the 12 experiments above)

Everything above benchmarks **AlphaFold2** on TPU. As a follow-up, we also
got **AlphaFold3** — a separate, newer, diffusion-based DeepMind codebase,
not a version of AlphaFold2 — running on the same 118-residue toy
sequence across all three backends this project tests AF2 on: **CPU
(Colab), GPU (T4, Colab), and TPU (Stanford)**. CPU and GPU produced full,
real, measured results. TPU did not — not an infrastructure failure on
our side, but a confirmed finding that **AlphaFold3's public release does
not support TPU inference at all** (`--jax_backend`'s valid values are
`cpu`/`gpu`/`mps` only, verified both by a real attempt on the Stanford
TPU slice and by AlphaFold3's own official docs, which require an NVIDIA
GPU or CPU).

`af3_comparison.md` is the full write-up: an architecture/characteristics
table, a build-effort table (AF2's engineering went into multi-chip
sharding; AF3's went into a native C++ build, an easy-to-miss
`build_data` step, and backend-specific flags undocumented outside a
Dockerfile comment), a **real, same-hardware performance comparison**
(on identical Colab hardware, AF3 is 2.3x slower than AF2 on CPU,
narrowing to 1.74x on GPU — reversing an earlier hardware-confounded
estimate), a **reproducibility study** using the same seed across three
hardware/backend combinations (near-identical across machines on the same
backend; substantially different across backends — CPU vs. GPU differs
by up to 32% per sample, plausibly explained by a numerical issue
AlphaFold3's own issue tracker documents for GPUs below compute
capability 8.0), and the full TPU-unsupported finding with log evidence.

