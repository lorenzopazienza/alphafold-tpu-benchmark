# Real Single-Query Sharding: Ensemble Averaging via pmap + pmean

## Motivation

`sharding.md` showed that neither `jax.vmap` nor GSPMD auto-mesh sharding
achieve real distributed computation for a *single* AlphaFold query, both
either stayed on one chip (`vmap`) or silently replicated the full
computation on every chip instead of splitting it (auto-mesh, confirmed
via identical per-chip memory). This experiment investigates why, and
builds a version that actually works.

## Why the earlier attempts failed

AlphaFold's own ensembling (`config.data.eval.num_ensemble > 1`, used to
average several stochastic MSA-sample forward passes of the same
protein) is implemented internally as a **sequential** `hk.while_loop`
(`modules.py`, `AlphaFoldIteration`): each ensemble member's Evoformer
pass runs one after another, accumulating a running sum. A sequential
loop with a carried accumulator has nothing independent for XLA's
auto-partitioner to distribute across chips, this is the concrete,
source-level reason GSPMD auto-mesh sharding defaulted to full
replication rather than splitting anything.

## What this experiment does instead

Re-implements the same mathematical operation, average N independent
forward passes of one query, using real distributed hardware:

- AlphaFold's own source code is **completely unmodified**
  (`num_ensemble` stays 1 internally; no monkey-patching of Haiku
  modules, avoiding any risk to AlphaFold's own parameter-initialization
  logic).
- Instead, N=8 independent copies of the same query are built externally
  (each with its own `random_seed`) and distributed one per physical
  chip via `jax.pmap`, the same primitive already proven to work in
  `sharding.md`'s data-parallel experiment.
- A genuine cross-device reduction (`jax.lax.pmean`) computes the average
  across chips, a real collective operation, not a host-side average
  after the fact.

**Implementation detail that mattered:** the reduction had to operate on
the model's *raw* `predicted_lddt` logits (a plain JAX array), not the
human-readable `plddt` score. AlphaFold computes `plddt` via
`get_confidence_metrics()`, which uses plain `numpy` internally, numpy
cannot operate on JAX tracers inside a `pmap`-traced function. This is
exactly why AlphaFold's own `predict()` keeps that conversion as a
separate step after `apply()` returns concrete arrays, never inside the
jitted/pmapped computation itself. The first version of this script hit
exactly this error; fixed by reducing the raw logits instead and leaving
the human-readable conversion for after the pmapped call returns.

## Results (118 residues, 8 ensemble members / 8 chips)

| | Value |
|---|---|
| Compile + first run | 16.61s |
| Steady-state | 0.538s |
| Chips with nonzero memory | **8 / 8** |
| Cross-device `pmean` verified consistent | **True** |

Per-chip memory (confirms genuine per-chip work, not replication, compare
to the identical 463MB-on-every-chip signature from the failed auto-mesh
attempt):

| Chip | In use | Peak |
|---|---|---|
| TPU_0 | 624 MB | 731 MB |
| TPU_1-5 | 427 MB | 450 MB |
| TPU_6-7 | 450 MB | 450 MB |

## Honest scope of what this proves

This is real, verified distributed computation and a real collective
reduction for one query's ensembling, a legitimate, narrower answer to
"real single-query sharding" than fully sharding the Evoformer's internal
tensors would be (splitting the pair/MSA representations *within* one
forward pass, which would need explicit `PartitionSpec` annotations
threaded through AlphaFold's own attention and triangular-multiplication
modules, a substantially larger rewrite, correctly scoped as further
future work, not attempted here).

**Also worth stating plainly:** this benchmark's MSA is a trivial
single-sequence toy MSA (no real alignment depth), so the 8 "ensemble
members" are near-identical inputs. This experiment proves the
distribution and reduction mechanism work correctly on real hardware, it
is not a scientifically meaningful ensemble average (that would need a
real multi-sequence MSA with genuine stochastic subsampling diversity
between members).

**Hardware:** Stanford GKE TPU v5e-8 (`tpu-v5-lite-podslice`, topology 2×4, 8 chips) via Kubernetes Job + Kueue. AF2 baseline comparison also uses Google Colab Intel Xeon CPU (2 vCPU) and Google Colab NVIDIA Tesla T4 (`results/comparison.md`).
