# Compilation Cache: The Actual Mitigation

## Why this experiment exists

Every other experiment in this project found the same thing: **XLA
compilation dominates the cost of a cold call** (27-55s of every run,
repeated identically every single time we ran the benchmark). This
experiment tests the actual fix -- JAX's persistent compilation cache
(`jax_compilation_cache_dir`), the same category of mitigation the
course's own Lab 1/3 uses in production (`VLLM_XLA_CACHE_PATH`).

## Method

Two **separate Python processes** (not two calls within one script --
a genuinely fresh process each time, simulating a real restart) pointed
at the same cache directory (`/tmp/jax_cache`). The first process
compiles from scratch and writes the cache; the second reads it back.

## Results (model_3, 118 residues, recycle=0, float32)

| | init_params (s) | compile+run (s) | steady-state (s) |
|---|---|---|---|
| Cold (writes cache) | 37.68 | 28.80 | 0.470 |
| **Warm (reuses cache)** | **5.53** | **15.19** | 0.469 |
| **Speedup** | **6.81x** | **1.90x** | ~1x (unaffected, as expected) |

## Finding

The cache works, and works well. `init_params` (the first JIT trace)
drops **6.8x** when a fresh process can read a previously-compiled
artifact instead of recompiling from scratch. `compile+run` drops
**1.9x** (less dramatic, since `predict()`'s own JIT trace is a
different, separately-cached artifact that still partially compiles
here -- a next-step optimization would be pre-warming both caches
before the timed region). Steady-state performance is unaffected, as
expected -- it was never compile-bound.

## This is the answer to this project's own bottleneck diagnosis

Every scaling experiment in `results/sweep/` found compilation to be
the dominant cost of a cold call. This experiment demonstrates the real
mitigation is not a hardware or precision change -- it's operational:
**persist the compiled artifact and don't pay the compile cost more than
once.** This is exactly the practice the course's own Lab 3 already
encodes for LLM serving; this experiment confirms the same technique
gives a large, measured win for a completely different real workload
(protein structure prediction).
