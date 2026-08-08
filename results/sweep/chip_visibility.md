# Chip-Visibility Experiment (1 vs 8 TPU chips)

## What this tests

Whether the number of TPU chips *visible* to the process (`TPU_VISIBLE_CHIPS`)
affects compile time, execution time, or memory usage for a single-query
AlphaFold forward pass.

**Important scoping note:** AlphaFold's base `RunModel.predict()` path does
**not** shard a single query's computation across multiple chips -- that
would require explicit model sharding (`pjit`/mesh annotations), which is
out of scope for this project. So this experiment does not test "does more
chips make one fold faster" (the honest answer, confirmed below, is no) --
it tests something narrower and still legitimate: does chip *visibility*
itself carry any overhead or side effect.

## Results: 1 chip vs 8 chips

| Chips visible | init_params (s) | compile+run (s) | steady-state (s) | HBM used (any chip) |
|---|---|---|---|---|
| 1 | 36.32 | 27.37 | 0.469 | 463 MB (TPU_0) |
| 8 | 36.18 | 27.43 | 0.470 | 463 MB (TPU_0 only; TPU_1-7 all 0 MB) |

**Finding:** No measurable difference. Compile time, execution time, and
memory footprint are identical within noise, regardless of how many chips
are visible. Direct, measured confirmation that the computation runs on
exactly one chip either way -- real multi-chip speedup for a single query
would require rewriting the model with explicit sharding, a genuinely
different (and larger) engineering task than what a 3-day systems spike
covers.

## Attempted: 2 and 4 chip subsets (documented failure)

We also attempted intermediate points (2 and 4 visible chips) to get a
denser sweep. Both failed:

- **2 chips** (`TPU_VISIBLE_CHIPS=0,1`): `JaxRuntimeError: TPU initialization
  failed: The number of devices found in the host does not match the
  topology, expected 8, actual: 2.` The runtime expects the visible chip
  count to match the physical slice topology (2x4 = 8) unless the topology
  bounds are also explicitly reconfigured.
- **Retry with topology bounds** (`LIBTPU_INIT_ARGS
  --deepsea_chips_per_host_bounds=2,1,1 --deepsea_host_bounds=1,1,1` for 2
  chips, `2,2,1` for 4 chips): the 2-chip attempt still failed; the 4-chip
  attempt crashed the **libtpu controller process itself**
  (`SLICE_FAILURE_INIT_ERROR`), aborting the whole pod.

These are undocumented, internal libtpu flags (the error message itself
says "Contact tfrt-devs@" for questions) -- getting arbitrary chip-count
subsets working would mean guessing at unpublished internal configuration
on a shared class TPU slice, with real risk of destabilizing the node for
other teams. We stopped after this one documented attempt rather than
continue guessing at internal flags under time pressure. The two clean
data points (1 and 8 chips) already answer the question this experiment
was designed to ask.
