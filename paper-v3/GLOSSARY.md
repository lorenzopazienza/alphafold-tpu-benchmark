# Glossary of the paper

The paper uses "query" and "slice" throughout. Each term is defined once, in the section that the table gives, and then always used in the same form. `tools/check_style.py` reads the list of forbidden variants at the end of this file.

| Term | Definition in the paper | Where | Variants not to use |
|---|---|---|---|
| first call | The first `predict` or `apply` call at a new input shape. It traces, compiles, and executes the model. | III-C | cold start, cold call, compile + run, first predict, first-predict call, cold first predict |
| steady-state call | A later call at the same input shape, which reuses the compiled program. | III-C | second predict, warm call, steady-state latency, steady state as a noun for the call |
| parameter initialization | The `init_params` call, which traces the model and creates the random weights. | III-C | init, init_params region, initialization without "parameter" |
| compilation | Tracing to a jaxpr plus XLA compilation. We do not separate the two. | II-B | JIT compilation, trace-and-compile, compile step |
| profiler teardown | The time the `jax.profiler.trace` context takes to close after the first call returns, inside the timer. | III-C | trace finalization, profiler overhead |
| compilation cache | JAX's cache of compiled programs on disk, shared by two processes. | II-B | persistent compilation cache, XLA cache, warm cache as a name |
| chip | One TPU v5e chip. JAX treats each chip as one device. | II-A | core, TPU_0 as a generic name, "8 cores", "exposes" for how JAX sees a chip |
| slice | The set of eight TPU v5e chips (2 x 4) that each Kubernetes Job received. | II-A | pod, TPU pod, pod slice, podslice, v5e-8 |
| host | The CPU machine that runs the Python process and drives the chips. | II-A | driver machine |
| platform | One of the three hardware targets: the TPU v5e chip, the NVIDIA T4 GPU, or the Colab CPU runtime. | III-A | backend, accelerator for the CPU |
| CPU runtime | A Google Colab CPU runtime. The September sessions had two logical CPUs. The August sessions did not record theirs. | III-A | Colab CPU, CPU backend |
| session | One Colab runtime, or one Kubernetes Job, in which a set of runs took place. | III-A | campaign, measurement campaign |
| query | One input sequence, predicted once. | I | protein or prediction as the unit of work |
| throughput | Queries completed per second in steady-state calls. Unit: queries/s. | III-C | proteins/s, proteins per second, predictions per hour |
| speedup | The throughput on n chips divided by the throughput on one chip, on the same input family and timed region. | III-C | throughput gain |
| parallel efficiency | The speedup divided by the number of chips. | III-C | scaling efficiency, efficiency without "parallel" |
| batching with `vmap` | One call that runs B queries on one chip through `jax.vmap`. | IV-A | vectorization, batched serving |
| multi-query `pmap` | Independent queries, one per chip, through `jax.pmap`. | IV-B | pmap data parallelism, real multi-chip parallelism |
| automatic partitioning | Running one query inside a JAX mesh with automatic axes, so that the compiler's partitioner decides the placement. | IV-C | auto-mesh, auto-sharding, automatic sharding, GSPMD or Shardy as the name of the experiment |
| external ensemble | One query featurized with eight seeds, one featurization per chip through `jax.pmap`, with the `predicted_lddt` logits averaged by `jax.lax.pmean`. | IV-C | ensemble sharding, ensemble-shard |
| matched grid | The 16 runs of multi-query `pmap` over 1, 2, 4, and 8 chips and 100, 250, 500, and 1000 residues. | IV-B | scaling grid, 16-point grid |
| power-law fit | The least-squares fit of log throughput on log chips and log length. | IV-B | scaling law, empirical scaling law, fitted law |
| fixed sequence | The 118-residue sequence used by the single-query experiments. | III-B | toy sequence, synthetic sequence |
| repeated-alphabet sequence | The 20 amino-acid letters repeated to the target length, shifted by one letter per query. | III-B | rotated alphabet, rotated-alphabet input family |
| random weights | Parameters initialized at random by Haiku. The trained checkpoint is never loaded. | III-B | untrained weights, random-init |
| HBM in use | The value of `bytes_in_use` that JAX reports for a chip after the first steady-state call. It is the device memory allocated at that moment. | III-C | memory footprint, per-chip footprint |
| peak memory | The largest HBM in use over the life of the process. | III-C | peak HBM, maximum memory |
| baseline configuration | The single-query script on the fixed sequence with `model_3`, no recycling, and float32 parameters. It times `predict`. One run of it is a baseline run. | III-B | baseline alone, TPU baseline, single-query baseline, baseline time, baselines |

## Forbidden variants (machine-readable, one regular expression per line, case-insensitive)

```
cold[- ]start
cold call
warm call
second predict
first[- ]predict
steady-state latency
trace-and-compile
persistent compilation cache
XLA cache
\bpod\b
\bpods\b
pod[- ]?slice
v5e-8
\bcores?\b
proteins?/s
proteins? per second
predictions per hour
measurement campaign
\bcampaign
\bbackends?\b
Colab CPU\b(?! runtime)
throughput gain
scaling efficiency
scaling law
auto-?mesh
auto-?sharding
automatic sharding
ensemble[- ]shard
toy sequence
synthetic sequence
rotated[- ]alphabet
untrained weights
random-init
memory footprint
per-chip footprint
\bvectori[sz]ation
\bexpos(?:e|es|ed|ing)\b
\bbaselines?\b(?! (?:configuration|runs?)\b)
```
