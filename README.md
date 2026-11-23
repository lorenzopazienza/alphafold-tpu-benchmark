# AlphaFold2 Inference on an Eight-Chip TPU v5e Slice

This repository holds the code, Kubernetes Jobs, notebooks, and results of a study of AlphaFold2 inference on an
eight-chip TPU v5e slice. It also holds baselines on an NVIDIA T4 GPU and on a CPU runtime in Google Colab. We carried
out the study in the course ME 344, Introduction to High Performance Computing, at Stanford University in the Summer
Session 2026.

Follow-up research continues at
[github.com/lorenzopazienza/alphafold-on-tpu](https://github.com/lorenzopazienza/alphafold-on-tpu).

## Paper

Lorenzo Pazienza and Ihab El Bani. *Accelerator Choice Is Not Enough: AlphaFold2 Inference on Cloud TPUs.*
[arXiv:2609.34818](https://arxiv.org/abs/2609.34818) (cs.DC), 2026.
[PDF](paper/preprint/Pazienza_ElBani_2026_AlphaFold2_Inference_on_Cloud_TPUs.pdf) ·
[arXiv PDF](https://arxiv.org/pdf/2609.34818) · [LaTeX source](paper/preprint/source/)

This paper is version 1. It cites commit `a63e955` of this repository, which carries the tag
[`arxiv-v1`](https://github.com/lorenzopazienza/alphafold-tpu-benchmark/tree/arxiv-v1).

A revised version of the paper is in preparation. We rewrote it from the result files in this repository and ran no
new experiment. It corrects the following claims of version 1.

- Version 1 reported speed ratios of 451 between one TPU chip and the CPU runtime and of 27.8 between one TPU chip and
  the T4. These ratios divide times from different sessions and different copies of the benchmark script. The revised
  version gives only the smallest ratio over all sessions.
- Version 1 said that the Colab values came from "the same measurement campaign as the TPU numbers." Our records do
  not support this claim.
- Version 1 reported that one query per chip with `jax.pmap` on eight chips reached 6.92 times the throughput of the
  single-query baseline, with a parallel efficiency of 86.5%. The two runs differed in input and in timed region. The
  revised version takes the speedup only from a grid of runs with the same script, input family, and timed region.
- Version 1 gave a ratio of 59.1 between the TPU call that compiles the model and the steady-state call. The call that
  compiles the model includes the time the profiler takes to close, and the TPU runs did not record this time. The
  revised version drops the ratio.
- Version 1 reported that about 76% of a traced TPU call was self time in JAX tracing and compilation. We did not keep
  the trace file, so the revised version removes this value.
- Version 1 discussed the automatic-partitioning experiment in terms of GSPMD. The runs did not record which
  partitioner ran. The revised version reports the experiment as a run without sharding annotations.
- Version 1 had a section and an appendix on AlphaFold 3. The revised version removes them.

## Main Results

All runs used AlphaFold2 `model_3` with random weights. The scripts never loaded the trained checkpoint, and the MSA
held only the query sequence. The study concerns execution, not the quality of the predicted structures.

A query is one input sequence that the model predicts once. A steady-state call is a later call at the same input
shape, which reuses the compiled program. Throughput is the number of queries completed per second in steady-state
calls. A session is one Colab runtime or one Kubernetes Job in which a set of runs took place. Most TPU settings ran
once. The TPU values come from summaries of the Job output, which the section on records describes. The numbers below
come from the files named after them.

- **By default, one chip works.** With all eight chips visible, chip 0 had 463 MB of HBM in use and the other seven
  chips had none. The steady-state call took 0.469 s with one chip visible and 0.47 s with eight. Each setting ran
  once. The values are in
  [`results/sweep/chip_visibility_sweep.json`](results/sweep/chip_visibility_sweep.json).
- **Batching with `vmap` on one chip did not raise throughput.** `jax.vmap` ran B queries of 118 residues in one call on
  one chip. With B = 1, 2, 4, and 8, the throughput was 2.051, 1.928, 1.487, and 1.508 queries/s. The values are in
  [`results/sweep/batching_sweep.json`](results/sweep/batching_sweep.json).
- **Multi-query `pmap` put all eight chips to work.** `jax.pmap` ran one independent query on each chip. With eight
  queries of 118 residues, every chip had HBM in use, from 445 MB to 644 MB. We also ran it on a grid of 1, 2, 4, and
  8 chips and 100, 250, 500, and 1000 residues. Each query of the grid repeats the 20 amino-acid letters to the target
  length. Every point of the grid used the same script, input family, and timed region, and each point ran once. On
  this grid, eight chips gave 6.53 to 7.91 times the throughput of one chip. The summary gives the one-chip throughput
  at 1000 residues with two digits only, so the value of 7.91 is less precise than the others. The values are in
  [`results/sweep/sharding.json`](results/sweep/sharding.json) and
  [`results/sweep/scaling_law_data.json`](results/sweep/scaling_law_data.json).
- **Automatic partitioning did not shorten the call.** We ran one query inside a JAX mesh of eight chips whose axes are
  marked automatic. Our script gave the partitioner no sharding annotations. The steady-state call took 0.472 s and
  0.473 s in two runs. The baseline steady-state call on one chip took 0.47 s. The values are in
  [`results/sweep/sharding.json`](results/sweep/sharding.json) and
  [`results/result_tpu-v5e-podslice.json`](results/result_tpu-v5e-podslice.json).
- **The compilation cache shortened the start of a second process.** Two processes of the single-query script
  shared one cache directory. Parameter initialization traces the model and creates the random weights. It took
  37.68 s in the process that wrote the cache and 5.53 s in the process that read it. The call that compiles the model
  took 28.80 s in the process that wrote the cache and 15.19 s in the process that read it. The single-query script
  runs this call with the profiler, so both values include the time the profiler takes to close. For this reason, we
  do not attribute the 15.19 s to compilation. One pair of processes ran. The values are in
  [`results/sweep/compilation_cache.json`](results/sweep/compilation_cache.json).
- **The steady-state call was shorter on one TPU chip than on the T4 and the CPU runtime.** In every session, the mean
  steady-state call took more than 13.9 times as long on the T4 as the recorded call on one TPU chip, and more than
  450 times as long on the CPU runtime. Each August session timed one call, so its mean is that call. For the TPU, we
  used 0.471 s, the longest steady-state call that the single-query script recorded in the baseline configuration.
  The three platforms ran in different sessions and with different copies of the script. On the T4, the steady-state
  call took 13.086 s in August 2026 and a mean of 6.57 s in September 2026. On the CPU runtime, it took 212.113 s in
  August and means of 348.9 s to 359.3 s in three September sessions. In September, the CPU runtime had two logical
  CPUs. Our records do not explain the change between sessions. The values are in
  [`results/result_gpu-t4.json`](results/result_gpu-t4.json),
  [`results/result_cpu-colab.json`](results/result_cpu-colab.json),
  [`results/sweep/model_comparison.json`](results/sweep/model_comparison.json), and
  [`results/repro/`](results/repro/).

## Records

- The files in [`results/sweep/`](results/sweep/) are summaries of the Job output. They are not the files that the
  scripts wrote. Lorenzo Pazienza wrote them from the output of the Jobs with the help of AI assistants. We report their
  values as they are.
- The baseline result file [`results/result_tpu-v5e-podslice.json`](results/result_tpu-v5e-podslice.json) is the only
  result file of a single TPU run in the repository. The logs of the TPU Jobs, the result files of the other TPU runs,
  and the profiler trace are not in the repository.
- The JSON files keep the fields and labels of version 1. One example is the speedup of 6.92 in
  `results/sweep/sharding.json`. The notes in the Markdown files apply to these fields too.
- The T4 and CPU runtime baselines ran on 8 August 2026 with a copy of the benchmark script inside the notebooks. The
  notebooks in commit
  [`2755250`](https://github.com/lorenzopazienza/alphafold-tpu-benchmark/tree/2755250f734272b6cfbb0ec0cb35343e56acdc10/notebooks)
  hold this copy and the output of the runs. This copy differs from the scripts in [`src/`](src/).
- We cannot identify the script that ran the TPU baseline.
- In September 2026, we ran the committed script again on the T4 and the CPU runtime. The folder
  [`results/repro/`](results/repro/) holds these runs. Each run stores its commit, its package versions, and its host
  CPU.
- We did not repeat the TPU runs.
- The files in this repository that describe the results of version 1 carry a dated note at the top. The note lists
  the statements that the revised version corrects.

## Repository Layout

```text
configs/     Kubernetes Jobs of the TPU experiments
src/         Benchmark scripts that the Jobs run
notebooks/   Colab notebooks for the T4 and the CPU runtime
results/     Baseline results, TPU summaries in sweep/, September runs in repro/
profiling/   Written analysis of one profiler trace
paper/       Version 1 of the paper, results catalogue, and methodology notes
figures/     Figures of version 1
structure/   Predicted structures
scripts/     Launch and plotting scripts
cloud/       Scripts for a repeat on Cloud TPU, prepared and never run
Dockerfile   Container image, not used for any reported result
```

## Reproduction

The notebooks in [`notebooks/`](notebooks/) run the benchmark on the T4 and the CPU runtime in Google Colab. Each
notebook clones a commit of this repository that the user sets and runs the script in [`src/`](src/). The files in
[`configs/`](configs/) are the Kubernetes Jobs that ran the TPU experiments. They target the course cluster, which no
longer exists, so they do not run as written.

## Historical Records

The repository also holds runs of [AlphaFold 3](https://github.com/google-deepmind/alphafold3) from August 2026. Their
files have `af3` in the name and are in `notebooks/`, `results/`, `results/sweep/`, `structure/`, `figures/`,
`configs/`, `scripts/`, and `src/`. The revised paper does not include them. The statement on TPU support in
[`results/sweep/af3_comparison.md`](results/sweep/af3_comparison.md) refers to the public release of AlphaFold 3 that
we tried in August 2026.

The tag [`arxiv-v1`](https://github.com/lorenzopazienza/alphafold-tpu-benchmark/tree/arxiv-v1) holds the full
[README of version 1](https://github.com/lorenzopazienza/alphafold-tpu-benchmark/blob/arxiv-v1/README.md), the course
slides, and the project website.

## Authors and Citation

The authors are Lorenzo Pazienza and Ihab El Bani. Steve Jones and Mourad Bouache taught the course.

If you use this benchmark or its results, please cite the preprint.

```bibtex
@misc{pazienza2026accelerator,
  title         = {Accelerator Choice Is Not Enough: {AlphaFold2} Inference on Cloud {TPUs}},
  author        = {Pazienza, Lorenzo and El Bani, Ihab},
  year          = {2026},
  eprint        = {2609.34818},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  doi           = {10.48550/arXiv.2609.34818},
  url           = {https://arxiv.org/abs/2609.34818}
}
```

## License

The benchmark scripts, the Kubernetes Job definitions, the notebooks, the plotting scripts, the result files, and the
results catalogue are released under the MIT License. The license text is in [`LICENSE`](LICENSE).

[AlphaFold2](https://github.com/google-deepmind/alphafold) and AlphaFold 3 are not included in this repository. The
scripts clone or download them at run time. They remain under their own licenses and terms of use, which are
DeepMind's.
