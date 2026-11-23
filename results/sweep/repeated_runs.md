# Statistical Rigor: 3 Repeated Runs

> Note (November 2026): this file is part of the record of version 1 of the paper. Version 3 of the paper
> (arXiv:2609.34818v3) corrects some statements below.
>
> - The three runs measured the TPU only. This file extends their low variation to the comparison with the T4 and
>   the CPU runtime. The times on these two platforms changed between sessions. In September 2026, the T4 took a
>   mean of 6.57 s per steady-state call instead of 13.086 s in August.

## What this tests

Every other number in this project is a single measurement. This repeats
the exact same configuration (model_3, 118 residues, recycle=0, float32)
3 times to check how much run-to-run noise exists.

## Results

| Repeat | init_params (s) | compile+run (s) | steady-state (s) |
|---|---|---|---|
| 1 | 37.54 | 27.56 | 0.470 |
| 2 | 37.02 | 27.96 | 0.469 |
| 3 | 37.01 | 27.43 | 0.469 |
| **mean ± stdev** | **37.19 ± 0.30** | **27.65 ± 0.28** | **0.4693 ± 0.0006** |

## Finding

Extremely low variance across all three metrics (stdev under 1% of the
mean in every case). This means every other single-measurement number
reported elsewhere in this project (CPU/GPU/TPU comparison on Google Colab Intel Xeon CPU (2 vCPU), Google Colab NVIDIA Tesla T4, and Stanford GKE TPU v5e-8 (2×4 lite); sequence
length sweep, recycle sweep, and so on) is very unlikely to be a fluke;
the underlying measurement is highly reproducible run-to-run on this
hardware.
