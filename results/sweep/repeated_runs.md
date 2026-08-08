# Statistical Rigor: 3 Repeated Runs

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
reported elsewhere in this project (CPU/GPU/TPU comparison, sequence
length sweep, recycle sweep, etc.) is very unlikely to be a fluke --
the underlying measurement is highly reproducible run-to-run on this
hardware.
