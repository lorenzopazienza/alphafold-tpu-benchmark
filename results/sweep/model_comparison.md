# Model Comparison (model_3 vs model_4 vs model_5)

## What this tests

AlphaFold ships 5 official model configs (model_1 through model_5), each
with slightly different architecture choices. Only model_3/4/5 default to
`use_templates=False`, matching this project's minimal (no-template)
pipeline; model_1/2 need template features we don't build.

## Results (118 residues, recycle=0, float32)

| Model | init_params (s) | compile+run (s) | steady-state (s) |
|---|---|---|---|
| model_3 | 36.67 | 28.72 | 0.471 |
| model_4 | 35.75 | 28.24 | 0.472 |
| model_5 | 35.62 | 28.00 | **0.403** |

## Finding

model_3 and model_4 are essentially identical across every metric. model_5
is consistently ~15% faster at steady-state (0.403s vs ~0.47s) while
having near-identical compile time, a real, measurable architectural
difference between the model variants, not noise (the gap is far larger
than the run-to-run variation measured in `repeated_runs.md`, ~0.06%
stdev). Worth noting for anyone choosing which of AlphaFold's model
checkpoints to deploy for latency-sensitive serving.

**Hardware:** Stanford GKE TPU v5e-8 (`tpu-v5-lite-podslice`, topology 2×4, 8 chips) via Kubernetes Job + Kueue. AF2 baseline comparison also uses Colab Intel Xeon CPU (2 vCPU) and Colab NVIDIA Tesla T4 (`results/comparison.md`).
