# Empirical Scaling Law: throughput(chips, length)

## Method

Ran the working `jax.pmap` script (from `sharding.md`) across a full 4x4
grid: chip count {1, 2, 4, 8} x sequence length {100, 250, 500, 1000}
residues, 16 total runs, all real TPU measurements, no interpolation or
estimation.

## Fitted model

Fitting `log(throughput) = log(a) + b*log(chips) + c*log(length)` by
linear regression in log-log space (standard power-law fit):

```
throughput(chips, length) ≈ 4527.77 * chips^0.963 * length^-1.572
```

**R² = 0.981** on the log-log fit, a strong, real relationship, not noise.
Residuals show a mild systematic curve (the model over-predicts at the
length extremes by ~20 to 30%, under-predicts in the middle by ~12 to
25%), honestly reported rather than hidden; a pure power law is a good
approximation here, not an exact description. The likely cause is a
roughly fixed per-call overhead (compile/dispatch) that a pure power law
in length doesn't capture at very short or very long sequences.

## What the two exponents mean

- **Chip exponent ≈ 0.96** (very close to 1.0): confirms the `pmap` result
  from `sharding.md` generalizes across every length we tested, throughput
  scales almost perfectly linearly with chip count, not just at the single
  length (118 residues) originally measured.
- **Length exponent ≈ -1.57**: consistent with (though not identical to)
  the super-linear cost growth found in the sequence-length sweep (8.3x
  length gives 14.6x time, i.e. time ~ length^1.28 there, vs the inverse
  relationship length^-1.57 here on throughput; same underlying
  super-linear cost, expressed as the reciprocal).

## The finding that surprised us: cost per prediction is roughly chip-count-invariant

Combining this formula with `cost_analysis.md`'s TPU pricing
($1.20/chip-hour):

![Scaling law charts](../../figures/scaling_law_chart.png)

| Length | Cost/1000 preds @ 1 chip | @ 2 chips | @ 4 chips | @ 8 chips |
|---|---|---|---|---|
| 100  | $0.114 | $0.121 | $0.128 | $0.139 |
| 250  | $0.386 | $0.394 | $0.399 | $0.413 |
| 500  | $1.004 | $1.012 | $1.019 | $1.036 |
| 1000 | $5.051 | $5.089 | $5.089 | $5.109 |

Because throughput scales at ~0.96 (just under perfectly linear) while
cost scales at exactly 1.0 (chips x $/chip-hour), **more chips barely
changes cost per prediction at all**: it's flat to within a few percent
across the whole grid, with a very slight cost *penalty* for using more
chips (the 0.04 shortfall from perfectly-linear scaling, likely
communication/dispatch overhead as chip count grows).

## Practical recommendation

**Chip count should be chosen for throughput/latency requirements, not for
cost**, since cost per prediction is nearly flat regardless of how many
chips you use. The right question isn't "how many chips is cheapest"
(they're all about the same), it's "how many chips do I need to hit my
required predictions-per-second." This is a cleaner, more actionable
version of the cost story than `cost_analysis.md` alone gave: that
analysis showed idle capacity is wasted money; this one adds that once
you're using `pmap` and chips are actually busy, adding more of them is
close to cost-neutral, so scale chip count for speed, not economics.

**Hardware:** Stanford GKE TPU v5e-8 (`tpu-v5-lite-podslice`, topology 2×4, 8 chips) via Kubernetes Job + Kueue. AF2 baseline comparison also uses Colab Intel Xeon CPU (2 vCPU) and Colab NVIDIA Tesla T4 (`results/comparison.md`).
