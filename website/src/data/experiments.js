/**
 * Voice target: lab notes you'd send a teammate.
 * Fact → measurement → implication. No slogans.
 *
 * Phase figures (e.g. scaling_charts.png) are shared across rows in that
 * phase so we never paste the same image under two experiments.
 */
export const PHASES = [
  {
    id: 'scaling',
    label: 'Scaling',
    figure: '/figures/scaling_charts.png',
    figureCaption:
      'Left: sequence length. Right: recycle depth. Same TPU v5e-8 runs.',
  },
  { id: 'bottleneck', label: 'Bottleneck & fix' },
  { id: 'multichip', label: 'Multi-chip' },
  { id: 'fit', label: 'Fit & rigor' },
]

export const EXPERIMENTS = [
  {
    id: 'seq-length',
    phase: 'scaling',
    title: 'Sequence length',
    finding:
      'Steady-state time grew faster than sequence length (8.3× length → 14.6× time).',
    stat: '14.6×',
    statLabel: 'time for 8.3× length',
    tone: 'default',
    body: 'We swept 60, 120, 250, and 500 residues. The super-linear curve matches AlphaFold’s pair representation (roughly quadratic to cubic). Compile time stayed about flat across those lengths. See the left panel in the scaling figure below.',
  },
  {
    id: 'recycles',
    phase: 'scaling',
    title: 'Recycle depth',
    finding:
      'Steady-state rose linearly with recycles; compile time did not.',
    stat: '~0.46s',
    statLabel: 'per extra recycle',
    tone: 'default',
    body: 'At 0, 1, and 3 recycles, each added recycle cost about 0.46s in steady-state. Compile time jumped from 0 to 1 recycle, then barely changed from 1 to 3, which suggests XLA kept the loop as a reusable structure. See the right panel in the scaling figure below.',
  },
  {
    id: 'chip-vis',
    phase: 'scaling',
    title: 'Chip visibility',
    finding: '1 visible chip and 8 visible chips gave the same single-query time.',
    stat: '1 of 8',
    statLabel: 'chips used',
    tone: 'default',
    body: 'The default AlphaFold path runs one query on one chip. Trying 2- and 4-chip subsets through undocumented libtpu flags failed (one crash took down the node controller). Those attempts are logged in the repo notes.',
  },
  {
    id: 'trace',
    phase: 'bottleneck',
    title: 'Profiler trace',
    finding:
      'cache_miss in JAX pjit accounted for ~76% of the first predict call.',
    stat: '76%',
    statLabel: 'of first call in cache_miss',
    tone: 'default',
    body: 'On the captured XLA trace, $pjit.py:250 cache_miss had 12.55s self-time inside a 16.56s first predict. The TPU device track was nearly idle for that interval, so the cost was host-side JIT work. That is why we tested a persistent compilation cache next.',
  },
  {
    id: 'cache',
    phase: 'bottleneck',
    title: 'Compilation cache',
    finding:
      'Persisting the JAX compile cache across a process restart cut init_params by 6.8×.',
    stat: '6.8×',
    statLabel: 'faster init_params',
    tone: 'default',
    body: 'First predict also improved 1.9× with a warm cache. Steady-state was unchanged. This matches the course vLLM setup: pay XLA compile once, reuse the artifact.',
  },
  {
    id: 'precision',
    phase: 'bottleneck',
    title: 'Precision (bf16)',
    finding:
      'bfloat16 cut steady-state HBM ~40% with little runtime change at this size.',
    stat: '−40%',
    statLabel: 'steady-state HBM',
    tone: 'default',
    body: 'Compared with float32, wall-clock barely moved for our problem size. Peak memory rose briefly during the cast, then settled lower.',
  },
  {
    id: 'batching',
    phase: 'multichip',
    title: 'Multi-query batching',
    finding:
      'jax.vmap reduced proteins/sec as batch size grew; only TPU_0 used HBM.',
    stat: 'lower',
    statLabel: 'proteins/sec vs batch=1',
    tone: 'negative',
    chart: '/figures/batching_chart.png',
    chartCaption: 'Wall-clock scales ~linearly; per-protein cost rises with batch size.',
    body: 'At batch sizes 1, 2, 4, and 8, throughput never beat batch=1 and per-protein cost rose. Memory stayed on one chip. vmap stacks work inside a single compiled program; it does not place work on the other seven chips.',
  },
  {
    id: 'pmap',
    phase: 'multichip',
    title: 'pmap parallelism',
    finding:
      'jax.pmap ran 8 proteins on 8 chips at 6.92× the single-chip throughput.',
    stat: '6.92×',
    statLabel: 'throughput speedup',
    tone: 'default',
    body: 'Throughput went from 2.13 to 14.72 proteins/sec. Per-chip HBM was 445–469 MB, close to the single-protein footprint, which matches eight independent runs rather than one replicated copy.',
  },
  {
    id: 'autoshard',
    phase: 'multichip',
    title: 'Auto-sharding (GSPMD)',
    finding:
      'GSPMD auto-mesh left a full ~463 MB copy on every chip (no split).',
    stat: '0×',
    statLabel: 'tensor split',
    tone: 'negative',
    body: 'We reused the auto-mesh pattern from the course Tunix lab on a single protein. Reproduced twice. Without sharding annotations in AlphaFold, the partitioner replicated instead of splitting.',
  },
  {
    id: 'scaling-law',
    phase: 'fit',
    title: 'Empirical scaling law',
    finding:
      'A power law on 16 TPU runs (4 chip counts × 4 lengths) fits with R² 0.981.',
    stat: 'R² 0.981',
    statLabel: '16-point TPU grid',
    tone: 'default',
    chart: '/figures/scaling_law_chart.png',
    chartCaption: 'Throughput vs chips and sequence length on the measured grid.',
    body: 'throughput ≈ 4527.77 · chips^0.963 · length^−1.572. The chip exponent near 1.0 says pmap’s scaling holds across the lengths we tested. Combined with list prices, cost per prediction stays nearly flat in chip count.',
  },
  {
    id: 'rigor',
    phase: 'fit',
    title: 'Model pick & repeats',
    finding:
      'model_5 was ~15% faster at steady-state; three repeats stayed under 1% stdev.',
    stat: '<1%',
    statLabel: 'stdev on repeats',
    tone: 'default',
    body: 'model_5 beat model_3 and model_4 by about 15% in steady-state. Across three repeats, every metric we tracked had under 1% standard deviation.',
  },
]
