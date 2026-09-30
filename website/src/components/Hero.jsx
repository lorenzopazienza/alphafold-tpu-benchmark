const ARXIV_ABS = 'https://arxiv.org/abs/2609.34818'
const ARXIV_PDF = 'https://arxiv.org/pdf/2609.34818'
const REPO = 'https://github.com/lorenzopazienza/alphafold-tpu-benchmark'

const SOFTWARE_ROWS = [
  {
    label: 'Default path (jax.jit)',
    value: '1 of 8 chips used',
    color: '#7a8796',
  },
  {
    label: 'Batching with jax.vmap',
    value: 'never above single-query throughput',
    color: '#3d5f94',
  },
  {
    label: 'Mapping with jax.pmap',
    value: '6.53–7.91× the throughput of one chip',
    color: '#0b6e7a',
  },
  {
    label: 'Automatic sharding',
    value: 'per-chip footprint unchanged (463 MB)',
    color: '#7a8796',
  },
]

export default function Hero() {
  return (
    <section id="top" className="viewport relative">
      <div className="shell flex flex-1 flex-col justify-center">
        <div className="flex flex-col gap-10 lg:flex-row lg:items-end lg:justify-between lg:gap-12">
          <div className="min-w-0 max-w-xl">
            <p className="animate-rise kicker">
              arXiv:2609.34818 · cs.DC · September 2026
            </p>
            <h1 className="animate-rise delay-1 mt-4 font-display text-[clamp(2.85rem,6vw,4.35rem)] font-semibold leading-[0.96] tracking-[-0.035em] text-ink">
              Accelerator Choice
              <br />
              Is Not Enough
            </h1>
            <p className="animate-rise delay-1 section-lede !mt-4 text-[1.1875rem] text-ink sm:text-[1.3125rem]">
              AlphaFold2 Inference on Cloud TPUs
            </p>
            <p className="animate-rise delay-2 section-lede !mt-5">
              AlphaFold2 runs the same JAX code on CPUs, GPUs and Google Cloud
              TPUs, so the accelerator looks like the main decision a user has to
              make. We show that it is not. On an eight-chip TPU v5e slice, the
              default execution path uses one chip, and how the work is expressed
              in JAX decides how much of the hardware a user actually gets.
            </p>
            <p className="animate-rise delay-3 mt-7">
              <a
                href={ARXIV_ABS}
                target="_blank"
                rel="noreferrer"
                className="link-quiet text-base font-medium"
              >
                Paper (arXiv)
              </a>
              <span className="mx-3 text-line">/</span>
              <a
                href={ARXIV_PDF}
                target="_blank"
                rel="noreferrer"
                className="link-quiet text-base font-medium"
              >
                PDF
              </a>
              <span className="mx-3 text-line">/</span>
              <a
                href={REPO}
                target="_blank"
                rel="noreferrer"
                className="link-quiet text-base font-medium"
              >
                Code
              </a>
              <span className="mx-3 text-line">/</span>
              <a href="#paper" className="link-quiet text-base font-medium">
                Cite
              </a>
            </p>
          </div>

          <div className="animate-rise delay-2 min-w-0 shrink-0 lg:text-right">
            <p className="kicker">TPU v5e-8, default execution path</p>
            <p className="eq mt-1 font-display text-[clamp(4rem,11vw,7rem)] font-bold leading-[0.84] tracking-[-0.05em] text-ink">
              1 <span className="text-teal">/ 8</span>
            </p>
            <p className="label-mono mt-2">chips hold data</p>

            <div className="hero-metric-stack mt-4 lg:ml-auto">
              <ul className="hero-metric-list">
                {SOFTWARE_ROWS.map((row) => (
                  <li key={row.label} className="hero-metric-row">
                    <span
                      className="hero-metric-dot"
                      style={{ background: row.color }}
                      aria-hidden
                    />
                    <div className="hero-metric-copy">
                      <span className="hero-metric-backend">{row.label}</span>
                    </div>
                    <div className="hero-metric-stats">
                      <span className="hero-metric-speedup eq">
                        {row.value}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
              <p className="label-mono mt-3">
                Single chip, steady state: 0.47 s per call vs 13.1 s on an NVIDIA
                T4 (August 2026 campaign)
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="shell shrink-0 pt-8">
        <hr className="rule" />
        <p className="section-note pt-4">
          Lorenzo Pazienza (LUISS Guido Carli University) &amp; Ihab El Bani (Al
          Akhawayn University) · Work carried out in Stanford&rsquo;s ME344, High
          Performance Computing and AI Systems, Summer Session 2026
        </p>
      </div>
    </section>
  )
}
