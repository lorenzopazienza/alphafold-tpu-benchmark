import { useEffect, useState } from 'react'

function useCountUp(target, duration = 1400) {
  const [value, setValue] = useState(0)

  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduce) {
      setValue(target)
      return
    }
    let frame
    const start = performance.now()
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration)
      const eased = 1 - (1 - t) ** 3
      setValue(Math.round(target * eased))
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [target, duration])

  return value
}

const METRIC_CONTEXT = '2nd predict · model_3 · 118 residues'

const METRIC_ROWS = [
  {
    backend: 'Google Colab Intel Xeon CPU (2 vCPU)',
    detail: 'Google Colab free-tier · 13.6 GB RAM',
    time: '212.1s',
    speedups: ['1× baseline'],
    color: '#7a8796',
  },
  {
    backend: 'Google Colab NVIDIA Tesla T4',
    detail: 'Google Colab free-tier · 15 GB VRAM',
    time: '13.1s',
    speedups: ['16.2× vs Google Colab Intel Xeon CPU (2 vCPU)'],
    color: '#3d5f94',
  },
  {
    backend: 'Stanford GKE TPU v5e-8',
    detail: 'tpu-v5-lite-podslice · 2×4 · 8 chips',
    time: '0.47s',
    speedups: [
      '451× vs Google Colab Intel Xeon CPU (2 vCPU)',
      '27.8× vs Google Colab NVIDIA Tesla T4',
    ],
    color: '#0b6e7a',
  },
]

export default function Hero() {
  const n = useCountUp(451)

  return (
    <section id="top" className="viewport relative">
      <div className="shell flex flex-1 flex-col justify-center">
        <div className="flex flex-col gap-10 lg:flex-row lg:items-end lg:justify-between lg:gap-12">
          <div className="min-w-0 max-w-md">
            <p className="animate-rise kicker">
              Stanford University · Summer Session 2026 · Introduction to High
              Performance Computing and AI Systems (ME344)
            </p>
            <h1 className="animate-rise delay-1 mt-4 font-display text-[clamp(2.85rem,6vw,4.35rem)] font-semibold leading-[0.96] tracking-[-0.035em] text-ink">
              AlphaFold
              <br />
              TPU Benchmark
            </h1>
            <p className="animate-rise delay-2 section-lede !mt-5">
              We timed AlphaFold 2’s real JAX forward pass on Google Colab Intel
              Xeon CPU (2 vCPU), Google Colab NVIDIA Tesla T4, and Stanford GKE
              TPU v5e-8 (2×4 lite podslice), tracking where time and idle chips
              go. As a follow-up we also ran AlphaFold 3 on the same Google Colab
              Intel Xeon CPU (2 vCPU) and Google Colab NVIDIA Tesla T4; its public
              release has no TPU. Steady-state AF2 on Stanford GKE TPU v5e-8 is
              451× faster than Google Colab Intel Xeon CPU (0.47s vs 212.1s).
            </p>
            <p className="animate-rise delay-3 mt-7">
              <a href="#results" className="link-quiet text-base font-medium">
                AF2 results
              </a>
              <span className="mx-3 text-line">/</span>
              <a href="#af3" className="link-quiet text-base font-medium">
                AF3
              </a>
              <span className="mx-3 text-line">/</span>
              <a
                href="/presentation/AlphaFold_on_Google_TPUs_Pazienza_Lorenzo_Ihab_El_Bani.pdf"
                download
                className="link-quiet text-base font-medium"
              >
                Slides PDF
              </a>
              <span className="mx-3 text-line">/</span>
              <a href="#reproduce" className="link-quiet text-base font-medium">
                Reproduce
              </a>
            </p>
          </div>

          <div className="animate-rise delay-2 shrink-0 lg:text-right">
            <p className="kicker">
              AF2 steady-state predict vs Google Colab Intel Xeon CPU (2 vCPU)
            </p>
            <p className="eq mt-1 font-display text-[clamp(5rem,14vw,8.5rem)] font-bold leading-[0.84] tracking-[-0.05em] text-ink">
              {n}
              <span className="text-teal">×</span>
            </p>

            <div className="hero-metric-stack mt-4 lg:ml-auto">
              <p className="label-mono">{METRIC_CONTEXT}</p>
              <ul className="hero-metric-list">
                {METRIC_ROWS.map((row) => (
                  <li key={row.backend} className="hero-metric-row">
                    <span
                      className="hero-metric-dot"
                      style={{ background: row.color }}
                      aria-hidden
                    />
                    <div className="hero-metric-copy">
                      <span className="hero-metric-backend">{row.backend}</span>
                      {row.detail ? (
                        <span className="hero-metric-detail">{row.detail}</span>
                      ) : null}
                    </div>
                    <div className="hero-metric-stats">
                      <span className="hero-metric-time eq">{row.time}</span>
                      {row.speedups.map((line) => (
                        <span
                          key={line}
                          className={`hero-metric-speedup eq${
                            line.includes('451×') ? ' hero-metric-speedup--highlight' : ''
                          }`}
                        >
                          {line}
                        </span>
                      ))}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="shell shrink-0 pt-8">
        <hr className="rule" />
        <p className="section-note pt-4">
          Authors: Lorenzo Pazienza & Ihab El Bani · Stanford University · Summer
          Session 2026 · Introduction to High Performance Computing and AI
          Systems (ME344) · Profs. Steve Jones & Mourad Bouache
        </p>
      </div>
    </section>
  )
}
