import { useEffect, useRef, useState } from 'react'

const DATA = [
  {
    name: 'CPU Xeon',
    seconds: 212.113,
    display: '212.1s',
    speedup: '1× (CPU baseline)',
    width: 100,
    color: '#7a8796',
  },
  {
    name: 'NVIDIA Tesla T4',
    seconds: 13.086,
    display: '13.1s',
    speedup: '16.2× vs CPU',
    width: 58,
    color: '#3d5f94',
  },
  {
    name: 'TPU v5e-8',
    seconds: 0.47,
    display: '0.47s',
    speedup: '451× vs CPU',
    width: 18,
    color: '#0b6e7a',
  },
]

export default function HeadlineResults() {
  const ref = useRef(null)
  const [on, setOn] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setOn(true)
      },
      { threshold: 0.35 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  return (
    <section id="results" className="border-t border-line">
      <div className="viewport-tight shell">
        <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
          <div className="max-w-lg">
            <p className="kicker">
              Single-chip comparison · August 2026 campaign
            </p>
            <h2 className="section-title">Steady-state latency, one chip</h2>
          </div>
          <p className="eq font-display text-[clamp(3rem,8vw,4.5rem)] font-bold leading-none tracking-[-0.04em] text-teal">
            0.47s
          </p>
        </div>

        <p className="section-note mt-4 max-w-xl">
          AlphaFold 2 <code className="font-mono text-[0.92em] text-ink">model_3</code>
          , 0 recycles, 118-residue input, Haiku random-init params. Backends:
          Google Colab Intel Xeon (2 vCPU), Google Colab NVIDIA Tesla T4,
          Stanford GKE TPU
          v5e-8 (2×4 lite podslice, 8 chips). Identical code path; bar lengths
          are log-scaled.
        </p>

        <div ref={ref} className="mt-8 space-y-5 md:mt-10">
          {DATA.map((d) => (
            <div
              key={d.name}
              className="grid items-center gap-2 sm:grid-cols-[6.5rem_1fr_auto] sm:gap-6"
            >
              <div className="flex items-baseline justify-between gap-3 sm:contents">
                <p className="font-display text-lg font-semibold text-ink">
                  {d.name}
                </p>
                <p className="eq font-display text-lg font-semibold tracking-tight text-ink sm:hidden">
                  {d.speedup}
                </p>
              </div>
              <div className="bar-track min-w-0">
                <div
                  className="bar-fill"
                  style={{
                    width: on ? `${d.width}%` : '0%',
                    background: d.color,
                    minWidth: on ? '4.25rem' : 0,
                  }}
                >
                  {d.display}
                </div>
              </div>
              <p className="eq hidden text-right font-display text-xl font-semibold tracking-tight text-ink sm:block sm:min-w-[4.5rem]">
                {d.speedup}
              </p>
            </div>
          ))}
        </div>

        <p className="section-note mt-6 max-w-2xl">
          The TPU figure is one chip of the eight in the slice. Speedups are
          relative to the August CPU baseline.
        </p>

        <div className="mt-8 max-w-2xl md:mt-10">
          <p className="kicker">Did the baselines reproduce?</p>
          <p className="section-body mt-3">
            No. Reruns five weeks later, with the same input and metrics, found
            the CPU 1.64–1.69× slower (three September sessions) and the T4 about
            2× faster (6.567 s, one September session), for reasons the records
            cannot identify. The TPU-over-GPU ratio above is therefore specific
            to the August campaign.
          </p>
        </div>

        <hr className="rule mt-10 md:mt-12" />

        <div className="mt-6 md:mt-8">
          <p className="kicker">First predict / steady-state</p>
          <p className="section-body mt-3 max-w-2xl">
            The first call includes XLA compilation. On CPU and GPU, part of the
            timed first call was profiler teardown, which we can subtract. In a
            retained trace of one TPU first call, about 76% of the traced
            apply_fn span is self time in JAX&rsquo;s tracing and compilation
            path.
          </p>
          <dl className="mt-6 grid gap-6 sm:grid-cols-3 sm:text-center">
            {[
              {
                k: 'CPU Xeon',
                v: '1.28×',
                s: '≈1.11× after removing profiler teardown',
              },
              {
                k: 'GPU NVIDIA Tesla T4',
                v: '7.46×',
                s: '≈4.25× after removing profiler teardown',
              },
              {
                k: 'TPU v5e-8',
                v: '59.1×',
                s: 'not correctable: no TPU run log survives',
              },
            ].map((row) => (
              <div key={row.k}>
                <dt className="section-note">{row.k}</dt>
                <dd className="eq mt-1 font-display text-2xl font-semibold tracking-tight text-ink">
                  {row.v}
                </dd>
                <p className="section-note mt-1 leading-snug">{row.s}</p>
              </div>
            ))}
          </dl>
        </div>

        <p className="section-note mt-10 md:mt-12">
          <a href="#af3" className="link-quiet font-medium text-ink">
            AlphaFold 3 side-investigation →
          </a>
        </p>
      </div>
    </section>
  )
}
