import { useEffect, useRef, useState } from 'react'

const DATA = [
  {
    name: 'CPU',
    seconds: 212.113,
    display: '212.1s',
    speedup: '1×',
    width: 100,
    color: '#7a8796',
  },
  {
    name: 'GPU T4',
    seconds: 13.086,
    display: '13.1s',
    speedup: '16.2×',
    width: 58,
    color: '#3d5f94',
  },
  {
    name: 'TPU v5e',
    seconds: 0.47,
    display: '0.47s',
    speedup: '451×',
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
            <p className="kicker">AlphaFold 2 · performance delta</p>
            <h2 className="section-title">Steady-state predict latency</h2>
          </div>
          <p className="eq font-display text-[clamp(3rem,8vw,4.5rem)] font-bold leading-none tracking-[-0.04em] text-teal">
            0.47s
          </p>
        </div>

        <p className="section-note mt-4 max-w-xl">
          AlphaFold 2 · model_3, 0 recycles, 118-residue input, identical code
          path. Bar lengths are log-scaled.
        </p>

        <div ref={ref} className="mt-8 space-y-5 md:mt-10">
          {DATA.map((d) => (
            <div
              key={d.name}
              className="grid items-center gap-2 sm:grid-cols-[5.5rem_1fr_auto] sm:gap-6"
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

        <hr className="rule mt-10 md:mt-12" />

        <div className="mt-6 md:mt-8">
          <p className="kicker">First predict / steady-state</p>
          <p className="section-body mt-3 max-w-2xl">
            First predict includes XLA compile. The gap to the second predict
            shows how compile-bound each backend is.
          </p>
          <dl className="mt-6 grid gap-6 sm:grid-cols-3 sm:text-center">
            {[
              { k: 'CPU', v: '1.28×', s: 'Mostly compute' },
              { k: 'GPU T4', v: '7.46×', s: 'Compile is visible' },
              { k: 'TPU v5e', v: '59.1×', s: 'Cold path is compile' },
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
