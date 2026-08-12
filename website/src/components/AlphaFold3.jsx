import { useEffect, useRef, useState } from 'react'

const ROWS = [
  {
    backend: 'CPU (Colab)',
    af2: '212.1s',
    af3: '490.8s',
    ratio: '2.3×',
    af2Pct: 43,
    af3Pct: 100,
  },
  {
    backend: 'GPU T4 (Colab)',
    af2: '13.1s',
    af3: '22.8s',
    ratio: '1.74×',
    af2Pct: 57,
    af3Pct: 100,
  },
]

export default function AlphaFold3() {
  const ref = useRef(null)
  const [on, setOn] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setOn(true)
      },
      { threshold: 0.25 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  return (
    <section id="af3" className="border-t border-line">
      <div className="shell section-y">
        <div className="max-w-2xl">
          <p className="kicker">Side investigation</p>
          <h2 className="mt-3 font-display text-[clamp(1.9rem,4vw,2.75rem)] font-semibold tracking-[-0.03em] text-ink">
            AlphaFold 3 on the same backends
          </h2>
          <p className="mt-4 max-w-xl text-base leading-relaxed text-slate">
            Separate DeepMind codebase — Pairformer + diffusion, not an AlphaFold
            2 version bump. Same 118-residue toy input and seed=1. Trained
            weights (~1.15 GB). Five diffusion samples per call.
          </p>
        </div>

        <div className="mt-10 grid items-start gap-10 lg:mt-12 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)] lg:gap-14">
          <figure className="min-w-0">
            <div className="overflow-hidden bg-panel">
              <img
                src="/figures/af3_toy_test_structure.jpg"
                alt="AlphaFold 3 predicted structure for the 118-residue toy sequence"
                className="block h-auto w-full"
                loading="lazy"
              />
            </div>
            <figcaption className="mt-3 text-sm leading-snug text-mute">
              AF3 toy-test structure · trained weights · seed=1
            </figcaption>
          </figure>

          <div ref={ref} className="min-w-0">
            <p className="kicker">Same-hardware · per sample</p>
            <p className="mt-2 text-sm leading-relaxed text-mute">
              Identical Colab CPU and T4 runs. AF3 is slower; it also gains more
              from the GPU (21.5× CPU→GPU vs AF2’s 16.2×).
            </p>

            <div className="mt-8 space-y-8">
              {ROWS.map((row) => (
                <div key={row.backend}>
                  <div className="flex items-baseline justify-between gap-4">
                    <p className="font-display text-lg font-semibold text-ink">
                      {row.backend}
                    </p>
                    <p className="eq font-mono text-sm text-mute">
                      AF3 {row.ratio} AF2
                    </p>
                  </div>
                  <div className="mt-3 space-y-2.5">
                    <div className="grid grid-cols-[3.25rem_1fr_auto] items-center gap-3">
                      <span className="font-mono text-[11px] uppercase tracking-wide text-mute">
                        AF2
                      </span>
                      <div className="bar-track min-w-0">
                        <div
                          className="bar-fill"
                          style={{
                            width: on ? `${row.af2Pct}%` : '0%',
                            background: '#3d5f94',
                            minWidth: on ? '3.5rem' : 0,
                          }}
                        >
                          {row.af2}
                        </div>
                      </div>
                      <span className="eq hidden text-sm text-mute sm:inline">
                        /sample
                      </span>
                    </div>
                    <div className="grid grid-cols-[3.25rem_1fr_auto] items-center gap-3">
                      <span className="font-mono text-[11px] uppercase tracking-wide text-mute">
                        AF3
                      </span>
                      <div className="bar-track min-w-0">
                        <div
                          className="bar-fill"
                          style={{
                            width: on ? `${row.af3Pct}%` : '0%',
                            background: '#0b6e7a',
                            minWidth: on ? '3.5rem' : 0,
                          }}
                        >
                          {row.af3}
                        </div>
                      </div>
                      <span className="eq hidden text-sm text-mute sm:inline">
                        /sample
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div
        className={`border-y border-amber/25 bg-amber-soft transition-opacity duration-700 ${
          on ? 'opacity-100' : 'opacity-90'
        }`}
      >
        <div className="shell py-10 md:py-12">
          <p className="kicker text-amber">Negative result · TPU</p>
          <h3 className="mt-3 max-w-2xl font-display text-[clamp(1.35rem,3vw,1.85rem)] font-semibold tracking-[-0.02em] text-ink">
            Every infrastructure step worked. The CLI has no{' '}
            <span className="eq font-mono text-amber">tpu</span>.
          </h3>
          <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-slate">
            Native C++ build, Chemical Component Dictionary, weights, and{' '}
            <code className="font-mono text-[0.92em] text-ink">jax[tpu]</code>{' '}
            all succeeded on Stanford’s v5e-8 Job. Inference died at flag
            validation: AlphaFold 3’s public{' '}
            <code className="font-mono text-[0.92em] text-ink">--jax_backend</code>{' '}
            only accepts <span className="eq font-mono">cpu | gpu | mps</span>.
            Matches DeepMind’s docs (NVIDIA GPU ≥7.0 or CPU). Not an infra miss
            on our side.
          </p>
          <pre className="mt-6 max-w-3xl overflow-x-auto bg-ink px-4 py-3 font-mono text-[11.5px] leading-relaxed text-white/85 sm:text-[12.5px]">
            {`FATAL Flags parsing error: flag --jax_backend=tpu:
value should be one of <cpu|gpu|mps>`}
          </pre>
        </div>
      </div>
    </section>
  )
}
