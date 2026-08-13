import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import {
  AF3_CONFIDENCE,
  AF3_MEAN_CONFIDENCE,
} from '../data/af3-confidence'

const ProteinViewer = lazy(() => import('./ProteinViewer'))

const ROWS = [
  {
    backend: 'CPU Xeon (Google Colab, 2 vCPU)',
    af2: '212.1s',
    af3: '490.8s',
    ratio: '2.3×',
    af2Pct: 43,
    af3Pct: 100,
  },
  {
    backend: 'GPU NVIDIA Tesla T4 (Google Colab)',
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
      <div className="shell section-y !pb-0">
        <div className="max-w-2xl">
          <p className="kicker">Side investigation</p>
          <h2 className="section-title">AlphaFold 3 on the same backends</h2>
          <p className="section-lede">
            Separate DeepMind codebase (Pairformer + diffusion), not an
            AlphaFold 2 version bump. Same 118-residue toy input and seed=1 on
            Google Colab Intel Xeon CPU (2 vCPU) and Google Colab NVIDIA Tesla
            T4. Trained
            weights (~1.15 GB). Five diffusion samples per call.
          </p>
        </div>

        <div ref={ref} className="mt-10 max-w-2xl lg:mt-12">
          <p className="kicker">Same-hardware · per sample</p>
            <p className="section-note mt-2">
              Identical Google Colab Intel Xeon CPU and NVIDIA Tesla T4 runs. AF3
              is slower; it also
              gains more from the GPU (21.5× CPU→GPU vs AF2’s 16.2×).
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
                    <span className="label-mono">AF2</span>
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
                    <span className="label-mono">AF3</span>
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

      <Suspense
        fallback={
          <div className="border-t border-line bg-panel py-16 text-center text-sm text-mute">
            Loading AF3 structure…
          </div>
        }
      >
        <ProteinViewer
          embedded
          structureUrl="/structure/af3_toy_test_cpu-colab_model.cif"
          format="cif"
          kicker="AlphaFold 3 · structure"
          title="AF3 toy structure"
          subtitle="Diffusion sample, seed 1 · 118 residues, no MSA"
          meanConfidence={(AF3_MEAN_CONFIDENCE / 100).toFixed(2)}
          confidenceLabel="ptm"
          confidenceData={AF3_CONFIDENCE}
          fallbackImage="/figures/af3_toy_test_structure.png"
          fallbackAlt="AlphaFold 3 toy-sequence structure"
          captionNote="Low confidence throughout: expected for a no-MSA input, not a defect. Not comparable to the ubiquitin mean pLDDT above."
          chartTitle="Per-residue confidence"
          chartBlurb="CA B-factors stay in the orange/yellow band across the chain. That is the expected no-MSA regime, not a failed render."
          shadeRange={null}
          bandNote={null}
        />
      </Suspense>

      <div
        className={`border-y border-amber/25 bg-amber-soft transition-opacity duration-700 ${
          on ? 'opacity-100' : 'opacity-90'
        }`}
      >
        <div className="shell py-10 md:py-12">
          <p className="kicker text-amber">Negative result · TPU</p>
          <h3 className="section-subhead max-w-2xl">
            Every infrastructure step worked. The CLI has no{' '}
            <span className="eq font-mono text-amber">tpu</span>.
          </h3>
          <p className="section-body mt-4 max-w-2xl">
            Native C++ build, Chemical Component Dictionary, weights, and{' '}
            <code className="font-mono text-[0.92em] text-ink">jax[tpu]</code>{' '}
            all succeeded on Stanford’s GKE TPU v5e-8 (2×4 lite podslice) Job.
            Inference died at flag validation: AlphaFold 3’s public{' '}
            <code className="font-mono text-[0.92em] text-ink">--jax_backend</code>{' '}
            only accepts <span className="eq font-mono">cpu | gpu | mps</span>.
            Matches DeepMind’s docs (NVIDIA GPU ≥7.0 or CPU). Not an infra miss
            on our side.
          </p>
          <pre className="mt-6 max-w-3xl overflow-x-auto bg-ink px-4 py-3 font-mono text-[0.75rem] leading-relaxed text-white/85 sm:text-[0.8125rem]">
            {`FATAL Flags parsing error: flag --jax_backend=tpu:
value should be one of <cpu|gpu|mps>`}
          </pre>
        </div>
      </div>
    </section>
  )
}
