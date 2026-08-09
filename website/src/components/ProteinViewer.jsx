import { useEffect, useRef, useState } from 'react'
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceArea,
} from 'recharts'
import { CONFIDENCE, MEAN_PLDDT, plddtColor } from '../data/confidence'

function ConfidenceChart() {
  return (
    <div className="h-[min(32vh,260px)] w-full min-h-[220px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={CONFIDENCE}
          margin={{ top: 8, right: 8, left: -8, bottom: 4 }}
        >
          <ReferenceArea
            x1={70}
            x2={76}
            fill="#ff7d45"
            fillOpacity={0.1}
            strokeOpacity={0}
          />
          <XAxis
            dataKey="res"
            tick={{ fill: '#6b7684', fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: '#e4e7eb' }}
          />
          <YAxis
            domain={[40, 100]}
            tick={{ fill: '#6b7684', fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            width={36}
          />
          <Tooltip
            contentStyle={{
              borderRadius: 4,
              border: '1px solid #e4e7eb',
              fontSize: 13,
              boxShadow: 'none',
            }}
            formatter={(v) => [`${v}`, 'pLDDT']}
            labelFormatter={(r) => `Residue ${r}`}
          />
          <Line
            type="monotone"
            dataKey="plddt"
            stroke="#0b6e7a"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3.5, fill: '#0b6e7a' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function Legend() {
  const stops = [
    { label: '>90', color: '#0053d6' },
    { label: '70–90', color: '#65cbf3' },
    { label: '50–70', color: '#ffdb13' },
    { label: '<50', color: '#ff7d45' },
  ]
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-mute">
      <span className="font-semibold text-ink">pLDDT</span>
      {stops.map((s) => (
        <span key={s.label} className="inline-flex items-center gap-2">
          <span
            className="h-2.5 w-2.5 shrink-0 rounded-full"
            style={{ background: s.color }}
          />
          <span className="eq">{s.label}</span>
        </span>
      ))}
    </div>
  )
}

export default function ProteinViewer() {
  const hostRef = useRef(null)
  const viewerRef = useRef(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    let onResize

    async function boot() {
      try {
        const { createViewer } = await import('3dmol/build/3Dmol.es6.js')
        if (cancelled || !hostRef.current) return

        hostRef.current.innerHTML = ''
        const viewer = createViewer(hostRef.current, {
          backgroundColor: '#f7f8f9',
          antialias: true,
        })
        viewerRef.current = viewer

        const pdb = await fetch('/structure/ubiquitin_predicted.pdb').then((r) =>
          r.text(),
        )
        if (cancelled) return

        viewer.addModel(pdb, 'pdb')
        viewer.setStyle(
          {},
          {
            cartoon: {
              colorfunc: (atom) => {
                const raw = atom.b
                const score = raw <= 1 ? raw * 100 : raw
                return plddtColor(score)
              },
            },
          },
        )
        viewer.zoomTo()
        viewer.zoom(1.15)
        viewer.render()
        requestAnimationFrame(() => {
          viewer.resize()
          viewer.render()
        })
        onResize = () => {
          viewer.resize()
          viewer.render()
        }
        window.addEventListener('resize', onResize)
        setReady(true)
      } catch (e) {
        console.error(e)
        if (!cancelled) setError('3D viewer failed to load.')
      }
    }

    boot()
    return () => {
      cancelled = true
      if (onResize) window.removeEventListener('resize', onResize)
      if (viewerRef.current) {
        try {
          viewerRef.current.clear()
        } catch {
          /* ignore */
        }
      }
    }
  }, [])

  return (
    <section id="structure" className="border-t border-line bg-panel">
      <div className="viewport-tight shell">
        <div className="mb-6 flex flex-col gap-3 md:mb-8 md:flex-row md:items-end md:justify-between md:gap-8">
          <div className="max-w-lg">
            <p className="kicker">Structure</p>
            <h2 className="mt-3 font-display text-[clamp(1.9rem,4vw,2.75rem)] font-semibold tracking-[-0.03em] text-ink">
              Human ubiquitin
            </h2>
          </div>
          <p className="max-w-md text-base leading-relaxed text-slate md:text-right">
            ESMFold prediction · 76 residues · mean pLDDT{' '}
            <span className="eq font-medium text-ink">{MEAN_PLDDT}</span>.
            Drag to rotate, scroll to zoom. Colors follow AlphaFold’s pLDDT
            scale.
          </p>
        </div>

        <div className="grid items-stretch gap-8 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] lg:gap-10">
          <figure className="relative min-w-0">
            <div className="relative aspect-[5/4] w-full overflow-hidden bg-paper lg:aspect-square lg:max-h-[28rem]">
              <div
                ref={hostRef}
                className="absolute inset-0 h-full w-full cursor-grab active:cursor-grabbing"
                aria-label="Interactive 3D structure of human ubiquitin"
              />
              {!ready && !error && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-mute">
                  Loading PDB…
                </div>
              )}
              {error && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
                  <img
                    src="/figures/ubiquitin_structure.png"
                    alt="ESMFold cartoon of human ubiquitin colored by pLDDT: high-confidence blue/cyan core, lower-confidence C-terminal tail"
                    className="max-h-[70%]"
                  />
                  <p className="text-sm text-mute">{error}</p>
                </div>
              )}
            </div>
            <figcaption className="mt-4 flex flex-col gap-2 border-t border-line pt-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
              <Legend />
              <p className="text-sm leading-snug text-mute sm:max-w-[16rem] sm:text-right">
                Lower confidence at the C-terminus (residues ~71–76)
              </p>
            </figcaption>
          </figure>

          <figure className="flex min-w-0 flex-col justify-center border-t border-line pt-8 lg:border-l lg:border-t-0 lg:pl-10 lg:pt-0">
            <div className="flex items-baseline justify-between gap-4">
              <h3 className="font-display text-xl font-semibold tracking-tight text-ink">
                Per-residue pLDDT
              </h3>
              <p className="eq font-mono text-sm text-mute">
                mean {MEAN_PLDDT}
              </p>
            </div>
            <p className="mt-2 max-w-sm text-base leading-relaxed text-slate">
              Confidence stays high through the structured core and drops in the
              last residues, the flexible conjugating tail.
            </p>
            <div className="mt-5">
              <ConfidenceChart />
            </div>
            <p className="mt-3 text-sm text-mute">
              Shaded band: residues 70–76
            </p>
          </figure>
        </div>
      </div>
    </section>
  )
}
