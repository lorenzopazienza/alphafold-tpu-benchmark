import { useState } from 'react'
import { EXPERIMENTS, PHASES } from '../data/experiments'

/** One glyph control — no repeated “Read notes” labels on every row. */
function ExpandToggle({ open }) {
  return (
    <span
      aria-hidden
      className={`grid h-9 w-9 shrink-0 place-items-center rounded-md border text-lg leading-none transition-colors ${
        open
          ? 'border-ink bg-ink text-paper'
          : 'border-line bg-panel text-ink group-hover:border-teal group-hover:text-teal'
      }`}
    >
      {open ? '−' : '+'}
    </span>
  )
}

function PhaseFigure({ src, caption }) {
  if (!src) return null
  return (
    <figure className="border-t border-line bg-paper/60 px-3.5 py-4 md:px-8 md:py-5 lg:px-10">
      <img
        src={src}
        alt={
          caption ||
          'Experiment figure from the AlphaFold TPU Benchmark scaling study'
        }
        className="w-full border border-line bg-panel"
        loading="lazy"
      />
      {caption && (
        <figcaption className="mt-2 text-[12px] leading-snug text-mute">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}

/** Compact mobile row */
function MobileRow({ exp, index, open, onToggle }) {
  const negative = exp.tone === 'negative'
  const n = String(index + 1).padStart(2, '0')

  return (
    <article
      className={`border-b border-line last:border-b-0 ${
        negative ? 'bg-amber-soft/50' : ''
      }`}
    >
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        aria-label={`${open ? 'Close' : 'More about'} ${exp.title}`}
        className="group flex w-full cursor-pointer items-center gap-3 px-3.5 py-3.5 text-left active:bg-paper/80"
      >
        <span className="eq w-6 shrink-0 font-mono text-[11px] text-mute">
          {n}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="truncate font-display text-[15px] font-semibold tracking-tight text-ink">
              {exp.title}
            </h3>
            {negative && (
              <span className="shrink-0 font-mono text-[9px] font-medium uppercase tracking-[0.1em] text-amber">
                Neg
              </span>
            )}
          </div>
          <p className="mt-0.5 truncate text-[12px] text-mute">{exp.statLabel}</p>
        </div>
        <span
          className={`eq shrink-0 font-display text-[1.05rem] font-bold tracking-tight ${
            negative ? 'text-amber' : 'text-ink'
          }`}
        >
          {exp.stat}
        </span>
        <ExpandToggle open={open} />
      </button>

      <div
        className={`grid transition-[grid-template-rows] duration-250 ease-out ${
          open ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'
        }`}
      >
        <div className="overflow-hidden">
          <div className="space-y-3 px-3.5 pb-4 pl-[2.65rem]">
            <p className="text-[14px] leading-relaxed text-slate">{exp.finding}</p>
            <p className="text-[13px] leading-relaxed text-mute">{exp.body}</p>
            {exp.chart && (
              <figure>
                <img
                  src={exp.chart}
                  alt={exp.chartCaption || exp.title}
                  className="w-full border border-line"
                  loading="lazy"
                />
                {exp.chartCaption && (
                  <figcaption className="mt-1.5 text-[11px] text-mute">
                    {exp.chartCaption}
                  </figcaption>
                )}
              </figure>
            )}
          </div>
        </div>
      </div>
    </article>
  )
}

/** Desktop row: chevron = expand control; per-row chart only if unique */
function DesktopRow({ exp, open, onToggle }) {
  const negative = exp.tone === 'negative'

  return (
    <article
      className={`grid grid-cols-[4px_minmax(0,1fr)] border-t border-line ${
        negative ? 'bg-amber-soft/45' : 'bg-transparent'
      }`}
    >
      <div aria-hidden className={negative ? 'bg-amber' : 'bg-transparent'} />

      <div className="min-w-0">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          aria-label={`${open ? 'Close' : 'More about'} ${exp.title}`}
          className="group grid w-full cursor-pointer grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-5 px-6 py-5 text-left transition-colors hover:bg-paper/80 md:gap-6 md:px-8 md:py-6 lg:gap-8 lg:px-10 lg:py-7"
        >
          <div className="min-w-0">
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <h3 className="font-display text-xl font-semibold tracking-tight text-ink">
                {exp.title}
              </h3>
              {negative && (
                <span className="font-mono text-[10px] font-medium uppercase tracking-[0.12em] text-amber">
                  Negative
                </span>
              )}
            </div>
            <p className="mt-2 text-base leading-relaxed text-slate">
              {exp.finding}
            </p>
          </div>

          <div className="shrink-0 text-right">
            <p
              className={`eq font-display text-[1.75rem] font-bold leading-none tracking-tight lg:text-[2rem] ${
                negative ? 'text-amber' : 'text-ink'
              }`}
            >
              {exp.stat}
            </p>
            <p className="mt-1.5 max-w-[9rem] text-sm leading-snug text-mute lg:max-w-[10.5rem]">
              {exp.statLabel}
            </p>
          </div>

          <ExpandToggle open={open} />
        </button>

        <div
          className={`grid transition-[grid-template-rows] duration-300 ease-out ${
            open ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'
          }`}
        >
          <div className="overflow-hidden">
            <div className="border-t border-line/70 px-8 pb-6 pt-4 lg:px-10">
              <p className="max-w-3xl text-[15px] leading-relaxed text-slate">
                {exp.body}
              </p>
              {exp.chart && (
                <figure className="mt-4">
                  <img
                    src={exp.chart}
                    alt={exp.chartCaption || exp.title}
                    className="w-full border border-line bg-panel"
                    loading="lazy"
                  />
                  {exp.chartCaption && (
                    <figcaption className="mt-2 text-[12px] text-mute">
                      {exp.chartCaption}
                    </figcaption>
                  )}
                </figure>
              )}
            </div>
          </div>
        </div>
      </div>
    </article>
  )
}

function PhaseBlock({ phase, openIds, onToggle }) {
  const items = EXPERIMENTS.filter((e) => e.phase === phase.id)

  return (
    <div>
      <p className="mb-2 font-mono text-[10px] font-medium uppercase tracking-[0.14em] text-mute md:mb-3">
        {phase.label}
      </p>

      {/* Mobile list */}
      <div className="overflow-hidden rounded-lg border border-line bg-panel md:hidden">
        {items.map((exp) => (
          <MobileRow
            key={exp.id}
            exp={exp}
            index={EXPERIMENTS.findIndex((e) => e.id === exp.id)}
            open={openIds.has(exp.id)}
            onToggle={() => onToggle(exp.id)}
          />
        ))}
        {phase.figure && (
          <PhaseFigure src={phase.figure} caption={phase.figureCaption} />
        )}
      </div>

      {/* Desktop list */}
      <div className="hidden overflow-hidden rounded-xl border border-line md:block">
        {items.map((exp) => (
          <DesktopRow
            key={exp.id}
            exp={exp}
            open={openIds.has(exp.id)}
            onToggle={() => onToggle(exp.id)}
          />
        ))}
        {phase.figure && (
          <PhaseFigure src={phase.figure} caption={phase.figureCaption} />
        )}
      </div>
    </div>
  )
}

export default function DeepDive() {
  const [openIds, setOpenIds] = useState(() => new Set())

  const toggle = (id) => {
    setOpenIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <section id="experiments" className="border-t border-line bg-panel">
      <div className="shell section-y">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between md:gap-6">
          <div className="max-w-xl">
            <p className="kicker">Experiments</p>
            <h2 className="mt-3 font-display text-[clamp(1.9rem,4vw,2.75rem)] font-semibold tracking-[-0.03em] text-ink">
              Twelve TPU follow-ups
            </h2>
          </div>
          <p className="max-w-sm text-sm leading-relaxed text-mute md:text-right">
            Click <span className="font-medium text-ink">+</span> on a row for
            method notes. Shared figures sit under each phase. Amber rail =
            negative result.
          </p>
        </div>

        <div className="mt-6 space-y-7 md:mt-8 md:space-y-8">
          {PHASES.map((phase) => (
            <PhaseBlock
              key={phase.id}
              phase={phase}
              openIds={openIds}
              onToggle={toggle}
            />
          ))}
        </div>
      </div>
    </section>
  )
}
