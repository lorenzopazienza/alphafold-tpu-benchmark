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

export default function Hero() {
  const n = useCountUp(451)

  return (
    <section id="top" className="viewport relative">
      <div className="shell flex flex-1 flex-col justify-center">
        <div className="flex flex-col gap-10 lg:flex-row lg:items-end lg:justify-between lg:gap-12">
          <div className="min-w-0 max-w-md">
            <p className="animate-rise kicker">Stanford HPC · Summer 2026</p>
            <h1 className="animate-rise delay-1 mt-4 font-display text-[clamp(2.85rem,6vw,4.35rem)] font-semibold leading-[0.96] tracking-[-0.035em] text-ink">
              AlphaFold
              <br />
              TPU Benchmark
            </h1>
            <p className="animate-rise delay-2 mt-5 text-[1.0625rem] leading-[1.55] text-slate md:text-[1.125rem]">
              AlphaFold predicts protein structure from sequence. We timed that
              real JAX forward pass on CPU, GPU (T4), and TPU (v5e-8) — and
              found where the time and idle chips go. Steady-state TPU is 451×
              faster than CPU.
            </p>
            <p className="animate-rise delay-3 mt-7">
              <a href="#structure" className="link-quiet text-base font-medium">
                Structure
              </a>
              <span className="mx-3 text-line">/</span>
              <a href="#results" className="link-quiet text-base font-medium">
                Results
              </a>
              <span className="mx-3 text-line">/</span>
              <a href="#reproduce" className="link-quiet text-base font-medium">
                Reproduce
              </a>
            </p>
          </div>

          <div className="animate-rise delay-2 shrink-0 lg:text-right">
            <p className="kicker">Steady-state vs CPU</p>
            <p className="eq mt-1 font-display text-[clamp(5rem,14vw,8.5rem)] font-bold leading-[0.84] tracking-[-0.05em] text-ink">
              {n}
              <span className="text-teal">×</span>
            </p>
            <p className="mt-3 text-sm text-mute lg:ml-auto lg:max-w-[16rem]">
              TPU 0.47s · GPU 13.1s (16.2×) · CPU 212s
            </p>
          </div>
        </div>
      </div>

      <div className="shell shrink-0 pt-8">
        <hr className="rule" />
        <p className="pt-4 text-[13px] leading-relaxed text-mute">
          Lorenzo Pazienza & Ihab El Bani · Intro to HPC and AI Systems · Profs.
          Steve Jones & Mourad Bouache
        </p>
      </div>
    </section>
  )
}
