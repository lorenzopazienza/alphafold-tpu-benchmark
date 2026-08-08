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
      <div className="shell grid flex-1 content-center gap-12 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end lg:gap-16">
        <div className="max-w-xl">
          <p className="animate-rise kicker">Stanford HPC · Summer 2026</p>
          <h1 className="animate-rise delay-1 mt-5 font-display text-[clamp(2.75rem,7vw,4.5rem)] font-semibold leading-[0.98] tracking-[-0.03em] text-ink">
            AlphaFold
            <br />
            on TPU
          </h1>
          <p className="animate-rise delay-2 mt-6 text-[1.125rem] leading-[1.55] text-slate md:text-[1.25rem]">
            We timed the same AlphaFold JAX forward pass on CPU, GPU (T4), and
            TPU (v5e-8). At steady-state the TPU is 451× faster than CPU.
          </p>
          <p className="animate-rise delay-3 mt-8">
            <a href="#structure" className="link-quiet text-[15px] font-medium">
              Structure
            </a>
            <span className="mx-3 text-line">/</span>
            <a href="#results" className="link-quiet text-[15px] font-medium">
              Results
            </a>
          </p>
        </div>

        <div className="animate-rise delay-2 lg:pb-1 lg:text-right">
          <p className="kicker">Steady-state vs CPU</p>
          <p className="eq mt-2 font-display text-[clamp(4.75rem,22vw,10.5rem)] font-bold leading-[0.85] tracking-[-0.05em] text-ink">
            {n}
            <span className="text-teal">×</span>
          </p>
          <p className="mt-4 text-sm text-mute lg:ml-auto lg:max-w-[16rem]">
            TPU 0.47s · GPU 13.1s (16.2×) · CPU 212s
          </p>
        </div>
      </div>

      <div className="shell mt-auto pt-14">
        <hr className="rule" />
        <p className="pt-5 text-[13px] leading-relaxed text-mute">
          Lorenzo Pazienza & Ihab El Bani · Intro to HPC and AI Systems · Profs.
          Steve Jones & Mourad Bouache
        </p>
      </div>
    </section>
  )
}
