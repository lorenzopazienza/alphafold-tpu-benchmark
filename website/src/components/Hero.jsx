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
            <p className="animate-rise delay-2 section-lede !mt-5">
              We timed AlphaFold 2’s real JAX forward pass on Colab Intel Xeon
              CPU (2 vCPU), Colab NVIDIA Tesla T4, and Stanford GKE TPU v5e-8
              (2×4 lite podslice), tracking where time and idle chips go. As a
              follow-up we also ran AlphaFold 3 on the same Colab Xeon CPU and
              Tesla T4; its public release has no TPU. Steady-state AF2 on TPU is
              451× faster than CPU.
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
            <p className="kicker">AF2 steady-state vs CPU</p>
            <p className="eq mt-1 font-display text-[clamp(5rem,14vw,8.5rem)] font-bold leading-[0.84] tracking-[-0.05em] text-ink">
              {n}
              <span className="text-teal">×</span>
            </p>
            <p className="section-note mt-3 lg:ml-auto lg:max-w-[18rem]">
              TPU v5e-8 0.47s · Tesla T4 13.1s (16.2×) · Xeon CPU 212s
            </p>
          </div>
        </div>
      </div>

      <div className="shell shrink-0 pt-8">
        <hr className="rule" />
        <p className="section-note pt-4">
          Lorenzo Pazienza & Ihab El Bani · Intro to HPC and AI Systems · Profs.
          Steve Jones & Mourad Bouache
        </p>
      </div>
    </section>
  )
}
