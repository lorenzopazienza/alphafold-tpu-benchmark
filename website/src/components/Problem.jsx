export default function Problem() {
  return (
    <section id="problem" className="border-t border-line">
      <div className="viewport-tight shell grid gap-12 lg:grid-cols-[0.85fr_1.15fr] lg:gap-20">
        <div>
          <p className="kicker">Problem</p>
          <h2 className="mt-4 font-display text-[clamp(1.9rem,4vw,2.75rem)] font-semibold leading-[1.1] tracking-[-0.03em] text-ink">
            Where does AlphaFold spend time on CPU, GPU, and TPU?
          </h2>
        </div>

        <div className="flex flex-col justify-center gap-6">
          <p className="max-w-xl text-[1.05rem] leading-[1.65] text-slate md:text-[1.125rem]">
            Protein structure prediction is a real JAX/Haiku workload. We run
            the same model config, the same script, and the same input shape on
            three backends, then measure init, first predict (compile + run),
            and steady-state predict.
          </p>
          <p className="max-w-xl text-[15px] leading-relaxed text-mute">
            Hardware: Colab CPU and T4 for the baselines; Stanford class cluster
            TPU v5e-8 (2×4) for the TPU numbers and the scaling sweeps.
          </p>
        </div>
      </div>
    </section>
  )
}
