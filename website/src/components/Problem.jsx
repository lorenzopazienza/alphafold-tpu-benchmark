export default function Problem() {
  return (
    <section id="problem" className="border-t border-line">
      <div className="viewport-tight shell grid gap-8 lg:grid-cols-[0.85fr_1.15fr] lg:gap-14">
        <div>
          <p className="kicker">Problem</p>
          <h2 className="mt-4 font-display text-[clamp(1.9rem,4vw,2.75rem)] font-semibold leading-[1.1] tracking-[-0.03em] text-ink">
            Where does AlphaFold spend time on CPU, GPU, and TPU?
          </h2>
        </div>

        <div className="flex flex-col justify-center gap-6">
          <p className="max-w-xl text-[1.05rem] leading-[1.65] text-slate md:text-[1.125rem]">
            Protein structure prediction is a real JAX/Haiku workload. Same
            model config, script, and input shape (118 residues, model_3, 0
            recycles) on CPU, GPU, and TPU — then measure init, cold predict
            (compile + run), and steady-state.
          </p>
          <p className="max-w-xl text-[15px] leading-relaxed text-mute">
            Resource challenge: cold XLA compile vs warm device time, and an
            8-chip TPU pod that often only busy one chip. Baselines on Colab
            CPU/T4; TPU numbers on Stanford GKE v5e-8 (2×4).
          </p>
        </div>
      </div>
    </section>
  )
}
