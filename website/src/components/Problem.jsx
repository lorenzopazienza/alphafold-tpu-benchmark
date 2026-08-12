export default function Problem() {
  return (
    <section id="problem" className="border-t border-line">
      <div className="viewport-tight shell grid gap-8 lg:grid-cols-[0.85fr_1.15fr] lg:gap-14">
        <div>
          <p className="kicker">Problem</p>
          <h2 className="mt-4 font-display text-[clamp(1.9rem,4vw,2.75rem)] font-semibold leading-[1.1] tracking-[-0.03em] text-ink">
            Where does AlphaFold 2 spend time on CPU, GPU, and TPU?
          </h2>
        </div>

        <div className="flex flex-col justify-center gap-6">
          <p className="max-w-xl text-[1.05rem] leading-[1.65] text-slate md:text-[1.125rem]">
            <span className="font-medium text-ink">The problem.</span> Biology
            needs a protein’s 3D shape. AlphaFold 2 (DeepMind) made
            high-accuracy structure prediction practical with a large JAX/Haiku
            Evoformer — but running that inference is expensive and opaque
            across accelerators: cold XLA compiles, underused TPU pods, unclear
            CPU vs GPU vs TPU trade-offs.
          </p>
          <p className="max-w-xl text-[1.05rem] leading-[1.65] text-slate md:text-[1.125rem]">
            <span className="font-medium text-ink">What we did.</span> Same
            AlphaFold 2 forward pass, script, and shape (118 residues, model_3,
            0 recycles) on Colab CPU, T4 GPU, and Stanford GKE TPU v5e-8 — then
            timed init, cold predict, and steady-state, and fixed the bottlenecks
            we found.
          </p>
          <p className="max-w-xl text-[15px] leading-relaxed text-mute">
            Follow-up:{' '}
            <a href="#af3" className="link-quiet font-medium text-ink">
              AlphaFold 3
            </a>{' '}
            — a separate diffusion codebase — on the same Colab CPU/GPU, plus a
            confirmed finding that its public release does not support TPU.
          </p>
        </div>
      </div>
    </section>
  )
}
