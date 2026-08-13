export default function Problem() {
  return (
    <section id="problem" className="border-t border-line">
      <div className="viewport-tight shell grid gap-8 lg:grid-cols-[0.85fr_1.15fr] lg:gap-14">
        <div>
          <p className="kicker">Problem</p>
          <h2 className="section-title">
            Where does AlphaFold 2 spend time on CPU, GPU, and TPU?
          </h2>
        </div>

        <div className="flex flex-col justify-center gap-5">
          <p className="section-lede !mt-0 max-w-xl">
            <span className="font-medium text-ink">The problem.</span> Biology
            needs a protein’s 3D shape. AlphaFold 2 (DeepMind) made
            high-accuracy structure prediction practical with a large JAX/Haiku
            Evoformer, but running that inference is expensive and opaque across
            accelerators: cold XLA compiles, underused TPU pods, unclear CPU vs
            GPU vs TPU trade-offs.
          </p>
          <p className="section-lede !mt-0 max-w-xl">
            <span className="font-medium text-ink">What we did.</span> Same
            AlphaFold 2 forward pass, script, and shape (118 residues,{' '}
            <code className="font-mono text-[0.92em] text-ink">model_3</code>, 0
            recycles, Haiku random-init params) on Colab Intel Xeon CPU (2
            vCPU), Colab NVIDIA Tesla T4, and Stanford GKE TPU v5e-8 (
            <code className="font-mono text-[0.92em] text-ink">
              tpu-v5-lite-podslice
            </code>
            , topology 2×4, 8 chips). Then we timed init, cold predict, and
            steady-state, and fixed the bottlenecks we found.
          </p>
          <p className="section-note max-w-xl">
            Follow-up:{' '}
            <a href="#af3" className="link-quiet font-medium text-ink">
              AlphaFold 3
            </a>{' '}
            (a separate diffusion codebase) on the same Colab Xeon CPU and Tesla
            T4, plus a confirmed finding that its public release does not
            support TPU.
          </p>
        </div>
      </div>
    </section>
  )
}
