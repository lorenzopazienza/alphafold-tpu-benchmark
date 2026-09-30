export default function Problem() {
  return (
    <section id="problem" className="border-t border-line">
      <div className="viewport-tight shell grid gap-8 lg:grid-cols-[0.85fr_1.15fr] lg:gap-14">
        <div>
          <p className="kicker">Question</p>
          <h2 className="section-title">
            What decides the performance a user actually gets?
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
            <span className="font-medium text-ink">What we did.</span> We ran
            AlphaFold2&rsquo;s JAX forward pass on a Google Colab CPU runtime (2
            vCPU), a Google Colab NVIDIA Tesla T4 and a Stanford GKE TPU v5e-8
            slice (
            <code className="font-mono text-[0.92em] text-ink">
              tpu-v5-lite-podslice
            </code>
            , 2×4, 8 chips). Single-query timings use the same 118-residue input,{' '}
            <code className="font-mono text-[0.92em] text-ink">model_3</code>, 0
            recycles and randomly initialized parameters on every backend. Then
            we asked where the time goes on a first call, and what it takes to
            put all eight chips to work.
          </p>
          <p className="section-note max-w-xl">
            Follow-up:{' '}
            <a href="#af3" className="link-quiet font-medium text-ink">
              AlphaFold 3
            </a>{' '}
            (a separate diffusion codebase) on the same Google Colab Intel Xeon
            CPU and NVIDIA Tesla T4, plus a confirmed finding that its public release does not
            support TPU.
          </p>
        </div>
      </div>
    </section>
  )
}
