export default function CostTakeaways() {
  return (
    <section id="cost" className="border-t border-line">
      <div className="shell section-y !pb-0">
        <div className="max-w-2xl">
          <p className="kicker">Cost</p>
          <h2 className="section-title">
            Idle chips cost as much as busy ones
          </h2>
          <p className="section-lede">
            At list prices, the default path&rsquo;s seven idle chips make the TPU
            slice cost about the same per prediction as the T4: $1.25 vs $1.27
            per 1,000 predictions (CPU $11.19), on the August baselines. Rented
            one chip at a time, the same TPU work would cost $0.157 per 1,000,
            about 8.1× cheaper than the GPU. Running eight independent proteins
            with jax.pmap on the full pod brings it to about $0.181 per 1,000.
          </p>
        </div>

        <div className="mt-10 overflow-x-auto border-y border-line py-8 text-center md:mt-12">
          <p className="kicker">Descriptive fit · 16 TPU runs · R² 0.981</p>
          <p className="eq mt-5 font-mono text-[clamp(0.95rem,2.6vw,1.35rem)] leading-relaxed tracking-tight text-ink">
            <span className="text-mute">throughput</span>
            <span className="mx-2 text-mute">≈</span>
            4527.77
            <span className="mx-1.5 text-mute">·</span>
            chips<sup className="text-teal">0.963</sup>
            <span className="mx-1.5 text-mute">·</span>
            length<sup className="text-teal">−1.572</sup>
          </p>
          <p className="section-note mx-auto mt-4 max-w-lg">
            A summary of this grid, not a model: the fit deviates from the
            measurements by up to 32% at the extremes of the length range.
          </p>
        </div>
      </div>

      <div className="bg-ink py-10 text-paper md:py-12">
        <div className="shell max-w-3xl">
          <p className="section-lede !mt-0 !max-w-none text-paper/90">
            More chips are not free. On the measured grid, going from 1 to 8
            chips raises cost per prediction by 23% at 100 residues, 7% at 250,
            3% at 500 and about 1% at 1000. The fitted exponent&rsquo;s near-flat
            cost holds only for long sequences. The external ensemble (pmap +
            pmean) fills all eight chips, but it has no throughput measurement
            and is not priced.
          </p>
        </div>
      </div>

      <div className="shell section-y !pt-10">
        <p className="section-note mb-6 text-center">
          Cost per 1,000 predictions
        </p>
        <dl className="grid gap-8 sm:grid-cols-3 sm:gap-6">
          {[
            { k: 'CPU Xeon', v: '$11.19' },
            { k: 'GPU NVIDIA Tesla T4', v: '$1.27' },
            { k: 'TPU pod, 1 of 8 chips active', v: '$1.25' },
          ].map((row) => (
            <div key={row.k} className="text-center">
              <dt className="section-note">{row.k}</dt>
              <dd className="eq mt-1 font-display text-3xl font-semibold tracking-tight text-ink">
                {row.v}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  )
}
