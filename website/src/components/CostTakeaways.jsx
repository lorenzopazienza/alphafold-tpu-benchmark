export default function CostTakeaways() {
  return (
    <section id="cost" className="border-t border-line">
      <div className="shell section-y !pb-0">
        <div className="max-w-2xl">
          <p className="kicker">Cost</p>
          <h2 className="section-title">
            Cost per prediction is nearly flat in chip count
          </h2>
          <p className="section-lede">
            Choose how many chips you need for latency or throughput. Do not
            expect a lower unit cost from renting a larger slice.
          </p>
        </div>

        <div className="mt-10 overflow-x-auto border-y border-line py-8 text-center md:mt-12">
          <p className="kicker">Empirical fit · 16 TPU runs · R² 0.981</p>
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
            chips {'{'}1, 2, 4, 8{'}'} × length {'{'}100, 250, 500, 1000{'}'}
          </p>
        </div>
      </div>

      <div className="bg-ink py-10 text-paper md:py-12">
        <div className="shell max-w-3xl">
          <p className="section-lede !mt-0 !max-w-none text-paper/90">
            On the Stanford GKE TPU v5e-8 (2×4 lite) baseline path only 1 of 8
            chips was active (~87% idle), so list-price TPU and Colab Tesla T4
            land almost together: $1.25 vs $1.27 per 1k predictions (Colab Xeon
            CPU is $11.19). Multi-query pmap and single-query ensemble sharding
            (pmap + pmean) both fill the slice; after that, cost per prediction
            stays roughly constant as chip count grows.
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
            { k: 'GPU Tesla T4', v: '$1.27' },
            { k: 'TPU v5e-8', v: '$1.25' },
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
