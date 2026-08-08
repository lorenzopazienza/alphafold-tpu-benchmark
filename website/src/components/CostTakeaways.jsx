export default function CostTakeaways() {
  return (
    <section id="cost" className="border-t border-line">
      <div className="shell pt-[clamp(4rem,9vh,6.5rem)]">
        <div className="max-w-2xl">
          <p className="kicker">Cost</p>
          <h2 className="mt-3 font-display text-[clamp(1.9rem,4vw,2.75rem)] font-semibold tracking-[-0.03em] text-ink">
            Cost per prediction is nearly flat in chip count
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-slate">
            Choose how many chips you need for latency or throughput. Do not
            expect a lower unit cost from renting a larger slice.
          </p>
        </div>

        <div className="mt-14 overflow-x-auto border-y border-line py-10 text-center">
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
          <p className="mx-auto mt-4 max-w-lg text-[13px] text-mute">
            chips {'{'}1, 2, 4, 8{'}'} × length {'{'}100, 250, 500, 1000{'}'}
          </p>
        </div>
      </div>

      <div className="bg-ink py-14 text-paper">
        <div className="shell max-w-3xl">
          <p className="text-[1.15rem] leading-relaxed md:text-[1.25rem]">
            On the baseline path only 1 of 8 chips was active (~87% idle), so
            list-price TPU and GPU land almost together: $1.25 vs $1.27 per 1k
            predictions. After pmap uses the full slice, cost per prediction
            stays roughly constant as chip count grows.
          </p>
        </div>
      </div>

      <div className="shell pb-[clamp(4rem,9vh,6.5rem)] pt-12">
        <p className="mb-6 text-center text-[13px] text-mute">
          Cost per 1,000 predictions
        </p>
        <dl className="grid gap-8 sm:grid-cols-3 sm:gap-6">
          {[
            { k: 'CPU', v: '$11.19' },
            { k: 'GPU T4', v: '$1.27' },
            { k: 'TPU 8-chip', v: '$1.25' },
          ].map((row) => (
            <div key={row.k} className="text-center">
              <dt className="text-[13px] text-mute">{row.k}</dt>
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
