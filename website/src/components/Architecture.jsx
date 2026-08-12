const NODES = [
  {
    id: 'cpu',
    label: 'CPU',
    detail: 'Colab / local',
    out: 'result_cpu.json',
  },
  {
    id: 'gpu',
    label: 'GPU',
    detail: 'Colab T4',
    out: 'result_gpu-t4.json',
  },
  {
    id: 'tpu',
    label: 'TPU',
    detail: 'Stanford v5e-8',
    out: 'result_tpu + traces',
  },
]

export default function Architecture() {
  return (
    <section id="approach" className="border-t border-line bg-panel">
      <div className="viewport-tight shell">
        <div className="max-w-2xl">
          <p className="kicker">System topology</p>
          <h2 className="mt-3 font-display text-[clamp(1.9rem,4vw,2.75rem)] font-semibold tracking-[-0.03em] text-ink">
            One script, three backends
          </h2>
          <p className="mt-4 text-[1.05rem] leading-relaxed text-slate">
            Docker for CPU/GPU images; Kubernetes Job + Kueue on GKE for the
            TPU worker. Every timing run calls{' '}
            <code className="font-mono text-[0.92em] text-ink">
              spike_tpu_forward_pass.py
            </code>
            . Only the accelerator changes.
          </p>
        </div>

        <div className="mt-8 grid gap-0 border-y border-line md:mt-10 md:grid-cols-[1.1fr_repeat(3,1fr)]">
          <div className="border-b border-line bg-ink px-5 py-6 text-paper md:border-b-0 md:border-r md:border-line/20">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-white/50">
              Script
            </p>
            <p className="mt-2 break-all font-display text-lg font-semibold leading-snug tracking-tight sm:text-xl">
              spike_tpu_forward_pass.py
            </p>
          </div>

          {NODES.map((n, i) => (
            <div
              key={n.id}
              className={`px-5 py-6 ${
                i < NODES.length - 1
                  ? 'border-b border-line md:border-b-0 md:border-r'
                  : ''
              }`}
            >
              <p className="font-display text-2xl font-semibold tracking-tight text-ink">
                {n.label}
              </p>
              <p className="mt-1 text-sm text-slate">{n.detail}</p>
              <p className="mt-6 font-mono text-[11px] text-mute">{n.out}</p>
            </div>
          ))}
        </div>

        <p className="mt-8 max-w-2xl text-[15px] leading-relaxed text-mute">
          <span className="font-medium text-ink">Parameters.</span> AF2 systems
          timings use Haiku random init so we exercise the same compiled graph
          without downloading trained weights (~350MB). The ubiquitin structure
          above is separate: ESMFold with trained weights, for biology, not for
          the CPU/GPU/TPU clocks. AlphaFold 3 uses its own notebooks and Jobs —
          see{' '}
          <a href="#af3" className="link-quiet font-medium text-ink">
            AF3
          </a>
          .
        </p>
      </div>
    </section>
  )
}
