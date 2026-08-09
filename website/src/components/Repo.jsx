const TREE = [
  ['configs/', 'Kubernetes job manifests'],
  ['src/', 'Benchmark script used on all backends'],
  ['notebooks/', 'Colab CPU, GPU, and ESMFold fold'],
  ['figures/', 'Charts referenced on this page'],
  ['structure/', 'Ubiquitin PDB'],
  ['profiling/', 'XLA trace notes'],
  ['results/', 'Per-run JSON, comparison, sweep/'],
  ['website/', 'This site'],
]

const REPO = 'https://github.com/lorenzopazienza/alphafold-tpu-benchmark'

export default function Repo() {
  return (
    <section id="reproduce" className="border-t border-line bg-panel">
      <div className="viewport-tight shell min-w-0">
        <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
          <div className="max-w-xl">
            <p className="kicker">Reproduce</p>
            <h2 className="mt-3 font-display text-[clamp(1.9rem,4vw,2.75rem)] font-semibold tracking-[-0.03em] text-ink">
              Code and commands
            </h2>
            <p className="mt-3 text-[15px] leading-relaxed text-slate">
              Docker builds for CPU/GPU and Kubernetes manifests for the class
              TPU cluster are in the repo.
            </p>
          </div>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="link-quiet inline-flex min-h-11 items-center self-start text-[15px] font-medium"
          >
            <span className="sm:hidden">Open on GitHub</span>
            <span className="hidden break-all sm:inline">
              github.com/lorenzopazienza/alphafold-tpu-benchmark
            </span>
          </a>
        </div>

        <div className="mt-8 grid min-w-0 gap-10 lg:grid-cols-2 lg:mt-10 lg:gap-12">
          <div className="min-w-0">
            <p className="kicker">Layout</p>
            <ul className="mt-5 space-y-0 font-mono text-[13px]">
              {TREE.map(([path, desc]) => (
                <li
                  key={path}
                  className="grid grid-cols-1 gap-0.5 border-t border-line py-2.5 last:border-b sm:grid-cols-[7.5rem_minmax(0,1fr)] sm:gap-3"
                >
                  <span className="text-ink">{path}</span>
                  <span className="min-w-0 text-mute">{desc}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="min-w-0">
            <p className="kicker">CPU example</p>
            <pre className="mt-5 max-w-full overflow-x-auto bg-ink p-4 font-mono text-[11.5px] leading-relaxed break-words whitespace-pre-wrap text-white/90 sm:p-5 sm:text-[12.5px] md:p-6 md:text-[13px] md:whitespace-pre md:break-normal">
{`docker build --build-arg JAX_VARIANT=cpu \\
  -t af-bench:cpu .
docker run -v $(pwd)/results:/alphafold/results \\
  af-bench:cpu --run_tag=cpu`}
            </pre>
            <p className="mt-4 text-[13px] leading-relaxed text-mute">
              GPU and TPU variants, plus Colab notebooks, are documented in the
              README. Sweep jobs are under configs/.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
