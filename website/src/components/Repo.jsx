const TREE = [
  ['configs/', 'K8s Jobs (AF2 sweeps + AF3 TPU attempt)'],
  ['figures/', 'Charts + AF3 structure still'],
  ['notebooks/', 'AF2 + AF3 Google Colab notebooks'],
  ['presentation/', 'Course slides PDF'],
  ['profiling/', 'XLA trace notes'],
  ['results/', 'AF2 clocks + sweep/ + af3_comparison.md'],
  ['src/', 'AF2 spikes + make_af3_input.py'],
  ['structure/', 'Ubiquitin PDB + AF3 CIF'],
  ['website/', 'This site'],
]

const REPO = 'https://github.com/lorenzopazienza/alphafold-tpu-benchmark'
const SLIDES =
  '/presentation/AlphaFold_on_Google_TPUs_Pazienza_Lorenzo_Ihab_El_Bani.pdf'

export default function Repo() {
  return (
    <section id="reproduce" className="border-t border-line bg-panel">
      <div className="viewport-tight shell min-w-0">
        <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
          <div className="max-w-xl">
            <p className="kicker">Reproduce</p>
            <h2 className="section-title">Code and commands</h2>
            <p className="section-lede !mt-3">
              AF2 Docker/K8s paths and AF3 Google Colab notebooks (
              <code className="font-mono text-[0.9em]">af3_cpu_colab</code> /{' '}
              <code className="font-mono text-[0.9em]">af3_gpu_colab</code>) live
              in the repo. Full AF3 write-up:{' '}
              <code className="font-mono text-[0.9em]">
                results/sweep/af3_comparison.md
              </code>
              . Course deck is in{' '}
              <code className="font-mono text-[0.9em]">presentation/</code>.
            </p>
          </div>
          <div className="flex flex-col gap-3 self-start sm:items-end">
            <a
              href={SLIDES}
              download
              className="link-quiet inline-flex min-h-11 items-center text-sm font-medium"
            >
              Download slides PDF
            </a>
            <a
              href={REPO}
              target="_blank"
              rel="noreferrer"
              className="link-quiet inline-flex min-h-11 items-center text-sm font-medium"
            >
              <span className="sm:hidden">Open on GitHub</span>
              <span className="hidden break-all sm:inline">
                github.com/lorenzopazienza/alphafold-tpu-benchmark
              </span>
            </a>
          </div>
        </div>

        <div className="mt-8 grid min-w-0 gap-10 lg:mt-10 lg:grid-cols-2 lg:gap-12">
          <div className="min-w-0">
            <p className="kicker">Layout</p>
            <ul className="mt-5 space-y-0 font-mono text-sm">
              {TREE.map(([path, desc]) => (
                <li
                  key={path}
                  className="grid grid-cols-1 gap-0.5 border-t border-line py-2.5 last:border-b sm:grid-cols-[8.5rem_minmax(0,1fr)] sm:gap-3"
                >
                  <span className="text-ink">{path}</span>
                  <span className="min-w-0 text-mute">{desc}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="min-w-0">
            <p className="kicker">CPU example</p>
            <pre className="mt-5 max-w-full overflow-x-auto bg-ink p-4 font-mono text-[0.75rem] leading-relaxed break-words whitespace-pre-wrap text-white/90 sm:p-5 sm:text-[0.8125rem] md:p-6 md:text-sm md:whitespace-pre md:break-normal">
{`docker build --build-arg JAX_VARIANT=cpu \\
  -t af-bench:cpu .
docker run -v $(pwd)/results:/alphafold/results \\
  af-bench:cpu --run_tag=cpu`}
            </pre>
            <p className="section-note mt-4">
              GPU and TPU variants for AF2 are in the README. AF3 TPU Job is
              kept as documentation of the unsupported CLI, not a working path.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
