import { useState } from 'react'

const ARXIV_ABS = 'https://arxiv.org/abs/2609.34818'
const ARXIV_PDF = 'https://arxiv.org/pdf/2609.34818'
const REPO = 'https://github.com/lorenzopazienza/alphafold-tpu-benchmark'
const SOURCE = `${REPO}/tree/main/paper/preprint/source`

const ABSTRACT =
  'AlphaFold2 is written in JAX, so the same inference code compiles and runs unchanged on CPUs, GPUs and Google Cloud TPUs. That portability makes the accelerator look like the main decision a user has to make. We show that it is not. Running one AlphaFold2 inference workload across a Colab CPU runtime, an NVIDIA T4 GPU and a dedicated eight-chip Cloud TPU v5e slice, we find a large hardware advantage for the TPU, 0.47 s per call in steady state on a single chip against 13.1 s on the T4 in the same measurement campaign, and three ways in which the software layer decides how much of it a user actually gets. The default execution path uses one chip of the eight, and at list prices the idle capacity makes the slice cost about as much per prediction as the GPU. Batching with JAX’s vmap never exceeds single-query throughput, while mapping queries across chips with JAX’s pmap gives eight chips 6.5-7.9x the throughput of one on a matched grid; automatic sharding leaves the per-chip footprint unchanged, consistent with replication, most plausibly because AlphaFold2 carries no sharding annotations. Our retained trace analysis of a first call at a new input shape reports about three quarters of the traced span in JAX tracing and compilation rather than execution. Reruns five weeks later reproduced neither cloud baseline, the GPU one off by roughly a factor of two, so the hardware ratio above is specific to one campaign.'

const BIBTEX = `@misc{pazienza2026accelerator,
  title         = {Accelerator Choice Is Not Enough: {AlphaFold2} Inference on Cloud {TPUs}},
  author        = {Pazienza, Lorenzo and El Bani, Ihab},
  year          = {2026},
  eprint        = {2609.34818},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  doi           = {10.48550/arXiv.2609.34818},
  url           = {https://arxiv.org/abs/2609.34818}
}`

const LINKS = [
  { href: ARXIV_ABS, label: 'arXiv page' },
  { href: ARXIV_PDF, label: 'PDF' },
  { href: SOURCE, label: 'LaTeX source' },
  { href: REPO, label: 'Code and data' },
]

export default function Paper() {
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(BIBTEX)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      /* clipboard unavailable: the block stays selectable by hand */
    }
  }

  return (
    <section id="paper" className="border-t border-line">
      <div className="viewport-tight shell">
        <div className="max-w-3xl">
          <p className="kicker">Paper</p>
          <h2 className="section-title">
            Accelerator Choice Is Not Enough: AlphaFold2 Inference on Cloud TPUs
          </h2>
          <p className="section-note mt-3">
            Lorenzo Pazienza, Ihab El Bani · arXiv preprint, September 2026 ·
            cs.DC
          </p>
        </div>

        <div className="mt-8 grid gap-10 lg:mt-10 lg:grid-cols-[1.15fr_0.85fr] lg:gap-12">
          <div className="min-w-0">
            <p className="kicker">Abstract</p>
            <p
              className={`section-body mt-3 max-w-2xl ${
                open ? '' : 'line-clamp-3'
              }`}
            >
              {ABSTRACT}
            </p>
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              aria-expanded={open}
              className="link-quiet mt-3 inline-flex min-h-11 items-center text-sm font-medium"
            >
              {open ? 'Hide the abstract' : 'Read the abstract'}
            </button>

            <div className="mt-6 flex flex-wrap gap-x-5 gap-y-2">
              {LINKS.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noreferrer"
                  className="link-quiet inline-flex min-h-11 min-w-11 items-center text-sm font-medium"
                >
                  {link.label}
                </a>
              ))}
            </div>
          </div>

          <div className="min-w-0">
            <div className="flex items-baseline justify-between gap-4">
              <p className="kicker">Cite</p>
              <button
                type="button"
                onClick={copy}
                className="link-quiet inline-flex min-h-11 min-w-11 items-center text-sm font-medium"
              >
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
            <pre className="mt-3 max-w-full overflow-x-auto bg-ink p-4 font-mono text-[0.75rem] leading-relaxed text-white/90 sm:p-5 sm:text-[0.8125rem]">
{BIBTEX}
            </pre>
            <p className="section-note mt-4">
              The paper cites commit a63e955 of the repository (tag arxiv-v1).
              Every number in it traces to paper/data/canonical_results.md.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
