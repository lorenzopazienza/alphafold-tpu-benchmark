const REPO = 'https://github.com/lorenzopazienza/alphafold-tpu-benchmark'
const ARXIV_ABS = 'https://arxiv.org/abs/2609.34818'
const ARXIV_PDF = 'https://arxiv.org/pdf/2609.34818'
const SLIDES =
  '/presentation/AlphaFold_on_Google_TPUs_Pazienza_Lorenzo_Ihab_El_Bani.pdf'

export default function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="shell flex flex-col gap-6 py-8 md:flex-row md:items-end md:justify-between md:py-10">
        <div>
          <p className="font-display text-lg font-semibold tracking-tight text-ink">
            Accelerator Choice Is Not Enough
          </p>
          <p className="section-note mt-2">
            <span itemProp="author">Lorenzo Pazienza</span> (LUISS Guido Carli
            University)
            {' & '}
            <span itemProp="author">Ihab El Bani</span> (Al Akhawayn University)
            <br />
            Work carried out during Stanford&rsquo;s ME344, High Performance
            Computing and AI Systems, Summer Session 2026.
            <br />
            Thanks to Mourad Bouache (Google), who taught the course, and Steve
            Jones (Stanford High Performance Computing Center), for access to the
            TPU v5e cluster.
          </p>
        </div>
        <div className="flex flex-col gap-2 sm:items-end">
          <a
            href={ARXIV_ABS}
            target="_blank"
            rel="noreferrer"
            className="link-quiet inline-flex min-h-11 min-w-11 items-center text-sm"
          >
            arXiv
          </a>
          <a
            href={ARXIV_PDF}
            target="_blank"
            rel="noreferrer"
            className="link-quiet inline-flex min-h-11 min-w-11 items-center text-sm"
          >
            PDF
          </a>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="link-quiet inline-flex min-h-11 min-w-11 items-center text-sm"
          >
            GitHub
          </a>
          <a
            href={SLIDES}
            download
            className="link-quiet inline-flex min-h-11 min-w-11 items-center text-sm"
          >
            Course slides
          </a>
          <a
            href="/llms.txt"
            className="link-quiet inline-flex min-h-11 min-w-11 items-center text-sm"
          >
            llms.txt
          </a>
        </div>
      </div>
    </footer>
  )
}
