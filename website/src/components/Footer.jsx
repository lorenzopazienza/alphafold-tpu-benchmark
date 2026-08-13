const REPO = 'https://github.com/lorenzopazienza/alphafold-tpu-benchmark'

export default function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="shell flex flex-col gap-6 py-8 md:flex-row md:items-end md:justify-between md:py-10">
        <div>
          <p className="font-display text-lg font-semibold tracking-tight text-ink">
            AlphaFold TPU Benchmark
          </p>
          <p className="section-note mt-2">
            Authors:{' '}
            <span itemProp="author">Lorenzo Pazienza</span>
            {' & '}
            <span itemProp="author">Ihab El Bani</span>
            <br />
            Stanford University · Summer Session 2026
            <br />
            Introduction to High Performance Computing and AI Systems (ME344)
            <br />
            Profs. Steve Jones & Mourad Bouache
          </p>
        </div>
        <div className="flex flex-col gap-2 sm:items-end">
          <a
            href="/presentation/AlphaFold_on_Google_TPUs_Pazienza_Lorenzo_Ihab_El_Bani.pdf"
            download
            className="link-quiet inline-flex min-h-11 items-center text-sm"
          >
            Slides PDF
          </a>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="link-quiet inline-flex min-h-11 items-center text-sm"
          >
            GitHub
          </a>
          <a
            href="/llms.txt"
            className="link-quiet inline-flex min-h-11 items-center text-sm"
          >
            llms.txt
          </a>
        </div>
      </div>
    </footer>
  )
}
