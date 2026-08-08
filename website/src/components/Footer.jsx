const REPO = 'https://github.com/lorenzopazienza/alphafold-tpu'

export default function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="shell flex flex-col gap-6 py-12 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-display text-lg font-semibold tracking-tight text-ink">
            AlphaFold on TPU
          </p>
          <p className="mt-2 text-[13px] leading-relaxed text-mute">
            Lorenzo Pazienza & Ihab El Bani
            <br />
            Stanford University · Summer Session 2026, IHP
            <br />
            Intro to High Performance Computing and AI Systems
            <br />
            Profs. Steve Jones & Mourad Bouache
          </p>
        </div>
        <a
          href={REPO}
          target="_blank"
          rel="noreferrer"
          className="link-quiet inline-flex min-h-11 items-center text-[13px]"
        >
          GitHub
        </a>
      </div>
    </footer>
  )
}
