const FOLLOW_UP = 'https://github.com/lorenzopazienza/alphafold-on-tpu'

export default function Next() {
  return (
    <section id="next" className="border-t border-line bg-panel">
      <div className="viewport-tight shell">
        <div className="max-w-2xl">
          <p className="kicker">Next</p>
          <h2 className="section-title">What comes next</h2>
          <p className="section-lede">
            Follow-up research continues in a separate repository,{' '}
            <a
              href={FOLLOW_UP}
              target="_blank"
              rel="noreferrer"
              className="link-quiet font-medium text-ink"
            >
              alphafold-on-tpu
            </a>
            . The results on this page are those of the paper and are not
            changed there; new results are reported alongside them, not in place
            of them.
          </p>
        </div>
      </div>
    </section>
  )
}
