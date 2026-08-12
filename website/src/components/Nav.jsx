import { useEffect, useState } from 'react'

const LINKS = [
  { href: '#structure', label: 'Structure' },
  { href: '#problem', label: 'Problem' },
  { href: '#approach', label: 'Approach' },
  { href: '#results', label: 'Results' },
  { href: '#experiments', label: 'Experiments' },
  { href: '#af3', label: 'AF3' },
  { href: '#cost', label: 'Cost' },
  { href: '#reproduce', label: 'Reproduce' },
]

const SECTION_IDS = LINKS.map((l) => l.href.slice(1))

function navOffset() {
  const header = document.querySelector('header')
  return (header?.offsetHeight || 64) + 12
}

/** Active = last section whose top has crossed below the sticky nav. */
function sectionAtScroll() {
  const y = window.scrollY + navOffset()
  let current = ''
  for (const id of SECTION_IDS) {
    const el = document.getElementById(id)
    if (!el) continue
    const top = el.getBoundingClientRect().top + window.scrollY
    if (top <= y) current = `#${id}`
  }
  return current
}

export default function Nav() {
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState('')

  useEffect(() => {
    const sync = () => {
      setScrolled(window.scrollY > 8)
      setActive(sectionAtScroll())
    }
    sync()
    window.addEventListener('scroll', sync, { passive: true })
    window.addEventListener('resize', sync)
    window.addEventListener('hashchange', sync)
    return () => {
      window.removeEventListener('scroll', sync)
      window.removeEventListener('resize', sync)
      window.removeEventListener('hashchange', sync)
    }
  }, [])

  const go = (href) => {
    setActive(href)
    setOpen(false)
  }

  return (
    <header
      className={`sticky top-0 z-50 transition-[background,border-color] duration-300 ${
        scrolled
          ? 'border-b border-line bg-paper/90 backdrop-blur-md'
          : 'border-b border-transparent bg-paper/70 backdrop-blur-sm'
      }`}
    >
      <nav className="shell flex h-16 items-center justify-between gap-6 lg:h-[4.25rem]">
        <a
          href="#top"
          onClick={() => go('')}
          className="inline-flex min-h-11 items-center font-display text-base font-semibold tracking-tight text-ink md:text-lg"
        >
          AlphaFold TPU Benchmark
        </a>

        <div className="hidden items-center gap-1 lg:flex">
          <ul className="flex items-center gap-1 xl:gap-0.5">
            {LINKS.map((link) => {
              const on = active === link.href
              return (
                <li key={link.href}>
                  <a
                    href={link.href}
                    onClick={() => go(link.href)}
                    className={`inline-flex items-center rounded-md px-2.5 py-2 text-[0.9375rem] transition-colors xl:px-3 ${
                      on
                        ? 'font-semibold text-teal'
                        : 'font-medium text-mute hover:text-ink'
                    }`}
                  >
                    {link.label}
                  </a>
                </li>
              )
            })}
          </ul>
        </div>

        <button
          type="button"
          className="inline-flex min-h-11 min-w-11 items-center justify-center text-[0.9375rem] font-semibold text-ink lg:hidden"
          aria-expanded={open}
          aria-controls="mobile-nav"
          onClick={() => setOpen((v) => !v)}
        >
          {open ? 'Close' : 'Menu'}
        </button>
      </nav>

      {open && (
        <ul id="mobile-nav" className="shell border-t border-line py-3 lg:hidden">
          {LINKS.map((link) => (
            <li key={link.href}>
              <a
                href={link.href}
                className={`block py-3 text-base ${
                  active === link.href
                    ? 'font-semibold text-teal'
                    : 'text-slate'
                }`}
                onClick={() => go(link.href)}
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>
      )}
    </header>
  )
}
