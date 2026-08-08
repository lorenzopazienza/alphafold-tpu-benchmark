import { chromium } from 'playwright'
import { createServer } from 'http'
import { readFileSync, existsSync, statSync } from 'fs'
import { join, extname } from 'path'
import { fileURLToPath } from 'url'

const root = join(fileURLToPath(new URL('.', import.meta.url)), '..', 'dist')
const mime = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.pdb': 'chemical/x-pdb',
  '.json': 'application/json',
  '.woff2': 'font/woff2',
}

function serve() {
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      let path = join(root, decodeURIComponent((req.url || '/').split('?')[0]))
      if (path.endsWith('/')) path = join(path, 'index.html')
      if (!existsSync(path) || statSync(path).isDirectory()) {
        path = join(root, 'index.html')
      }
      const type = mime[extname(path)] || 'application/octet-stream'
      res.writeHead(200, { 'Content-Type': type })
      res.end(readFileSync(path))
    })
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address()
      resolve({ server, url: `http://127.0.0.1:${port}/` })
    })
  })
}

const VIEWPORTS = [
  // mobile
  { name: 'Galaxy Fold cover', width: 280, height: 653 },
  { name: 'iPhone SE', width: 320, height: 568 },
  { name: 'Galaxy S20', width: 360, height: 800 },
  { name: 'iPhone 12/13 mini', width: 375, height: 812 },
  { name: 'iPhone 14', width: 390, height: 844 },
  { name: 'Pixel 7', width: 412, height: 915 },
  { name: 'iPhone 14 Pro Max', width: 430, height: 932 },
  // landscape phones
  { name: 'iPhone SE landscape', width: 568, height: 320 },
  { name: 'iPhone 14 landscape', width: 844, height: 390 },
  // tablets
  { name: 'iPad Mini', width: 768, height: 1024 },
  { name: 'iPad Pro 11', width: 834, height: 1194 },
  { name: 'iPad Pro 12.9', width: 1024, height: 1366 },
  // desktop
  { name: 'Laptop 1280', width: 1280, height: 800 },
  { name: 'Desktop 1440', width: 1440, height: 900 },
  { name: 'Desktop 1920', width: 1920, height: 1080 },
]

const SECTIONS = [
  'top',
  'structure',
  'problem',
  'approach',
  'results',
  'experiments',
  'cost',
  'reproduce',
]

function isVisible(el) {
  const style = window.getComputedStyle(el)
  if (style.display === 'none' || style.visibility === 'hidden') return false
  if (Number(style.opacity) === 0) return false
  const r = el.getBoundingClientRect()
  return r.width > 0 && r.height > 0
}

const { server, url } = await serve()
const browser = await chromium.launch()
const report = []

try {
  for (const vp of VIEWPORTS) {
    const mobile = vp.width < 768
    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      deviceScaleFactor: vp.width < 1024 ? 2 : 1,
      isMobile: mobile,
      hasTouch: mobile || vp.width < 1024,
      reducedMotion: 'reduce',
    })
    const page = await context.newPage()
    const errors = []
    const failed = []
    page.on('pageerror', (e) => errors.push(String(e.message || e)))
    page.on('response', (res) => {
      if (res.status() >= 400) failed.push(`${res.status()} ${res.url()}`)
    })

    await page.goto(url, { waitUntil: 'networkidle', timeout: 45000 })
    await page.waitForTimeout(400)

    const size = page.viewportSize()

    const metrics = await page.evaluate(
      ({ sections, isVisibleSrc }) => {
        // eslint-disable-next-line no-new-func
        const visible = new Function(`return (${isVisibleSrc})`)()
        const doc = document.documentElement
        const body = document.body
        const overflowX =
          Math.max(doc.scrollWidth, body.scrollWidth) - window.innerWidth

        const wideNodes = []
        for (const el of document.querySelectorAll(
          'h1, h2, h3, p, li, pre, code, figure, img, button, a, dl, dd, dt, nav, canvas, svg, table',
        )) {
          if (!visible(el)) continue
          const r = el.getBoundingClientRect()
          if (r.right > window.innerWidth + 2 || r.left < -2) {
            wideNodes.push({
              tag: el.tagName,
              class: (el.className || '').toString().slice(0, 60),
              text: (el.textContent || '').trim().slice(0, 40),
              left: Math.round(r.left),
              right: Math.round(r.right),
            })
          }
        }

        const missingSections = sections.filter(
          (id) => !document.getElementById(id),
        )

        const menuBtn = document.querySelector(
          'button[aria-controls="mobile-nav"]',
        )
        const menuBtnVisible = !!(menuBtn && visible(menuBtn))

        // tiny tap targets on interactive controls (mobile concern)
        const smallTargets = []
        if (window.innerWidth < 768) {
          for (const el of document.querySelectorAll(
            'a, button, [role="button"]',
          )) {
            if (!visible(el)) continue
            const r = el.getBoundingClientRect()
            if (r.width > 0 && r.height > 0 && (r.height < 32 || r.width < 32)) {
              // allow inline text links in prose
              const inProse = el.closest('p, li, dd, figcaption')
              if (inProse && el.tagName === 'A') continue
              smallTargets.push({
                tag: el.tagName,
                text: (el.textContent || '').trim().slice(0, 32),
                w: Math.round(r.width),
                h: Math.round(r.height),
              })
            }
          }
        }

        return {
          overflowX,
          wideNodes: wideNodes.slice(0, 12),
          missingSections,
          menuBtnVisible,
          smallTargets: smallTargets.slice(0, 8),
          title: document.title,
        }
      },
      { sections: SECTIONS, isVisibleSrc: isVisible.toString() },
    )

    let menuOk = true
    if (metrics.menuBtnVisible) {
      await page.click('button[aria-controls="mobile-nav"]')
      await page.waitForTimeout(80)
      menuOk = await page.locator('#mobile-nav').isVisible()
      // close
      await page.click('button[aria-controls="mobile-nav"]')
      await page.waitForTimeout(40)
    } else {
      // desktop: expect desktop link list visible, not mobile drawer
      menuOk = await page.evaluate(() => {
        const lists = [...document.querySelectorAll('nav ul')]
        return lists.some((ul) => {
          const s = getComputedStyle(ul)
          return s.display !== 'none' && ul.getBoundingClientRect().height > 0
        })
      })
    }

    // scroll through page catching overflow + expand first experiment
    let maxOverflow = metrics.overflowX
    const scrollYs = [0, 600, 1400, 2400, 3600, 5200, 7000, 9000]
    for (const y of scrollYs) {
      await page.evaluate((yy) => window.scrollTo(0, yy), y)
      await page.waitForTimeout(40)
      const ox = await page.evaluate(
        () =>
          Math.max(
            document.documentElement.scrollWidth,
            document.body.scrollWidth,
          ) - window.innerWidth,
      )
      if (ox > maxOverflow) maxOverflow = ox
    }

    // expand a visible experiment row (mobile + desktop both render buttons)
    const expandBtn = page
      .locator('#experiments button[aria-expanded]:visible')
      .first()
    if (await expandBtn.count()) {
      await expandBtn.scrollIntoViewIfNeeded()
      await expandBtn.click({ timeout: 5000 })
      await page.waitForTimeout(120)
      const ox = await page.evaluate(
        () =>
          Math.max(
            document.documentElement.scrollWidth,
            document.body.scrollWidth,
          ) - window.innerWidth,
      )
      if (ox > maxOverflow) maxOverflow = ox
      await expandBtn.click({ timeout: 5000 })
    }

    // asset check: critical public files
    const assetFails = failed.filter(
      (f) =>
        f.includes('/assets/') ||
        f.includes('/figures/') ||
        f.includes('/structure/') ||
        f.includes('favicon'),
    )

    report.push({
      device: vp.name,
      requested: vp.width,
      actual: size?.width,
      overflowX: maxOverflow,
      menuOk,
      menuBtnVisible: metrics.menuBtnVisible,
      missingSections: metrics.missingSections,
      wideNodes: metrics.wideNodes,
      smallTargets: metrics.smallTargets,
      errors,
      assetFails,
    })
    await context.close()
  }
} finally {
  await browser.close()
  server.close()
}

const bad = report.filter(
  (r) =>
    r.overflowX > 1 ||
    !r.menuOk ||
    r.errors.length ||
    r.wideNodes.length ||
    r.missingSections.length ||
    r.assetFails.length ||
    r.actual !== r.requested,
)

console.log(JSON.stringify({ ok: bad.length === 0, bad, report }, null, 2))
process.exit(bad.length ? 1 : 0)
