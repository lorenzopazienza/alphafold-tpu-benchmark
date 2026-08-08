import { chromium } from 'playwright'
import { createServer } from 'http'
import { readFileSync, existsSync, statSync, mkdirSync } from 'fs'
import { join, extname } from 'path'
import { fileURLToPath } from 'url'

const root = join(fileURLToPath(new URL('.', import.meta.url)), '..', 'dist')
const outDir = join(fileURLToPath(new URL('.', import.meta.url)), '..', 'audit-shots')
mkdirSync(outDir, { recursive: true })

const mime = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.pdb': 'chemical/x-pdb',
}

function serve() {
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      let path = join(root, decodeURIComponent((req.url || '/').split('?')[0]))
      if (path.endsWith('/')) path = join(path, 'index.html')
      if (!existsSync(path) || statSync(path).isDirectory()) {
        path = join(root, 'index.html')
      }
      res.writeHead(200, {
        'Content-Type': mime[extname(path)] || 'application/octet-stream',
      })
      res.end(readFileSync(path))
    })
    server.listen(0, '127.0.0.1', () => {
      resolve({ server, url: `http://127.0.0.1:${server.address().port}/` })
    })
  })
}

const VIEWPORTS = [
  { name: '280', width: 280, height: 653 },
  { name: '375', width: 375, height: 812 },
  { name: '768', width: 768, height: 1024 },
  { name: '1280', width: 1280, height: 800 },
  { name: '1440', width: 1440, height: 900 },
]

const { server, url } = await serve()
const browser = await chromium.launch()

try {
  for (const vp of VIEWPORTS) {
    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      deviceScaleFactor: 1,
      reducedMotion: 'reduce',
    })
    const page = await context.newPage()
    await page.goto(url, { waitUntil: 'networkidle' })
    await page.waitForTimeout(500)
    await page.screenshot({
      path: join(outDir, `${vp.name}-top.png`),
      fullPage: false,
    })
    await page.evaluate(() =>
      document.getElementById('experiments')?.scrollIntoView(),
    )
    await page.waitForTimeout(200)
    await page.screenshot({
      path: join(outDir, `${vp.name}-experiments.png`),
      fullPage: false,
    })
    await page.evaluate(() =>
      document.getElementById('cost')?.scrollIntoView(),
    )
    await page.waitForTimeout(200)
    await page.screenshot({
      path: join(outDir, `${vp.name}-cost.png`),
      fullPage: false,
    })
    await context.close()
  }
} finally {
  await browser.close()
  server.close()
}

console.log('shots written to', outDir)
