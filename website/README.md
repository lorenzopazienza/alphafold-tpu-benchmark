# AlphaFold TPU Benchmark · Project Website

Phase 4 showcase for the Stanford ME344 AlphaFold / TPU scaling study.

**Hardware under test (AF2):** Google Colab Intel Xeon CPU (2 vCPU); Google Colab NVIDIA Tesla T4; Stanford GKE TPU v5e-8 (tpu-v5-lite-podslice, 2×4, 8 chips).

**Repo:** [lorenzopazienza/alphafold-tpu-benchmark](https://github.com/lorenzopazienza/alphafold-tpu-benchmark)

## Stack

- Vite + React 19
- Tailwind CSS v4
- 3Dmol.js (interactive ubiquitin viewer)
- Recharts (confidence plot)

## Develop

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
npm run preview
```

## Layout audit

Runs a production build, then Playwright checks across phone / tablet / desktop viewports (overflow, menu, sections, assets):

```bash
npm run audit:layout
```

## Deploy (Vercel)

From the **repo root** (recommended): import the GitHub repo in Vercel. Root `vercel.json` builds `website/` and publishes `website/dist`.

Or set the Vercel project **Root Directory** to `website` and deploy that folder alone (`website/vercel.json` still applies cache headers).

```bash
# local production check
npm run build && npm run preview
```

Static assets for the structure and charts live under `public/` (copied from the repo’s `structure/` and `figures/` folders).
