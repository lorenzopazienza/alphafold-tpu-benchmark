import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    include: ['3dmol'],
  },
  build: {
    // 3dmol is lazy-loaded; keep the warning threshold above that chunk.
    chunkSizeWarningLimit: 600,
  },
})
