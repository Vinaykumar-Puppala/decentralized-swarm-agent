import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `npm run dev` proxies the API and the AG-UI endpoint to the FastAPI server; `npm run build` emits dist/ for FastAPI to serve.
const backend = process.env.SWARM_API ?? 'http://127.0.0.1:8000'
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': backend, '/agui': backend } },
  build: { outDir: 'dist', sourcemap: false },
})
