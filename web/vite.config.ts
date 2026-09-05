import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dashboard talks to the FastAPI detector service (pipeline/api.py, port 8010).
// Proxying through Vite keeps everything same-origin in dev, so no CORS config
// and no hardcoded hostnames in the components.
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8010', changeOrigin: true },
      // Alerts stream over a websocket; `ws: true` is what makes Vite forward the upgrade.
      '/ws': { target: 'ws://127.0.0.1:8010', ws: true },
      // Clips are served by the API so the player can seek into the event interval.
      '/clips': { target: 'http://127.0.0.1:8010', changeOrigin: true },
    },
  },
  plugins: [react()],
})
