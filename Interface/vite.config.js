import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true, // bind 0.0.0.0 so other devices on the same Wi-Fi can reach it
    allowedHosts: true, // accept the ngrok tunnel domain (random subdomain)
    // Proxy the API + WebSockets to the backend so the whole app lives behind
    // ONE origin — a single ngrok HTTPS tunnel to :3000 then serves UI + API,
    // and getUserMedia (mic) works because the origin is https.
    proxy: {
      '/v1': { target: 'http://localhost:8000', changeOrigin: true, ws: true },
      '/v2': { target: 'http://localhost:8000', changeOrigin: true, ws: true },
      '/healthz': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
