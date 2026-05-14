import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import path from "path"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      // Proxy all remaining requests to the Python backend that aren't handling frontend
      // Using 127.0.0.1 instead of localhost to avoid IPv6 ECONNREFUSED on Windows
      '/token': 'http://127.0.0.1:8080',
      '/search': 'http://127.0.0.1:8080',
      '/audio': 'http://127.0.0.1:8080',
      '/text': 'http://127.0.0.1:8080',
      '/reset': 'http://127.0.0.1:8080',
      '/health': 'http://127.0.0.1:8080',
      '/call': 'http://127.0.0.1:8080',
      '/twilio': 'http://127.0.0.1:8080',
      '/cache': 'http://127.0.0.1:8080',
      '/sessions': 'http://127.0.0.1:8080',
      '/metrics': 'http://127.0.0.1:8080',
      '/ws': {
        target: 'ws://127.0.0.1:8080',
        ws: true,
      },
    }
  }
})
