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
      '/token': 'http://localhost:8080',
      '/search': 'http://localhost:8080',
      '/audio': 'http://localhost:8080',
      '/text': 'http://localhost:8080',
      '/reset': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
    }
  }
})
