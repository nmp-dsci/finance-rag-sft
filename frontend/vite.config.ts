import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Dev: Vite serves the SPA on :5176 and proxies the API to FastAPI on :8084. Built, FastAPI
// serves the SPA from the same origin, so the proxy does not exist there.
const API = process.env.VITE_API_PROXY ?? 'http://localhost:8084';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5176,
    // The house UI lives one level up, in ../.lavish (house.css, project.css, fonts/), and is
    // imported from src/main.tsx. Dev serving needs permission to read it; the build bundles it.
    fs: { allow: ['..'] },
    proxy: {
      '/api': { target: API, changeOrigin: false },
      '/healthz': { target: API, changeOrigin: false },
    },
  },
});
