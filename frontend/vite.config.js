import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: 'localhost',
    port: 5173,
    strictPort: true,
    proxy: {
      // Forward /api calls to FastAPI backend
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // Forward /feedback to FastAPI backend
      '/feedback': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // Forward /auth API calls (like /auth/token, /auth/me) but let React Router handle /auth/callback
      '/auth': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        bypass: (req) => {
          if (req.url && req.url.startsWith('/auth/callback')) {
            return '/index.html';
          }
        },
      },
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
  },
});
