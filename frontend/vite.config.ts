import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 3000,
    proxy: {
      '/v1': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: true,
      },
      '/models': 'http://127.0.0.1:8000',
      '/chat': 'http://127.0.0.1:8000',
      '/topology': 'http://127.0.0.1:8000',
      '/healthcheck': 'http://127.0.0.1:8000',
    }
  }
});
