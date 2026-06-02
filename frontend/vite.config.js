import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/v1': {
        target: 'http://127.0.0.1:52415',
        changeOrigin: true,
        secure: false,
      },
      '/_static': {
        target: 'http://localhost:5173',
        rewrite: (path) => path.replace(/^\/_static/, '')
      }
    }
  }
});
