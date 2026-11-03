import { defineConfig } from 'vite';

// `npm run dev` serves the UI on :3000 and proxies the API to a local node
// (SYNAPSE_API, default http://127.0.0.1:52415). `npm run build` writes
// dist/, which the node itself serves at GET /.
const target = process.env.SYNAPSE_API || 'http://127.0.0.1:52415';

export default defineConfig({
  root: '.',
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    port: 3000,
    proxy: {
      '/v1': { target, changeOrigin: true, ws: true },
      '/healthcheck': target,
    },
  },
});
