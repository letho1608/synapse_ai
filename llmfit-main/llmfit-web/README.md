# llmfit-web

React + Vite frontend for the llmfit local web dashboard.

## Development

```sh
npm ci
npm run dev
```

This starts Vite on `http://127.0.0.1:5173` and proxies `/api/*` to `http://127.0.0.1:8787`.

## Build

```sh
npm run build
```

Build output is written to `llmfit-web/dist` and embedded into `llmfit serve` at compile time.
