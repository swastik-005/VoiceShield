import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8080",
      "/models": "http://127.0.0.1:8080",
      "/reference": "http://127.0.0.1:8080",
      "/stream": { target: "ws://127.0.0.1:8080", ws: true },
    },
  },
});
