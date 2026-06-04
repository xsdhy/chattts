import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发期把 /api 与 /health 代理到后端开发端口（默认 8001），支持 HMR。
// 生产构建产物输出到 dist/，由 Docker 复制为 /app/static，再由 FastAPI 托管。
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8001",
      "/health": "http://127.0.0.1:8001",
    },
  },
});
