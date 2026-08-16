import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Built output is committed to stp_server/manage/ui and served by the plugin
// at /stp-v1/manage/. base './' keeps assets relative so the same build works
// when the Stimma app proxies it under a different prefix.
export default defineConfig({
  plugins: [vue()],
  base: './',
  build: {
    outDir: '../stp_server/manage/ui',
    emptyOutDir: true,
    assetsDir: 'assets',
    sourcemap: false,
  },
  server: {
    port: 5178,
    proxy: {
      '/stp-v1': { target: process.env.COMFY || 'http://127.0.0.1:8188', changeOrigin: true },
    },
  },
})
