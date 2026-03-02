import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    proxy: {
      '/tts': 'http://localhost:8013',
      '/voices': 'http://localhost:8013',
      '/health': 'http://localhost:8013',
      '/api/v1/qwen': 'http://localhost:8013'
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
}); 