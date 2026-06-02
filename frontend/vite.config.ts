import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    proxy: {
      '/tts': process.env.VITE_TTS_BACKEND || 'http://localhost:8016',
      '/voices': process.env.VITE_TTS_BACKEND || 'http://localhost:8016',
      '/health': process.env.VITE_TTS_BACKEND || 'http://localhost:8016',
      '/api/v1/qwen': process.env.VITE_TTS_BACKEND || 'http://localhost:8013'
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
}); 