import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    proxy: {
      '/tts': 'http://tts_api:8000',
      '/voices': 'http://tts_api:8000'
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
}); 