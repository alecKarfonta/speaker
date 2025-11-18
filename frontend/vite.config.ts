import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    proxy: {
      '/tts': 'http://localhost:8010',
      '/voices': 'http://localhost:8010'
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
}); 