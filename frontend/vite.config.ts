import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    proxy: {
      '/tts': 'http://localhost:8016',
      '/voices': 'http://localhost:8016',
      '/health': 'http://localhost:8016'
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
}); 