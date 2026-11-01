import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE_PATH || '/video_to_data/',
  build: { chunkSizeWarningLimit: 650 },
});
