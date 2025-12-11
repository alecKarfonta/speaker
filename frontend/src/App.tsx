import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import TTSWorkspace from './components/tts/TTSWorkspace';

// Placeholder components for other routes
const LiveStream: React.FC = () => (
  <div className="flex items-center justify-center h-screen bg-background text-text-secondary">
    Live Stream (Coming Soon)
  </div>
);

const VoiceLibrary: React.FC = () => (
  <div className="flex items-center justify-center h-screen bg-background text-text-secondary">
    Voice Library (Coming Soon)
  </div>
);

const SettingsPage: React.FC = () => (
  <div className="flex items-center justify-center h-screen bg-background text-text-secondary">
    Settings (Coming Soon)
  </div>
);

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<TTSWorkspace />} />
        <Route path="/stream" element={<LiveStream />} />
        <Route path="/voices" element={<VoiceLibrary />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
      
      <Toaster 
        position="bottom-right"
        toastOptions={{
          style: {
            background: 'var(--bg-secondary)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-default)',
          },
          success: {
            iconTheme: {
              primary: 'var(--success)',
              secondary: 'white',
            },
          },
          error: {
            iconTheme: {
              primary: 'var(--error)',
              secondary: 'white',
            },
          },
        }}
      />
    </Router>
  );
}

export default App;
