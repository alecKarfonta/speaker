import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Navigation from './components/Navigation';
import TTSGenerator from './components/TTSGenerator';
import LiveStreamTTS from './components/LiveStreamTTS';

function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen bg-gray-50">
        <header className="bg-blue-600 text-white p-4 shadow-md">
          <h1 className="text-2xl font-bold">Text-to-Speech API Interface</h1>
        </header>
        
        <Navigation />
        
        <main className="flex-1 p-4 max-w-4xl mx-auto w-full">
          <Routes>
            <Route path="/" element={<TTSGenerator />} />
            <Route path="/stream" element={<LiveStreamTTS />} />
          </Routes>
        </main>
        
        <footer className="bg-gray-100 p-4 text-center text-gray-600 text-sm">
          TTS Frontend Interface | API is available at /docs for Swagger UI
        </footer>
        
        <Toaster position="top-right" />
      </div>
    </Router>
  );
}

export default App;