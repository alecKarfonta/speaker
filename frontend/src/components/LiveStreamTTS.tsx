import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, Settings, Volume2, VolumeX, Radio, AlertCircle, Wifi, WifiOff, RotateCcw, Loader2 } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { YouTubeApiService, YouTubeApiError, LiveChatMessage } from '../services/youtubeApi';

interface Voice {
  id: string;
  name: string;
}

interface SuperChat {
  id: string;
  author: string;
  amount: string;
  message: string;
  timestamp: string;
  processed: boolean;
  originalMessage?: LiveChatMessage;
  audioUrl?: string;
  selectedVoice?: string;
  isGenerating?: boolean;
  isPlaying?: boolean;
}

interface StreamSettings {
  videoId: string;
  apiKey: string;
  selectedVoice: string;
  autoTTS: boolean;
  autoGenerate: boolean;
  autoPlay: boolean;
  language: string;
}

function convertFloat32ToWav(float32Array: Float32Array): ArrayBuffer {
  const numChannels = 1; // Mono
  const sampleRate = 24000; // Standard sample rate for TTS
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = float32Array.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // Write WAV header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  const offset = 44;
  for (let i = 0; i < float32Array.length; i++) {
    const sample = Math.max(-1, Math.min(1, float32Array[i]));
    const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    view.setInt16(offset + i * 2, int16, true);
  }

  return buffer;
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

const LiveStreamTTS: React.FC = () => {
  const [voiceList, setVoiceList] = useState<Voice[]>([]);
  const [superChats, setSuperChats] = useState<SuperChat[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [settings, setSettings] = useState<StreamSettings>({
    videoId: '',
    apiKey: '',
    selectedVoice: '',
    autoTTS: true,
    autoGenerate: true,
    autoPlay: false,
    language: 'en'
  });

  const youtubeApiRef = useRef<YouTubeApiService | null>(null);
  const superChatGeneratorRef = useRef<AsyncGenerator<LiveChatMessage[], void, unknown> | null>(null);
  const processedMessages = useRef<Set<string>>(new Set());
  const isStreamActiveRef = useRef<boolean>(false);

  useEffect(() => {
    fetchVoices();
    return () => {
      // Cleanup on component unmount
      disconnectFromStream();
      if (currentAudio) {
        currentAudio.pause();
      }
    };
  }, []);

  const fetchVoices = async () => {
    try {
      const response = await fetch('/voices');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const responseData = await response.json();
      const voiceNames = Array.isArray(responseData) ? responseData : responseData.voices || [];
      
      const voices: Voice[] = voiceNames.map((name: string) => ({
        id: name,
        name: name
      }));
      
      setVoiceList(voices);
      if (voices.length > 0 && !settings.selectedVoice) {
        setSettings(prev => ({ ...prev, selectedVoice: voices[0].name }));
      }
    } catch (error) {
      console.error('Error fetching voices:', error);
      toast.error(`Failed to fetch voices: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const generateTTS = async (text: string, voiceName?: string): Promise<string> => {
    try {
      const requestBody = {
        text: text,
        voice_name: voiceName || settings.selectedVoice,
        language: settings.language
      };
      
      const response = await fetch('/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${await response.text()}`);
      }
      
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength === 0) {
        throw new Error('Received empty audio response from the server');
      }

      const float32Array = new Float32Array(buffer);
      const wavBuffer = convertFloat32ToWav(float32Array);
      const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
      
      return URL.createObjectURL(wavBlob);
    } catch (error) {
      console.error('Error generating TTS:', error);
      throw error;
    }
  };

  const playTTS = async (text: string) => {
    try {
      setIsLoading(true);
      const audioUrl = await generateTTS(text);
      
      if (currentAudio) {
        currentAudio.pause();
      }
      
      const audio = new Audio(audioUrl);
      audio.onended = () => {
        setIsPlaying(false);
        setCurrentAudio(null);
      };
      
      audio.onerror = () => {
        toast.error('Failed to play audio');
        setIsPlaying(false);
        setCurrentAudio(null);
      };
      
      setCurrentAudio(audio);
      setIsPlaying(true);
      await audio.play();
      
    } catch (error) {
      toast.error(`Failed to generate TTS: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setIsPlaying(false);
    } finally {
      setIsLoading(false);
    }
  };

  const stopAudio = () => {
    if (currentAudio) {
      currentAudio.pause();
      setCurrentAudio(null);
      setIsPlaying(false);
    }
  };

  // Generate TTS for a specific superchat
  const generateSuperChatTTS = async (superChatId: string, voiceName?: string) => {
    const superChat = superChats.find(sc => sc.id === superChatId);
    if (!superChat) return;

    // Mark as generating
    setSuperChats(prev => 
      prev.map(sc => sc.id === superChatId ? { ...sc, isGenerating: true } : sc)
    );

    try {
      const audioUrl = await generateTTS(superChat.message, voiceName || superChat.selectedVoice || settings.selectedVoice);
      
      // Update superchat with audio URL and voice
      setSuperChats(prev => 
        prev.map(sc => sc.id === superChatId ? { 
          ...sc, 
          audioUrl, 
          selectedVoice: voiceName || superChat.selectedVoice || settings.selectedVoice,
          isGenerating: false,
          processed: true 
        } : sc)
      );

      // Auto-play if enabled
      if (settings.autoPlay) {
        playSuperChatAudio(superChatId);
      }
    } catch (error) {
      console.error('TTS generation failed for super chat:', error);
      toast.error(`TTS failed for message from ${superChat.author}`);
      
      // Clear generating state
      setSuperChats(prev => 
        prev.map(sc => sc.id === superChatId ? { ...sc, isGenerating: false } : sc)
      );
    }
  };

  // Play audio for a specific superchat
  const playSuperChatAudio = (superChatId: string) => {
    const superChat = superChats.find(sc => sc.id === superChatId);
    if (!superChat?.audioUrl) return;

    // Stop any currently playing audio
    if (currentAudio) {
      currentAudio.pause();
    }

    // Stop any other superchat audio
    setSuperChats(prev => 
      prev.map(sc => ({ ...sc, isPlaying: false }))
    );

    const audio = new Audio(superChat.audioUrl);
    
    audio.onended = () => {
      setCurrentAudio(null);
      setSuperChats(prev => 
        prev.map(sc => sc.id === superChatId ? { ...sc, isPlaying: false } : sc)
      );
    };
    
    audio.onerror = () => {
      toast.error('Failed to play audio');
      setCurrentAudio(null);
      setSuperChats(prev => 
        prev.map(sc => sc.id === superChatId ? { ...sc, isPlaying: false } : sc)
      );
    };

    // Mark as playing
    setSuperChats(prev => 
      prev.map(sc => sc.id === superChatId ? { ...sc, isPlaying: true } : sc)
    );
    
    setCurrentAudio(audio);
    audio.play().catch(error => {
      console.error('Failed to play audio:', error);
      toast.error('Failed to play audio');
    });
  };

  // Stop audio for a specific superchat
  const stopSuperChatAudio = (superChatId: string) => {
    if (currentAudio) {
      currentAudio.pause();
      setCurrentAudio(null);
    }
    
    setSuperChats(prev => 
      prev.map(sc => sc.id === superChatId ? { ...sc, isPlaying: false } : sc)
    );
  };

  // Regenerate TTS with a different voice
  const regenerateSuperChatTTS = async (superChatId: string, newVoice: string) => {
    await generateSuperChatTTS(superChatId, newVoice);
  };

  // Real YouTube API integration
  const startSuperChatStream = async () => {
    if (!youtubeApiRef.current || !isStreamActiveRef.current) {
      return;
    }

    try {
      const superChatGenerator = youtubeApiRef.current.getSuperChats(settings.videoId);
      superChatGeneratorRef.current = superChatGenerator;

      // Process super chats as they come in
      for await (const messages of superChatGenerator) {
        if (!isStreamActiveRef.current) {
          break; // Stop if disconnected
        }

        const newSuperChats: SuperChat[] = messages.map(message => ({
          id: message.id,
          author: message.authorDetails.displayName,
          amount: YouTubeApiService.getSuperChatAmount(message),
          message: YouTubeApiService.extractMessageText(message),
          timestamp: new Date(message.snippet.publishedAt).toLocaleTimeString(),
          processed: false,
          originalMessage: message,
          selectedVoice: settings.selectedVoice
        }));

        if (newSuperChats.length > 0) {
          console.log(`Received ${newSuperChats.length} new super chats`);
          
          // Add new super chats to the list
          setSuperChats(prev => [...newSuperChats, ...prev]);

          // Process TTS based on settings
          for (const chat of newSuperChats) {
            if (!processedMessages.current.has(chat.id) && chat.message.trim()) {
              processedMessages.current.add(chat.id);
              
              if (settings.autoGenerate) {
                // Auto-generate TTS (will auto-play if autoPlay is enabled)
                await generateSuperChatTTS(chat.id);
              } else if (settings.autoTTS) {
                // Legacy auto TTS mode - generate and play immediately
                try {
                  await playTTS(chat.message);
                  
                  // Mark as processed
                  setSuperChats(prev => 
                    prev.map(c => c.id === chat.id ? { ...c, processed: true } : c)
                  );
                } catch (ttsError) {
                  console.error('TTS generation failed for super chat:', ttsError);
                  toast.error(`TTS failed for message from ${chat.author}`);
                }
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Super chat stream error:', error);
      
      if (error instanceof YouTubeApiError) {
        let errorMessage = 'Connection to live stream failed';
        
        switch (error.code) {
          case 'quotaExceeded':
            errorMessage = 'YouTube API quota exceeded. Please try again later.';
            break;
          case 'forbidden':
            errorMessage = 'API key access denied. Check your permissions.';
            break;
          case 'notFound':
            errorMessage = 'Video not found or not a live stream.';
            break;
          case 'liveChatNotAvailable':
            errorMessage = 'Live chat is not available for this stream.';
            break;
          case 'streamEnded':
            errorMessage = 'Live stream has ended.';
            break;
          case 'rateLimitExceeded':
            errorMessage = 'Rate limit exceeded. Please wait before reconnecting.';
            break;
          case 'tooManyErrors':
            errorMessage = 'Too many connection errors. Please check your settings.';
            break;
          default:
            errorMessage = error.message;
        }
        
        setConnectionError(errorMessage);
        toast.error(errorMessage);
      } else {
        const errorMessage = 'Unexpected error occurred while connecting to stream';
        setConnectionError(errorMessage);
        toast.error(errorMessage);
      }
      
      // Auto-disconnect on error
      disconnectFromStream();
    }
  };

  const connectToStream = async () => {
    if (!settings.videoId.trim() || !settings.apiKey.trim()) {
      toast.error('Please provide both Video ID and API Key');
      return;
    }

    if (!settings.selectedVoice) {
      toast.error('Please select a voice');
      return;
    }

    setIsConnecting(true);
    setConnectionError(null);

    try {
      // Initialize YouTube API service
      youtubeApiRef.current = new YouTubeApiService(settings.apiKey);
      
      // Validate API key
      toast.loading('Validating API key...', { id: 'validation' });
      const isValidKey = await youtubeApiRef.current.validateApiKey();
      toast.dismiss('validation');
      
      if (!isValidKey) {
        throw new YouTubeApiError(
          'Invalid API key. Please check your YouTube Data API key.',
          403,
          'invalidApiKey'
        );
      }

      // Validate video and get live chat ID
      toast.loading('Connecting to live stream...', { id: 'connecting' });
      await youtubeApiRef.current.getLiveChatId(settings.videoId);
      toast.dismiss('connecting');

      // Set connection state
      setIsConnected(true);
      isStreamActiveRef.current = true;
      processedMessages.current.clear();
      
      toast.success('Connected to live stream successfully!');
      
      // Start monitoring super chats
      startSuperChatStream();

    } catch (error) {
      console.error('Connection failed:', error);
      toast.dismiss(); // Clear any loading toasts
      
      let errorMessage = 'Failed to connect to live stream';
      
      if (error instanceof YouTubeApiError) {
        switch (error.code) {
          case 'invalidApiKey':
          case 'forbidden':
            errorMessage = 'Invalid API key or insufficient permissions. Please check your YouTube Data API key.';
            break;
          case 'notFound':
            errorMessage = 'Video not found. Please check the video ID.';
            break;
          case 'notLiveStream':
            errorMessage = 'This video is not a live stream.';
            break;
          case 'liveChatNotAvailable':
            errorMessage = 'Live chat is not available for this stream. It may be disabled or the stream may have ended.';
            break;
          case 'quotaExceeded':
            errorMessage = 'YouTube API quota exceeded. Please try again later.';
            break;
          case 'badRequest':
            errorMessage = 'Invalid request. Please check your video ID and API key.';
            break;
          default:
            errorMessage = error.message;
        }
      }
      
      setConnectionError(errorMessage);
      toast.error(errorMessage);
      
      // Clean up
      youtubeApiRef.current = null;
      isStreamActiveRef.current = false;
    } finally {
      setIsConnecting(false);
    }
  };

  const disconnectFromStream = () => {
    // Stop the stream
    isStreamActiveRef.current = false;
    
    // Clean up generator
    if (superChatGeneratorRef.current) {
      superChatGeneratorRef.current = null;
    }
    
    // Clean up API service
    youtubeApiRef.current = null;
    
    // Reset state
    setIsConnected(false);
    setConnectionError(null);
    processedMessages.current.clear();
    
    toast.success('Disconnected from live stream');
  };



  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold flex items-center">
            {isConnecting ? (
              <Wifi className="mr-2 text-yellow-500 animate-pulse" size={24} />
            ) : isConnected ? (
              <Wifi className="mr-2 text-green-500" size={24} />
            ) : (
              <WifiOff className="mr-2 text-gray-400" size={24} />
            )}
            Live Stream TTS
            {isConnecting && (
              <span className="ml-2 px-2 py-1 bg-yellow-100 text-yellow-800 text-xs rounded-full">
                Connecting...
              </span>
            )}
            {isConnected && (
              <span className="ml-2 px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                Connected
              </span>
            )}
          </h2>
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center text-sm text-blue-600 hover:text-blue-800"
          >
            <Settings size={16} className="mr-1" />
            Settings
          </button>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="bg-gray-50 p-4 rounded-md mb-4 space-y-4">
            <h3 className="font-medium text-gray-700">Stream Settings</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  YouTube Video ID
                </label>
                <input
                  type="text"
                  value={settings.videoId}
                  onChange={(e) => setSettings(prev => ({ ...prev, videoId: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., dQw4w9WgXcQ"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  YouTube API Key
                </label>
                <input
                  type="password"
                  value={settings.apiKey}
                  onChange={(e) => setSettings(prev => ({ ...prev, apiKey: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Your YouTube Data API key"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Voice
                </label>
                <select
                  value={settings.selectedVoice}
                  onChange={(e) => setSettings(prev => ({ ...prev, selectedVoice: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {voiceList.length > 0 ? (
                    voiceList.map(voice => (
                      <option key={voice.name} value={voice.name}>{voice.name}</option>
                    ))
                  ) : (
                    <option value="" disabled>No voices available</option>
                  )}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Language
                </label>
                <select
                  value={settings.language}
                  onChange={(e) => setSettings(prev => ({ ...prev, language: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="en">English (en)</option>
                  <option value="es">Spanish (es)</option>
                  <option value="fr">French (fr)</option>
                  <option value="de">German (de)</option>
                  <option value="it">Italian (it)</option>
                  <option value="pt">Portuguese (pt)</option>
                  <option value="ru">Russian (ru)</option>
                  <option value="zh">Chinese (zh)</option>
                  <option value="ja">Japanese (ja)</option>
                </select>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="autoGenerate"
                  checked={settings.autoGenerate}
                  onChange={(e) => setSettings(prev => ({ ...prev, autoGenerate: e.target.checked }))}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label htmlFor="autoGenerate" className="ml-2 block text-sm text-gray-700">
                  Automatically generate TTS for new super chats
                </label>
              </div>
              
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="autoPlay"
                  checked={settings.autoPlay}
                  onChange={(e) => setSettings(prev => ({ ...prev, autoPlay: e.target.checked }))}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  disabled={!settings.autoGenerate}
                />
                <label htmlFor="autoPlay" className="ml-2 block text-sm text-gray-700">
                  Automatically play generated TTS audio
                </label>
              </div>
              
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="autoTTS"
                  checked={settings.autoTTS}
                  onChange={(e) => setSettings(prev => ({ ...prev, autoTTS: e.target.checked }))}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label htmlFor="autoTTS" className="ml-2 block text-sm text-gray-700">
                  Legacy mode: Generate and play TTS immediately (overrides above settings)
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {connectionError && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <div className="flex items-start">
              <AlertCircle className="text-red-600 mt-0.5 mr-2" size={16} />
              <div>
                <h4 className="text-sm font-medium text-red-800">Connection Error</h4>
                <p className="text-sm text-red-700 mt-1">{connectionError}</p>
              </div>
            </div>
          </div>
        )}

        {/* Connection Controls */}
        <div className="flex items-center space-x-4">
          {!isConnected ? (
            <button
              onClick={connectToStream}
              disabled={isConnecting}
              className={`px-4 py-2 text-white rounded-md focus:outline-none focus:ring-2 ${
                isConnecting 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-green-600 hover:bg-green-700 focus:ring-green-500'
              }`}
            >
              {isConnecting ? 'Connecting...' : 'Connect to Stream'}
            </button>
          ) : (
            <button
              onClick={disconnectFromStream}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500"
            >
              Disconnect
            </button>
          )}
          
          {isPlaying && (
            <button
              onClick={stopAudio}
              className="flex items-center px-3 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
            >
              {isLoading ? <Volume2 className="animate-pulse" size={16} /> : <VolumeX size={16} />}
              <span className="ml-1">Stop Audio</span>
            </button>
          )}
        </div>
      </div>

      {/* API Setup Instructions */}
      {!isConnected && !isConnecting && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start">
            <AlertCircle className="text-blue-600 mt-0.5" size={20} />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-blue-800">YouTube API Setup Required</h3>
              <p className="text-sm text-blue-700 mt-1">
                To connect to real YouTube live streams, you need a YouTube Data API v3 key. 
                Get one from the <a href="https://console.developers.google.com/" target="_blank" rel="noopener noreferrer" className="underline">Google Cloud Console</a>.
                Make sure to enable the YouTube Data API v3 for your project.
              </p>
              <div className="mt-2 text-xs text-blue-600">
                <strong>Required permissions:</strong> YouTube Data API v3 read access
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Super Chats */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-bold mb-4">Super Chats</h2>
        
        {superChats.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            {isConnected ? 'Waiting for super chats...' : 'Connect to a stream to see super chats'}
          </div>
        ) : (
          <div className="space-y-4">
            {superChats.map((superChat) => (
              <div 
                key={superChat.id} 
                className={`border rounded-lg p-4 ${
                  superChat.processed ? 'bg-green-50 border-green-200' : 'bg-white border-gray-200'
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="font-medium text-gray-900">{superChat.author}</h3>
                    <p className="text-sm text-gray-500">
                      {superChat.amount} | {superChat.timestamp}
                      {superChat.processed && (
                        <span className="ml-2 text-green-600">✓ Processed</span>
                      )}
                      {superChat.isGenerating && (
                        <span className="ml-2 text-blue-600">⏳ Generating...</span>
                      )}
                    </p>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {/* Voice selector for individual superchat */}
                    <select
                      value={superChat.selectedVoice || settings.selectedVoice}
                      onChange={(e) => {
                        const newVoice = e.target.value;
                        setSuperChats(prev => 
                          prev.map(sc => sc.id === superChat.id ? { ...sc, selectedVoice: newVoice } : sc)
                        );
                      }}
                      className="text-xs px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                      disabled={superChat.isGenerating}
                    >
                      {voiceList.map(voice => (
                        <option key={voice.name} value={voice.name}>{voice.name}</option>
                      ))}
                    </select>
                    
                    {/* Audio controls */}
                    {superChat.audioUrl ? (
                      <div className="flex items-center space-x-1">
                        {superChat.isPlaying ? (
                          <button
                            onClick={() => stopSuperChatAudio(superChat.id)}
                            className="flex items-center px-2 py-1 bg-red-50 text-red-700 rounded hover:bg-red-100"
                          >
                            <Pause size={14} className="mr-1" />
                            Stop
                          </button>
                        ) : (
                          <button
                            onClick={() => playSuperChatAudio(superChat.id)}
                            className="flex items-center px-2 py-1 bg-green-50 text-green-700 rounded hover:bg-green-100"
                          >
                            <Play size={14} className="mr-1" />
                            Play
                          </button>
                        )}
                        
                        <button
                          onClick={() => regenerateSuperChatTTS(superChat.id, superChat.selectedVoice || settings.selectedVoice)}
                          className="flex items-center px-2 py-1 bg-blue-50 text-blue-700 rounded hover:bg-blue-100"
                          disabled={superChat.isGenerating}
                        >
                          {superChat.isGenerating ? (
                            <Loader2 size={14} className="mr-1 animate-spin" />
                          ) : (
                            <RotateCcw size={14} className="mr-1" />
                          )}
                          Regenerate
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => generateSuperChatTTS(superChat.id)}
                        className="flex items-center px-3 py-1 bg-blue-50 text-blue-700 rounded hover:bg-blue-100"
                        disabled={superChat.isGenerating}
                      >
                        {superChat.isGenerating ? (
                          <>
                            <Loader2 size={16} className="mr-1 animate-spin" />
                            Generating...
                          </>
                        ) : (
                          <>
                            <Play size={16} className="mr-1" />
                            Generate TTS
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>
                
                <p className="text-gray-700">{superChat.message}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveStreamTTS;
