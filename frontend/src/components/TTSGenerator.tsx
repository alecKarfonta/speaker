import React, { useState, useEffect, useRef } from 'react';
import { Play, Upload, Settings, X, ChevronDown, ChevronUp } from 'lucide-react';
import { toast } from 'react-hot-toast';

interface TTSParams {
  tau: number;
  gpt_cond_len: number;
  top_k: number;
  top_p: number;
  decoder_iterations: number;
  split_sentences: boolean;
}

interface Voice {
  id: string;
  name: string;
}

interface TTSResponse {
  id: number;
  text: string;
  voice: string;
  audioUrl: string;
  timestamp: string;
}

interface UploadData {
  voiceName: string;
  file: File | null;
}

interface AudioRefs {
  [key: number]: HTMLAudioElement;
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
  // "RIFF" chunk descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // "fmt " sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // audio format (1 = PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // "data" sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Write audio data
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

const TTSGenerator: React.FC = () => {
  const [text, setText] = useState('');
  const [voiceList, setVoiceList] = useState<Voice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [language, setLanguage] = useState('en');
  const [responses, setResponses] = useState<TTSResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [advancedParams, setAdvancedParams] = useState<TTSParams>({
    tau: 0.6,
    gpt_cond_len: 10,
    top_k: 10,
    top_p: 0.9,
    decoder_iterations: 30,
    split_sentences: true
  });
  const [uploadData, setUploadData] = useState<UploadData>({
    voiceName: '',
    file: null
  });
  
  const audioRefs = useRef<AudioRefs>({});
  
  useEffect(() => {
    fetchVoices();
  }, []);
  
  const fetchVoices = async () => {
    try {
      console.log('Fetching voices...');
      const response = await fetch('/voices');
      console.log('Response status:', response.status);
      console.log('response = ', response);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const responseData = await response.json();
      console.log('Fetched voices:', responseData);
      
      // Handle both array format and object with voices property
      const voiceNames = Array.isArray(responseData) ? responseData : responseData.voices || [];
      
      // Convert array of strings to array of Voice objects
      const voices: Voice[] = voiceNames.map((name: string) => ({
        id: name,
        name: name
      }));
      
      setVoiceList(voices);
      if (voices.length > 0 && !selectedVoice) {
        setSelectedVoice(voices[0].name);
      }
    } catch (error) {
      console.error('Error fetching voices:', error);
      toast.error(`Failed to fetch voices: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };
  
  const handleParamChange = (param: keyof TTSParams, value: number | boolean) => {
    setAdvancedParams(prev => ({
      ...prev,
      [param]: value
    }));
  };
  
  const handleSubmit = async () => {
    if (!text.trim() || !selectedVoice) {
      toast.error('Please enter text and select a voice');
      return;
    }
    
    setLoading(true);
    
    try {
      const requestBody = {
        text: text,
        voice_name: selectedVoice,
        language: language,
        ...advancedParams
      };
      
      console.log('Sending request:', requestBody);
      
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
      
      // Get the ArrayBuffer instead of Blob for more control
      const buffer = await response.arrayBuffer();
      console.log('Received response with size:', buffer.byteLength);
      
      if (buffer.byteLength === 0) {
        throw new Error('Received empty audio response from the server');
      }

      // Convert raw float32 data to WAV format
      const float32Array = new Float32Array(buffer);
      const wavBuffer = convertFloat32ToWav(float32Array);
      const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
      
      // Create a URL for the blob
      const audioUrl = URL.createObjectURL(wavBlob);
      console.log('Created audio URL:', audioUrl);
      
      // Create an audio element to test the format
      const audioElement = new Audio();
      audioElement.src = audioUrl;
      
      // Test if audio can be played
      await new Promise<void>((resolve, reject) => {
        audioElement.oncanplaythrough = () => {
          console.log("Audio can play through successfully");
          resolve();
        };
        
        audioElement.onerror = (e) => {
          console.error("Audio test error:", audioElement.error);
          console.error("Audio element state:", {
            networkState: audioElement.networkState,
            readyState: audioElement.readyState,
            error: audioElement.error,
            src: audioElement.src
          });
          reject(new Error(`Audio test failed: ${audioElement.error?.message || 'Unknown error'}`));
        };
        
        // Set a timeout in case oncanplaythrough doesn't fire
        setTimeout(() => {
          // If we haven't resolved yet, but also don't have an error, assume it's ok
          if (audioElement.error === null) {
            console.log("Audio test timed out but no error detected");
            resolve();
          } else {
            reject(new Error("Audio test timed out"));
          }
        }, 2000);
        
        audioElement.load();
      });
      
      const newResponse: TTSResponse = {
        id: Date.now(),
        text,
        voice: selectedVoice,
        audioUrl,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setResponses(prev => [newResponse, ...prev]);
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      toast.error(`Failed to generate speech: ${errorMessage}`);
      console.error('Error generating speech:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const validateVoiceName = (name: string): boolean => {
    return /^[a-zA-Z0-9_]+$/.test(name);
  };
  
  const handleUploadVoice = async () => {
    if (!uploadData.voiceName || !uploadData.file) {
      toast.error('Please provide both voice name and file');
      return;
    }

    if (!validateVoiceName(uploadData.voiceName)) {
      toast.error('Voice name can only contain letters, numbers, and underscores');
      return;
    }
    
    try {
      const formData = new FormData();
      formData.append('file', uploadData.file, uploadData.file.name);
      
      const response = await fetch(`/voices?voice_name=${uploadData.voiceName}`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - let the browser set it with the boundary
        headers: {
          // Remove any Content-Type header to let the browser set it automatically
        }
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server responded with ${response.status}: ${errorText}`);
      }
      
      toast.success('Voice uploaded successfully!');
      setUploadData({ voiceName: '', file: null });
      setShowUpload(false);
      fetchVoices();
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      toast.error(`Failed to upload voice: ${errorMessage}`);
      console.error('Upload error:', error);
    }
  };
  
  const playAudio = (id: number) => {
    const responseItem = responses.find((r: TTSResponse) => r.id === id);
    if (!responseItem) {
      toast.error('Audio not found');
      return;
    }
    
    // First try to play using the audio ref
    if (audioRefs.current[id]) {
      try {
        // Ensure the audio element has loaded the source
        audioRefs.current[id].load();
        
        const playPromise = audioRefs.current[id].play();
        if (playPromise !== undefined) {
          playPromise.catch((err: Error) => {
            console.error('Error playing audio from ref:', err);
            
            // Fallback: create a new Audio element and try to play
            const audio = new Audio();
            audio.src = responseItem.audioUrl;
            audio.load(); // Explicitly load before playing
            
            audio.play().catch(fallbackErr => {
              console.error('Fallback audio play also failed:', fallbackErr);
              toast.error('Failed to play audio: ' + err.message);
            });
          });
        }
      } catch (err) {
        console.error('Error playing audio:', err);
        toast.error('Failed to play audio');
      }
    } else {
      // If ref not available, use direct Audio object
      const audio = new Audio();
      audio.src = responseItem.audioUrl;
      audio.load(); // Explicitly load before playing
      
      audio.play().catch(err => {
        console.error('Error playing audio directly:', err);
        toast.error('Failed to play audio: ' + err.message);
      });
    }
  };
  
  const deleteResponse = (id: number) => {
    setResponses(prev => prev.filter(response => response.id !== id));
  };
  
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Text to Synthesize
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={4}
              placeholder="Enter text to convert to speech..."
              maxLength={1000}
            />
            <div className="text-xs text-gray-500 text-right mt-1">
              {text.length}/1000 characters
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Voice
              </label>
              <select
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value)}
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
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
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
          
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center text-sm text-blue-600 hover:text-blue-800"
            >
              <Settings size={16} className="mr-1" />
              {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
              {showAdvanced ? <ChevronUp size={16} className="ml-1" /> : <ChevronDown size={16} className="ml-1" />}
            </button>
            
            <button
              type="button"
              onClick={() => setShowUpload(!showUpload)}
              className="flex items-center text-sm text-blue-600 hover:text-blue-800"
            >
              <Upload size={16} className="mr-1" />
              {showUpload ? 'Hide' : 'Upload'} Voice
              {showUpload ? <ChevronUp size={16} className="ml-1" /> : <ChevronDown size={16} className="ml-1" />}
            </button>
          </div>
          
          {showAdvanced && (
            <div className="bg-gray-50 p-4 rounded-md space-y-4">
              <h3 className="font-medium text-gray-700">Advanced Parameters</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-700 mb-1">
                    Temperature (tau): {advancedParams.tau}
                  </label>
                  <input
                    type="range"
                    min="-1"
                    max="1"
                    step="0.1"
                    value={advancedParams.tau}
                    onChange={(e) => handleParamChange('tau', parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm text-gray-700 mb-1">
                    GPT Conditioning Length: {advancedParams.gpt_cond_len}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    value={advancedParams.gpt_cond_len}
                    onChange={(e) => handleParamChange('gpt_cond_len', parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm text-gray-700 mb-1">
                    Top K: {advancedParams.top_k}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    value={advancedParams.top_k}
                    onChange={(e) => handleParamChange('top_k', parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm text-gray-700 mb-1">
                    Top P: {advancedParams.top_p}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={advancedParams.top_p}
                    onChange={(e) => handleParamChange('top_p', parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm text-gray-700 mb-1">
                    Decoder Iterations: {advancedParams.decoder_iterations}
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={advancedParams.decoder_iterations}
                    onChange={(e) => handleParamChange('decoder_iterations', parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
                
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="split_sentences"
                    checked={advancedParams.split_sentences}
                    onChange={(e) => handleParamChange('split_sentences', e.target.checked)}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <label htmlFor="split_sentences" className="ml-2 block text-sm text-gray-700">
                    Split Sentences
                  </label>
                </div>
              </div>
            </div>
          )}
          
          {showUpload && (
            <div className="bg-gray-50 p-4 rounded-md">
              <h3 className="font-medium text-gray-700 mb-3">Upload New Voice</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Voice Name
                  </label>
                  <input
                    type="text"
                    value={uploadData.voiceName}
                    onChange={(e) => setUploadData(prev => ({ ...prev, voiceName: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., new_voice_name"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Use only letters, numbers, and underscores
                  </p>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    WAV File
                  </label>
                  <input
                    type="file"
                    accept=".wav"
                    onChange={(e) => setUploadData(prev => ({ ...prev, file: e.target.files?.[0] || null }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                
                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={handleUploadVoice}
                    className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500"
                  >
                    Upload Voice
                  </button>
                </div>
              </div>
            </div>
          )}
          
          <div className="flex justify-center">
            <button
              type="button"
              onClick={handleSubmit}
              disabled={loading}
              className={`px-6 py-3 rounded-md shadow-sm text-white font-medium ${
                loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {loading ? 'Generating...' : 'Generate Speech'}
            </button>
          </div>
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-bold mb-4">Generated Speech History</h2>
        
        {responses.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No speech generated yet.
          </div>
        ) : (
          <div className="space-y-4">
            {responses.map((response) => (
              <div 
                key={response.id} 
                className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50"
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="font-medium text-gray-900 truncate max-w-md">{response.text}</h3>
                    <p className="text-sm text-gray-500">
                      Voice: {response.voice} | {response.timestamp}
                    </p>
                  </div>
                  <button 
                    onClick={() => deleteResponse(response.id)}
                    className="text-gray-400 hover:text-red-500"
                  >
                    <X size={18} />
                  </button>
                </div>
                
                <div className="flex items-center">
                  <button
                    onClick={() => playAudio(response.id)}
                    className="flex items-center px-3 py-1 bg-blue-50 text-blue-700 rounded hover:bg-blue-100"
                  >
                    <Play size={16} className="mr-1" />
                    Play
                  </button>
                  <audio
                    ref={ref => {
                      if (ref) audioRefs.current[response.id] = ref;
                    }}
                    src={response.audioUrl}
                    controls
                    className="ml-4 w-64"
                    onError={(e: React.SyntheticEvent<HTMLAudioElement>) => {
                      console.error('Audio element error:', e);
                      // Get more specific error information
                      const audioEl = e.target as HTMLAudioElement;
                      console.error('Audio error details:', {
                        error: audioEl.error,
                        networkState: audioEl.networkState,
                        readyState: audioEl.readyState
                      });
                      toast.error('Error loading audio. See console for details.');
                    }}
                    preload="metadata"
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default TTSGenerator;
