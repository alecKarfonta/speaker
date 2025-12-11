import React, { useState, useEffect, useRef } from 'react';
import { Play, Upload, Settings, X, ChevronDown, ChevronUp } from 'lucide-react';
import { toast } from 'react-hot-toast';

interface TTSParams {
  // XTTS params
  tau: number;
  gpt_cond_len: number;
  top_k: number;
  top_p: number;
  decoder_iterations: number;
  split_sentences: boolean;
  // GLM-TTS params
  sampling: number;
  min_token_text_ratio: number;
  max_token_text_ratio: number;
  beam_size: number;
  temperature: number;
  glm_top_p: number;
  repetition_penalty: number;
  sample_method: 'ras' | 'topk';
  // Output format
  output_format: 'raw' | 'wav';
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
  language: string;
  backend: 'glm-tts' | 'xtts';
  params: Partial<TTSParams>;
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
    // XTTS defaults
    tau: 0.6,
    gpt_cond_len: 10,
    top_k: 10,
    top_p: 0.9,
    decoder_iterations: 30,
    split_sentences: true,
    // GLM-TTS defaults
    sampling: 25,
    min_token_text_ratio: 8,
    max_token_text_ratio: 30,
    beam_size: 1,
    temperature: 1.0,
    glm_top_p: 0.8,
    repetition_penalty: 0.1,
    sample_method: 'ras',
    // Output format
    output_format: 'wav'
  });
  const [backend, setBackend] = useState<'xtts' | 'glm-tts'>('glm-tts');
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
      // Build request body based on backend type
      const requestBody: Record<string, any> = {
        text: text,
        voice_name: selectedVoice,
        language: language,
        output_format: advancedParams.output_format,
      };
      
      // Add backend-specific params
      if (backend === 'glm-tts') {
        requestBody.sampling = advancedParams.sampling;
        requestBody.min_token_text_ratio = advancedParams.min_token_text_ratio;
        requestBody.max_token_text_ratio = advancedParams.max_token_text_ratio;
        requestBody.beam_size = advancedParams.beam_size;
        requestBody.temperature = advancedParams.temperature;
        requestBody.top_p = advancedParams.glm_top_p;
        requestBody.repetition_penalty = advancedParams.repetition_penalty;
        requestBody.sample_method = advancedParams.sample_method;
      } else {
        requestBody.tau = advancedParams.tau;
        requestBody.gpt_cond_len = advancedParams.gpt_cond_len;
        requestBody.top_k = advancedParams.top_k;
        requestBody.top_p = advancedParams.top_p;
        requestBody.decoder_iterations = advancedParams.decoder_iterations;
        requestBody.split_sentences = advancedParams.split_sentences;
      }
      
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

      let wavBlob: Blob;
      
      if (advancedParams.output_format === 'wav') {
        // Already WAV format from server
        wavBlob = new Blob([buffer], { type: 'audio/wav' });
      } else {
        // Convert raw float32 data to WAV format
        const float32Array = new Float32Array(buffer);
        const wavBuffer = convertFloat32ToWav(float32Array);
        wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
      }
      
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
        timestamp: new Date().toLocaleTimeString(),
        language,
        backend,
        params: backend === 'glm-tts' ? {
          sampling: advancedParams.sampling,
          temperature: advancedParams.temperature,
          glm_top_p: advancedParams.glm_top_p,
          min_token_text_ratio: advancedParams.min_token_text_ratio,
          max_token_text_ratio: advancedParams.max_token_text_ratio,
          beam_size: advancedParams.beam_size,
          repetition_penalty: advancedParams.repetition_penalty,
          sample_method: advancedParams.sample_method,
        } : {
          tau: advancedParams.tau,
          gpt_cond_len: advancedParams.gpt_cond_len,
          top_k: advancedParams.top_k,
          top_p: advancedParams.top_p,
          decoder_iterations: advancedParams.decoder_iterations,
        }
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
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium text-gray-700">Advanced Parameters</h3>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600">Backend:</span>
                  <select
                    value={backend}
                    onChange={(e) => setBackend(e.target.value as 'xtts' | 'glm-tts')}
                    className="px-2 py-1 border border-gray-300 rounded text-sm"
                  >
                    <option value="glm-tts">GLM-TTS</option>
                    <option value="xtts">XTTS</option>
                  </select>
                </div>
              </div>
              
              {backend === 'glm-tts' ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm text-gray-700 mb-1">
                      Sampling (top-k): {advancedParams.sampling}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="100"
                      value={advancedParams.sampling}
                      onChange={(e) => handleParamChange('sampling', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500">Lower = more deterministic</p>
                  </div>
                  
                  <div>
                    <label className="block text-sm text-gray-700 mb-1">
                      Temperature: {advancedParams.temperature.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="2.0"
                      step="0.1"
                      value={advancedParams.temperature}
                      onChange={(e) => handleParamChange('temperature', parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500">Higher = more random</p>
                  </div>
                  
                  <div>
                    <label className="block text-sm text-gray-700 mb-1">
                      Top-P (nucleus): {advancedParams.glm_top_p.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={advancedParams.glm_top_p}
                      onChange={(e) => handleParamChange('glm_top_p', parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500">Nucleus sampling threshold</p>
                  </div>
                  
                  <div>
                    <label className="block text-sm text-gray-700 mb-1">
                      Min Token Ratio: {advancedParams.min_token_text_ratio}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      value={advancedParams.min_token_text_ratio}
                      onChange={(e) => handleParamChange('min_token_text_ratio', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500">Min audio length</p>
                  </div>
                  
                  <div>
                    <label className="block text-sm text-gray-700 mb-1">
                      Max Token Ratio: {advancedParams.max_token_text_ratio}
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="100"
                      value={advancedParams.max_token_text_ratio}
                      onChange={(e) => handleParamChange('max_token_text_ratio', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500">Max audio length</p>
                  </div>
                  
                  <div>
                    <label className="block text-sm text-gray-700 mb-1">
                      Beam Size: {advancedParams.beam_size}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      value={advancedParams.beam_size}
                      onChange={(e) => handleParamChange('beam_size', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500">Quality vs speed</p>
                  </div>
                  
                  <div>
                    <label className="block text-sm text-gray-700 mb-1">
                      Repetition Penalty: {advancedParams.repetition_penalty.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={advancedParams.repetition_penalty}
                      onChange={(e) => handleParamChange('repetition_penalty', parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500">Prevent repetition</p>
                  </div>
                  
                  <div>
                    <label className="block text-sm text-gray-700 mb-1">
                      Sample Method
                    </label>
                    <select
                      value={advancedParams.sample_method}
                      onChange={(e) => handleParamChange('sample_method', e.target.value as any)}
                      className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                    >
                      <option value="ras">RAS (Repetition-Aware)</option>
                      <option value="topk">Top-K</option>
                    </select>
                    <p className="text-xs text-gray-500">Sampling algorithm</p>
                  </div>
                </div>
              ) : (
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
              )}
              
              <div className="border-t pt-4 mt-4">
                <label className="block text-sm text-gray-700 mb-2">Output Format:</label>
                <div className="flex space-x-4">
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="output_format"
                      value="wav"
                      checked={advancedParams.output_format === 'wav'}
                      onChange={() => handleParamChange('output_format', 'wav' as any)}
                      className="h-4 w-4 text-blue-600"
                    />
                    <span className="ml-2 text-sm">WAV (recommended)</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="output_format"
                      value="raw"
                      checked={advancedParams.output_format === 'raw'}
                      onChange={() => handleParamChange('output_format', 'raw' as any)}
                      className="h-4 w-4 text-blue-600"
                    />
                    <span className="ml-2 text-sm">Raw PCM</span>
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
                  <div className="flex-1">
                    <h3 className="font-medium text-gray-900 truncate max-w-md">{response.text}</h3>
                    <p className="text-sm text-gray-500">
                      Voice: {response.voice} | {response.language?.toUpperCase() || 'EN'} | {response.timestamp}
                    </p>
                    <p className="text-xs text-blue-600 mt-1">
                      {response.backend?.toUpperCase() || 'GLM-TTS'}
                      {response.params && (
                        <span className="text-gray-400 ml-2">
                          {response.backend === 'glm-tts' ? (
                            <>
                              sampling={response.params.sampling} | 
                              temp={response.params.temperature} | 
                              top_p={response.params.glm_top_p} | 
                              beam={response.params.beam_size} | 
                              min_ratio={response.params.min_token_text_ratio} | 
                              max_ratio={response.params.max_token_text_ratio}
                            </>
                          ) : (
                            <>
                              tau={response.params.tau} | 
                              top_k={response.params.top_k} | 
                              top_p={response.params.top_p} | 
                              gpt_cond={response.params.gpt_cond_len}
                            </>
                          )}
                        </span>
                      )}
                    </p>
                  </div>
                  <button 
                    onClick={() => deleteResponse(response.id)}
                    className="text-gray-400 hover:text-red-500 ml-2"
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
