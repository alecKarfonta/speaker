import React, { useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  Settings2, 
  ChevronDown, 
  ChevronUp,
  Sparkles,
  Zap,
  Target,
  Volume2,
  Globe,
  Cpu,
  Wand2
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import { cn, generateId } from '../../lib/utils';
import { useTTSStore, HistoryItem } from '../../stores/ttsStore';
import Button from '../ui/Button';
import Select from '../ui/Select';
import Slider from '../ui/Slider';
import Layout from '../layout/Layout';
import HistoryPanel from './HistoryPanel';

// Language options
const LANGUAGES = [
  { value: 'en', label: 'English' },
  { value: 'zh', label: 'Chinese' },
  { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'ja', label: 'Japanese' },
  { value: 'ko', label: 'Korean' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'ru', label: 'Russian' },
  { value: 'it', label: 'Italian' },
];

// Presets
const PRESETS = {
  fast: { sampling: 10, temperature: 0.7, beam_size: 1 },
  balanced: { sampling: 25, temperature: 1.0, beam_size: 1 },
  quality: { sampling: 50, temperature: 1.0, beam_size: 3 },
};

const TTSWorkspace: React.FC = () => {
  const {
    text,
    setText,
    selectedVoice,
    setSelectedVoice,
    voices,
    setVoices,
    language,
    setLanguage,
    backend,
    setBackend,
    params,
    setParam,
    resetParams,
    isGenerating,
    setIsGenerating,
    showAdvanced,
    setShowAdvanced,
    addToHistory,
  } = useTTSStore();

  // Fetch voices on mount
  useEffect(() => {
    const fetchVoices = async () => {
      try {
        const response = await fetch('/voices');
        const data = await response.json();
        const voiceList = Array.isArray(data) ? data : data.voices || [];
        const voiceObjects = voiceList.map((name: string) => ({
          id: name,
          name: name,
        }));
        setVoices(voiceObjects);
        if (voiceObjects.length > 0 && !selectedVoice) {
          setSelectedVoice(voiceObjects[0].name);
        }
      } catch (error) {
        toast.error('Failed to load voices');
      }
    };
    fetchVoices();
  }, []);

  // Generate speech
  const handleGenerate = useCallback(async () => {
    if (!text.trim()) {
      toast.error('Please enter some text');
      return;
    }
    if (!selectedVoice) {
      toast.error('Please select a voice');
      return;
    }

    setIsGenerating(true);

    try {
      const requestBody: Record<string, any> = {
        text,
        voice_name: selectedVoice,
        language,
        output_format: 'wav',
      };

      if (backend === 'glm-tts') {
        requestBody.sampling = params.sampling;
        requestBody.temperature = params.temperature;
        requestBody.top_p = params.glm_top_p;
        requestBody.min_token_text_ratio = params.min_token_text_ratio;
        requestBody.max_token_text_ratio = params.max_token_text_ratio;
        requestBody.beam_size = params.beam_size;
        requestBody.repetition_penalty = params.repetition_penalty;
        requestBody.sample_method = params.sample_method;
      } else {
        requestBody.tau = params.tau;
        requestBody.gpt_cond_len = params.gpt_cond_len;
        requestBody.top_k = params.top_k;
        requestBody.top_p = params.top_p;
        requestBody.decoder_iterations = params.decoder_iterations;
        requestBody.split_sentences = params.split_sentences;
      }

      const response = await fetch('/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const buffer = await response.arrayBuffer();
      const blob = new Blob([buffer], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(blob);

      const historyItem: HistoryItem = {
        id: generateId(),
        text,
        voice: selectedVoice,
        language,
        backend,
        params: backend === 'glm-tts' 
          ? {
              sampling: params.sampling,
              temperature: params.temperature,
              glm_top_p: params.glm_top_p,
              beam_size: params.beam_size,
            }
          : {
              tau: params.tau,
              top_k: params.top_k,
              top_p: params.top_p,
            },
        audioUrl,
        timestamp: Date.now(),
      };

      addToHistory(historyItem);
      toast.success('Audio generated successfully!');

      const audio = new Audio(audioUrl);
      audio.play();

    } catch (error) {
      toast.error(`Generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsGenerating(false);
    }
  }, [text, selectedVoice, language, backend, params]);

  // Keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        handleGenerate();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleGenerate]);

  const applyPreset = (preset: keyof typeof PRESETS) => {
    const values = PRESETS[preset];
    Object.entries(values).forEach(([key, value]) => {
      setParam(key as any, value);
    });
    toast.success(`Applied ${preset} preset`);
  };

  return (
    <Layout rightPanel={<HistoryPanel />}>
      <div className="h-full flex flex-col relative overflow-hidden">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-accent/5 via-transparent to-purple-500/5 pointer-events-none" />
        <div className="absolute top-0 right-0 w-96 h-96 bg-accent/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl pointer-events-none" />
        
        {/* Content */}
        <div className="relative z-10 h-full flex flex-col p-8">
          {/* Header */}
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent to-blue-600 flex items-center justify-center shadow-lg shadow-accent/25">
                <Wand2 className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-text-primary">
                  Text to Speech
                </h1>
                <p className="text-sm text-text-tertiary">
                  Transform text into natural speech with AI voice cloning
                </p>
              </div>
            </div>
          </motion.div>

          {/* Main content */}
          <div className="flex-1 flex flex-col gap-6 min-h-0">
            {/* Text input card */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="flex-1 min-h-[200px] flex flex-col"
            >
              <div className="card flex-1 flex flex-col p-1 bg-gradient-to-b from-bg-secondary to-bg-primary">
                <div className="flex-1 p-4 flex flex-col">
                  <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Type or paste your text here..."
                    className={cn(
                      'flex-1 w-full bg-transparent resize-none',
                      'text-text-primary placeholder:text-text-tertiary',
                      'focus:outline-none text-lg leading-relaxed',
                      'min-h-[120px]'
                    )}
                    maxLength={2000}
                  />
                </div>
                
                {/* Input footer */}
                <div className="flex items-center justify-between px-4 py-3 border-t border-border/50 bg-bg-tertiary/30 rounded-b-xl">
                  <div className="flex items-center gap-4">
                    <span className={cn(
                      'text-xs font-mono transition-colors',
                      text.length > 1800 ? 'text-warning' : 'text-text-tertiary'
                    )}>
                      {text.length.toLocaleString()} / 2,000
                    </span>
                    {text.length > 0 && (
                      <span className="text-xs text-text-tertiary">
                        ~{Math.ceil(text.length / 150)} sec estimated
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-text-tertiary">
                    <kbd className="px-1.5 py-0.5 bg-bg-tertiary border border-border rounded text-xxs font-mono">
                      {navigator.platform.includes('Mac') ? '⌘' : 'Ctrl'}
                    </kbd>
                    <span>+</span>
                    <kbd className="px-1.5 py-0.5 bg-bg-tertiary border border-border rounded text-xxs font-mono">
                      Enter
                    </kbd>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Controls section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="space-y-4"
            >
              {/* Control cards row */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Voice card */}
                <div className="card p-4 hover:border-accent/50 transition-colors group">
                  <div className="flex items-center gap-2 mb-3">
                    <Volume2 className="w-4 h-4 text-accent" />
                    <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">Voice</span>
                  </div>
                  <select
                    value={selectedVoice}
                    onChange={(e) => setSelectedVoice(e.target.value)}
                    className="w-full bg-bg-tertiary border-0 rounded-lg px-3 py-2.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent/50 cursor-pointer"
                  >
                    {voices.map((v) => (
                      <option key={v.name} value={v.name}>{v.name}</option>
                    ))}
                  </select>
                </div>

                {/* Language card */}
                <div className="card p-4 hover:border-accent/50 transition-colors group">
                  <div className="flex items-center gap-2 mb-3">
                    <Globe className="w-4 h-4 text-green-400" />
                    <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">Language</span>
                  </div>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="w-full bg-bg-tertiary border-0 rounded-lg px-3 py-2.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent/50 cursor-pointer"
                  >
                    {LANGUAGES.map((l) => (
                      <option key={l.value} value={l.value}>{l.label}</option>
                    ))}
                  </select>
                </div>

                {/* Model card */}
                <div className="card p-4 hover:border-accent/50 transition-colors group">
                  <div className="flex items-center gap-2 mb-3">
                    <Cpu className="w-4 h-4 text-purple-400" />
                    <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">Model</span>
                  </div>
                  <select
                    value={backend}
                    onChange={(e) => setBackend(e.target.value as 'glm-tts' | 'xtts')}
                    className="w-full bg-bg-tertiary border-0 rounded-lg px-3 py-2.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent/50 cursor-pointer"
                  >
                    <option value="glm-tts">GLM-TTS</option>
                    <option value="xtts">XTTS v2</option>
                  </select>
                </div>
              </div>

              {/* Advanced settings toggle */}
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg transition-all w-full justify-center',
                  showAdvanced 
                    ? 'bg-accent/10 text-accent border border-accent/30' 
                    : 'bg-bg-secondary text-text-secondary hover:bg-bg-tertiary hover:text-text-primary border border-transparent'
                )}
              >
                <Settings2 className="w-4 h-4" />
                <span className="text-sm font-medium">
                  {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
                </span>
                {showAdvanced ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>

              {/* Advanced settings panel */}
              <AnimatePresence>
                {showAdvanced && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="card p-6 bg-gradient-to-b from-bg-secondary to-bg-primary">
                      {/* Presets */}
                      <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-3">
                          <span className="text-sm font-medium text-text-secondary">Quick Presets</span>
                          <div className="flex gap-2">
                            <button
                              onClick={() => applyPreset('fast')}
                              className="btn-ghost btn-sm flex items-center gap-1.5 text-yellow-400 hover:bg-yellow-400/10"
                            >
                              <Zap className="w-3.5 h-3.5" />
                              Fast
                            </button>
                            <button
                              onClick={() => applyPreset('balanced')}
                              className="btn-ghost btn-sm flex items-center gap-1.5 text-green-400 hover:bg-green-400/10"
                            >
                              <Target className="w-3.5 h-3.5" />
                              Balanced
                            </button>
                            <button
                              onClick={() => applyPreset('quality')}
                              className="btn-ghost btn-sm flex items-center gap-1.5 text-purple-400 hover:bg-purple-400/10"
                            >
                              <Sparkles className="w-3.5 h-3.5" />
                              Quality
                            </button>
                          </div>
                        </div>
                        <button
                          onClick={resetParams}
                          className="text-xs text-text-tertiary hover:text-text-secondary transition-colors"
                        >
                          Reset defaults
                        </button>
                      </div>

                      {/* Parameters grid */}
                      {backend === 'glm-tts' ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                          <Slider
                            label="Sampling"
                            value={params.sampling}
                            onChange={(v) => setParam('sampling', v)}
                            min={1}
                            max={100}
                            description="Lower = more deterministic"
                          />
                          <Slider
                            label="Temperature"
                            value={params.temperature}
                            onChange={(v) => setParam('temperature', v)}
                            min={0.1}
                            max={2}
                            step={0.1}
                            formatValue={(v) => v.toFixed(1)}
                            description="Higher = more variation"
                          />
                          <Slider
                            label="Top-P"
                            value={params.glm_top_p}
                            onChange={(v) => setParam('glm_top_p', v)}
                            min={0.1}
                            max={1}
                            step={0.05}
                            formatValue={(v) => v.toFixed(2)}
                            description="Nucleus sampling"
                          />
                          <Slider
                            label="Beam Size"
                            value={params.beam_size}
                            onChange={(v) => setParam('beam_size', v)}
                            min={1}
                            max={5}
                            description="Quality vs speed"
                          />
                          <Slider
                            label="Min Token Ratio"
                            value={params.min_token_text_ratio}
                            onChange={(v) => setParam('min_token_text_ratio', v)}
                            min={1}
                            max={20}
                            description="Minimum audio length"
                          />
                          <Slider
                            label="Max Token Ratio"
                            value={params.max_token_text_ratio}
                            onChange={(v) => setParam('max_token_text_ratio', v)}
                            min={10}
                            max={100}
                            description="Maximum audio length"
                          />
                          <Slider
                            label="Repetition Penalty"
                            value={params.repetition_penalty}
                            onChange={(v) => setParam('repetition_penalty', v)}
                            min={0}
                            max={1}
                            step={0.05}
                            formatValue={(v) => v.toFixed(2)}
                            description="Prevent repetition"
                          />
                          <div className="space-y-2">
                            <label className="text-sm font-medium text-text-secondary">
                              Sample Method
                            </label>
                            <select
                              value={params.sample_method}
                              onChange={(e) => setParam('sample_method', e.target.value as 'ras' | 'topk')}
                              className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent/50"
                            >
                              <option value="ras">RAS (Repetition-Aware)</option>
                              <option value="topk">Top-K</option>
                            </select>
                            <p className="text-xs text-text-tertiary">Sampling algorithm</p>
                          </div>
                        </div>
                      ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                          <Slider
                            label="Temperature (Tau)"
                            value={params.tau}
                            onChange={(v) => setParam('tau', v)}
                            min={-1}
                            max={1}
                            step={0.1}
                            formatValue={(v) => v.toFixed(1)}
                          />
                          <Slider
                            label="GPT Conditioning"
                            value={params.gpt_cond_len}
                            onChange={(v) => setParam('gpt_cond_len', v)}
                            min={1}
                            max={50}
                          />
                          <Slider
                            label="Top-K"
                            value={params.top_k}
                            onChange={(v) => setParam('top_k', v)}
                            min={1}
                            max={50}
                          />
                          <Slider
                            label="Top-P"
                            value={params.top_p}
                            onChange={(v) => setParam('top_p', v)}
                            min={0}
                            max={1}
                            step={0.05}
                            formatValue={(v) => v.toFixed(2)}
                          />
                          <Slider
                            label="Decoder Iterations"
                            value={params.decoder_iterations}
                            onChange={(v) => setParam('decoder_iterations', v)}
                            min={10}
                            max={100}
                          />
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Generate button */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="flex justify-center pt-2"
            >
              <motion.button
                onClick={handleGenerate}
                disabled={isGenerating || !text.trim()}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={cn(
                  'relative group px-8 py-4 rounded-xl font-semibold text-white',
                  'bg-gradient-to-r from-accent to-blue-600',
                  'shadow-lg shadow-accent/25',
                  'disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none',
                  'transition-all duration-300',
                  !isGenerating && 'hover:shadow-xl hover:shadow-accent/30'
                )}
              >
                {/* Glow effect */}
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-accent to-blue-600 blur-xl opacity-50 group-hover:opacity-75 transition-opacity" />
                
                <span className="relative flex items-center gap-2">
                  {isGenerating ? (
                    <>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                      >
                        <Sparkles className="w-5 h-5" />
                      </motion.div>
                      Generating...
                    </>
                  ) : (
                    <>
                      <Send className="w-5 h-5" />
                      Generate Speech
                    </>
                  )}
                </span>
              </motion.button>
            </motion.div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default TTSWorkspace;
