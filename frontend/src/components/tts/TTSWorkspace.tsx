import React, { useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  Send, 
  Settings2, 
  ChevronDown, 
  ChevronUp,
  Sparkles,
  Zap,
  Target
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

      // Add backend-specific params
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

      // Add to history
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
      toast.success('Audio generated!');

      // Auto-play
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

  // Apply preset
  const applyPreset = (preset: keyof typeof PRESETS) => {
    const values = PRESETS[preset];
    Object.entries(values).forEach(([key, value]) => {
      setParam(key as any, value);
    });
    toast.success(`Applied ${preset} preset`);
  };

  return (
    <Layout rightPanel={<HistoryPanel />}>
      <div className="h-full flex flex-col p-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-text-primary">
            Text to Speech
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            Generate natural speech from text using AI voice cloning
          </p>
        </div>

        {/* Main content */}
        <div className="flex-1 flex flex-col gap-4">
          {/* Text input */}
          <div className="panel flex-1 flex flex-col">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter the text you want to convert to speech..."
              className={cn(
                'flex-1 w-full bg-transparent resize-none',
                'text-text-primary placeholder:text-text-tertiary',
                'focus:outline-none text-base leading-relaxed'
              )}
              maxLength={2000}
            />
            <div className="flex items-center justify-between pt-3 border-t border-border mt-3">
              <span className="text-xs text-text-tertiary">
                {text.length} / 2000 characters
              </span>
              <div className="flex items-center gap-2 text-xs text-text-tertiary">
                <kbd className="px-1.5 py-0.5 bg-bg-tertiary rounded text-xxs">
                  {navigator.platform.includes('Mac') ? '⌘' : 'Ctrl'}
                </kbd>
                <span>+</span>
                <kbd className="px-1.5 py-0.5 bg-bg-tertiary rounded text-xxs">
                  Enter
                </kbd>
                <span>to generate</span>
              </div>
            </div>
          </div>

          {/* Controls row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Voice select */}
            <Select
              label="Voice"
              value={selectedVoice}
              onChange={setSelectedVoice}
              options={voices.map((v) => ({ value: v.name, label: v.name }))}
              placeholder="Select a voice"
            />

            {/* Language select */}
            <Select
              label="Language"
              value={language}
              onChange={setLanguage}
              options={LANGUAGES}
            />

            {/* Backend select */}
            <Select
              label="Model"
              value={backend}
              onChange={(v) => setBackend(v as 'glm-tts' | 'xtts')}
              options={[
                { value: 'glm-tts', label: 'GLM-TTS' },
                { value: 'xtts', label: 'XTTS v2' },
              ]}
            />
          </div>

          {/* Advanced settings toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary transition-colors"
          >
            <Settings2 className="w-4 h-4" />
            <span>{showAdvanced ? 'Hide' : 'Show'} advanced settings</span>
            {showAdvanced ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>

          {/* Advanced settings panel */}
          {showAdvanced && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="panel"
            >
              {/* Presets */}
              <div className="flex items-center gap-2 mb-4">
                <span className="text-sm text-text-secondary">Presets:</span>
                <button
                  onClick={() => applyPreset('fast')}
                  className="btn-ghost btn-sm flex items-center gap-1.5"
                >
                  <Zap className="w-3.5 h-3.5" />
                  Fast
                </button>
                <button
                  onClick={() => applyPreset('balanced')}
                  className="btn-ghost btn-sm flex items-center gap-1.5"
                >
                  <Target className="w-3.5 h-3.5" />
                  Balanced
                </button>
                <button
                  onClick={() => applyPreset('quality')}
                  className="btn-ghost btn-sm flex items-center gap-1.5"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  Quality
                </button>
                <div className="flex-1" />
                <button
                  onClick={resetParams}
                  className="text-xs text-text-tertiary hover:text-text-secondary"
                >
                  Reset to defaults
                </button>
              </div>

              {/* Parameters grid */}
              {backend === 'glm-tts' ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
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
                  <Select
                    label="Sample Method"
                    value={params.sample_method}
                    onChange={(v) => setParam('sample_method', v as 'ras' | 'topk')}
                    options={[
                      { value: 'ras', label: 'RAS (Repetition-Aware)' },
                      { value: 'topk', label: 'Top-K' },
                    ]}
                  />
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
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
            </motion.div>
          )}

          {/* Generate button */}
          <div className="flex justify-center pt-2">
            <Button
              onClick={handleGenerate}
              loading={isGenerating}
              size="lg"
              className={cn(
                'min-w-[200px]',
                !isGenerating && 'glow-on-hover'
              )}
            >
              {isGenerating ? (
                'Generating...'
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  Generate Speech
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default TTSWorkspace;

