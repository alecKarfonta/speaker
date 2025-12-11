import React, { useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform } from 'framer-motion';
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
  Wand2,
  Waves,
  Mic2
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import { cn, generateId } from '../../lib/utils';
import { useTTSStore, HistoryItem } from '../../stores/ttsStore';
import Slider from '../ui/Slider';
import Layout from '../layout/Layout';
import HistoryPanel from './HistoryPanel';

const LANGUAGES = [
  { value: 'en', label: 'English', flag: '🇺🇸' },
  { value: 'zh', label: 'Chinese', flag: '🇨🇳' },
  { value: 'es', label: 'Spanish', flag: '🇪🇸' },
  { value: 'fr', label: 'French', flag: '🇫🇷' },
  { value: 'de', label: 'German', flag: '🇩🇪' },
  { value: 'ja', label: 'Japanese', flag: '🇯🇵' },
  { value: 'ko', label: 'Korean', flag: '🇰🇷' },
];

const PRESETS = {
  fast: { sampling: 10, temperature: 0.7, beam_size: 1, label: 'Fast', icon: Zap, color: 'from-yellow-500 to-orange-500' },
  balanced: { sampling: 25, temperature: 1.0, beam_size: 1, label: 'Balanced', icon: Target, color: 'from-green-500 to-emerald-500' },
  quality: { sampling: 50, temperature: 1.0, beam_size: 3, label: 'Quality', icon: Sparkles, color: 'from-purple-500 to-pink-500' },
};

// Animated background orbs
const BackgroundOrbs: React.FC = () => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none">
    <motion.div
      animate={{
        x: [0, 100, 0],
        y: [0, -50, 0],
        scale: [1, 1.2, 1],
      }}
      transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
      className="absolute -top-40 -right-40 w-96 h-96 rounded-full"
      style={{
        background: 'radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%)',
        filter: 'blur(40px)',
      }}
    />
    <motion.div
      animate={{
        x: [0, -80, 0],
        y: [0, 80, 0],
        scale: [1, 1.3, 1],
      }}
      transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      className="absolute -bottom-20 -left-20 w-80 h-80 rounded-full"
      style={{
        background: 'radial-gradient(circle, rgba(168, 85, 247, 0.12) 0%, transparent 70%)',
        filter: 'blur(40px)',
      }}
    />
    <motion.div
      animate={{
        x: [0, 50, 0],
        y: [0, 100, 0],
      }}
      transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
      className="absolute top-1/2 left-1/2 w-64 h-64 rounded-full"
      style={{
        background: 'radial-gradient(circle, rgba(34, 211, 238, 0.08) 0%, transparent 70%)',
        filter: 'blur(40px)',
      }}
    />
  </div>
);

// Voice card component
const VoiceCard: React.FC<{
  voice: string;
  selected: boolean;
  onClick: () => void;
}> = ({ voice, selected, onClick }) => (
  <motion.button
    whileHover={{ scale: 1.02, y: -2 }}
    whileTap={{ scale: 0.98 }}
    onClick={onClick}
    className={cn(
      'relative p-4 rounded-xl text-left transition-all duration-300 overflow-hidden group',
      selected 
        ? 'bg-gradient-to-br from-accent/20 to-purple-500/10 border-accent/50' 
        : 'bg-bg-secondary/50 border-white/5 hover:border-white/10',
      'border'
    )}
  >
    {selected && (
      <motion.div
        layoutId="voiceIndicator"
        className="absolute inset-0 bg-gradient-to-br from-accent/10 to-purple-500/5"
        transition={{ type: "spring", stiffness: 500, damping: 30 }}
      />
    )}
    <div className="relative flex items-center gap-3">
      <div className={cn(
        'w-10 h-10 rounded-lg flex items-center justify-center',
        selected 
          ? 'bg-gradient-to-br from-accent to-purple-500 shadow-lg shadow-accent/30' 
          : 'bg-bg-tertiary group-hover:bg-bg-hover'
      )}>
        <Mic2 className={cn(
          'w-5 h-5',
          selected ? 'text-white' : 'text-text-tertiary group-hover:text-text-secondary'
        )} />
      </div>
      <div>
        <p className={cn(
          'font-medium capitalize',
          selected ? 'text-white' : 'text-text-secondary group-hover:text-text-primary'
        )}>
          {voice.replace(/_/g, ' ')}
        </p>
        <p className="text-xs text-text-tertiary">Voice clone</p>
      </div>
    </div>
    {selected && (
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        className="absolute top-2 right-2 w-2 h-2 rounded-full bg-accent shadow-lg shadow-accent/50"
      />
    )}
  </motion.button>
);

const TTSWorkspace: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  const {
    text, setText,
    selectedVoice, setSelectedVoice,
    voices, setVoices,
    language, setLanguage,
    backend, setBackend,
    params, setParam, resetParams,
    isGenerating, setIsGenerating,
    showAdvanced, setShowAdvanced,
    addToHistory,
  } = useTTSStore();

  // Track mouse for spotlight effect
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        mouseX.set(((e.clientX - rect.left) / rect.width) * 100);
        mouseY.set(((e.clientY - rect.top) / rect.height) * 100);
      }
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useEffect(() => {
    const fetchVoices = async () => {
      try {
        const response = await fetch('/voices');
        const data = await response.json();
        const voiceList = Array.isArray(data) ? data : data.voices || [];
        const voiceObjects = voiceList.map((name: string) => ({ id: name, name }));
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
        Object.assign(requestBody, {
          sampling: params.sampling,
          temperature: params.temperature,
          top_p: params.glm_top_p,
          min_token_text_ratio: params.min_token_text_ratio,
          max_token_text_ratio: params.max_token_text_ratio,
          beam_size: params.beam_size,
          repetition_penalty: params.repetition_penalty,
          sample_method: params.sample_method,
        });
      }

      const response = await fetch('/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const buffer = await response.arrayBuffer();
      const blob = new Blob([buffer], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(blob);

      addToHistory({
        id: generateId(),
        text,
        voice: selectedVoice,
        language,
        backend,
        params: { sampling: params.sampling, temperature: params.temperature },
        audioUrl,
        timestamp: Date.now(),
      });

      toast.success('Audio generated!');
      new Audio(audioUrl).play();

    } catch (error) {
      toast.error(`Failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsGenerating(false);
    }
  }, [text, selectedVoice, language, backend, params]);

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
    const { sampling, temperature, beam_size } = PRESETS[preset];
    setParam('sampling', sampling);
    setParam('temperature', temperature);
    setParam('beam_size', beam_size);
    toast.success(`${PRESETS[preset].label} preset applied`);
  };

  const spotlightStyle = {
    '--mouse-x': `${mouseX.get()}%`,
    '--mouse-y': `${mouseY.get()}%`,
  } as React.CSSProperties;

  return (
    <Layout rightPanel={<HistoryPanel />}>
      <div 
        ref={containerRef}
        className="h-full flex flex-col relative overflow-hidden aurora-bg noise"
        style={spotlightStyle}
      >
        <BackgroundOrbs />
        
        <div className="relative z-10 h-full flex flex-col p-8 max-w-5xl mx-auto w-full">
          {/* Header */}
          <motion.div 
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="mb-10 text-center"
          >
            <motion.div 
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-6"
              whileHover={{ scale: 1.02 }}
            >
              <Waves className="w-4 h-4 text-cyan" />
              <span className="text-sm text-text-secondary">AI Voice Synthesis</span>
            </motion.div>
            
            <h1 className="text-5xl font-bold mb-4">
              <span className="gradient-text">Transform Text</span>
              <br />
              <span className="text-text-primary">Into Natural Speech</span>
            </h1>
            
            <p className="text-text-tertiary text-lg max-w-xl mx-auto">
              Clone any voice with just a few seconds of audio.
              Create stunning voiceovers in seconds.
            </p>
          </motion.div>

          {/* Main content */}
          <div className="flex-1 flex flex-col gap-6 min-h-0">
            {/* Text input */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.5 }}
              className="glow-card"
            >
              <div className="relative p-6">
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter your text here... Let the magic begin."
                  className={cn(
                    'w-full min-h-[140px] bg-transparent resize-none',
                    'text-lg text-text-primary placeholder:text-text-tertiary/50',
                    'focus:outline-none leading-relaxed'
                  )}
                  maxLength={2000}
                />
                
                <div className="flex items-center justify-between pt-4 border-t border-white/5">
                  <div className="flex items-center gap-4">
                    <span className={cn(
                      'text-sm font-mono',
                      text.length > 1800 ? 'text-warning' : 'text-text-tertiary'
                    )}>
                      {text.length.toLocaleString()}<span className="text-text-tertiary/50"> / 2,000</span>
                    </span>
                    {text.length > 0 && (
                      <span className="text-sm text-text-tertiary">
                        ~{Math.ceil(text.length / 150)}s
                      </span>
                    )}
                  </div>
                  <kbd className="hidden md:flex items-center gap-1 px-2 py-1 rounded-lg bg-white/5 text-xs text-text-tertiary">
                    <span>{navigator.platform.includes('Mac') ? '⌘' : 'Ctrl'}</span>
                    <span>+</span>
                    <span>Enter</span>
                  </kbd>
                </div>
              </div>
            </motion.div>

            {/* Voice selection */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.5 }}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-text-secondary flex items-center gap-2">
                  <Volume2 className="w-4 h-4 text-accent" />
                  Select Voice
                </h3>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-text-secondary focus:outline-none focus:border-accent/50"
                >
                  {LANGUAGES.map(l => (
                    <option key={l.value} value={l.value}>{l.flag} {l.label}</option>
                  ))}
                </select>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {voices.slice(0, 8).map((voice) => (
                  <VoiceCard
                    key={voice.name}
                    voice={voice.name}
                    selected={selectedVoice === voice.name}
                    onClick={() => setSelectedVoice(voice.name)}
                  />
                ))}
              </div>
            </motion.div>

            {/* Advanced toggle */}
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              onClick={() => setShowAdvanced(!showAdvanced)}
              className={cn(
                'flex items-center justify-center gap-2 py-3 rounded-xl transition-all',
                showAdvanced
                  ? 'bg-accent/10 text-accent border border-accent/20'
                  : 'bg-white/5 text-text-secondary hover:bg-white/10 border border-transparent'
              )}
            >
              <Settings2 className="w-4 h-4" />
              <span className="text-sm font-medium">
                {showAdvanced ? 'Hide' : 'Show'} Advanced Controls
              </span>
              {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </motion.button>

            {/* Advanced panel */}
            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="card p-6">
                    {/* Presets */}
                    <div className="flex flex-wrap items-center gap-3 mb-6">
                      {Object.entries(PRESETS).map(([key, preset]) => {
                        const Icon = preset.icon;
                        return (
                          <motion.button
                            key={key}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => applyPreset(key as keyof typeof PRESETS)}
                            className={cn(
                              'flex items-center gap-2 px-4 py-2 rounded-xl',
                              'bg-gradient-to-r bg-opacity-10 border border-white/10',
                              'hover:border-white/20 transition-all',
                              preset.color
                            )}
                            style={{
                              background: `linear-gradient(135deg, rgba(255,255,255,0.05), transparent)`,
                            }}
                          >
                            <Icon className="w-4 h-4" />
                            <span className="text-sm font-medium text-white">{preset.label}</span>
                          </motion.button>
                        );
                      })}
                      <button
                        onClick={resetParams}
                        className="text-xs text-text-tertiary hover:text-text-secondary ml-auto"
                      >
                        Reset
                      </button>
                    </div>

                    {/* Sliders */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                      <Slider
                        label="Sampling"
                        value={params.sampling}
                        onChange={(v) => setParam('sampling', v)}
                        min={1} max={100}
                        description="Randomness"
                      />
                      <Slider
                        label="Temperature"
                        value={params.temperature}
                        onChange={(v) => setParam('temperature', v)}
                        min={0.1} max={2} step={0.1}
                        formatValue={(v) => v.toFixed(1)}
                        description="Variation"
                      />
                      <Slider
                        label="Top-P"
                        value={params.glm_top_p}
                        onChange={(v) => setParam('glm_top_p', v)}
                        min={0.1} max={1} step={0.05}
                        formatValue={(v) => v.toFixed(2)}
                        description="Nucleus"
                      />
                      <Slider
                        label="Beam Size"
                        value={params.beam_size}
                        onChange={(v) => setParam('beam_size', v)}
                        min={1} max={5}
                        description="Quality"
                      />
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Generate button */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.5 }}
              className="flex justify-center pt-4"
            >
              <motion.button
                onClick={handleGenerate}
                disabled={isGenerating || !text.trim()}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                className={cn(
                  'relative group px-10 py-4 rounded-2xl font-semibold text-lg',
                  'disabled:opacity-40 disabled:cursor-not-allowed',
                  'transition-all duration-300'
                )}
              >
                {/* Animated gradient background */}
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-accent via-purple-500 to-pink-500 opacity-90" />
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-accent via-purple-500 to-pink-500 blur-xl opacity-50 group-hover:opacity-75 transition-opacity" />
                
                {/* Shimmer effect */}
                <div className="absolute inset-0 rounded-2xl overflow-hidden">
                  <div className="absolute inset-0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                </div>
                
                {/* Content */}
                <span className="relative flex items-center gap-3 text-white">
                  {isGenerating ? (
                    <>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <Sparkles className="w-5 h-5" />
                      </motion.div>
                      <span>Creating Magic...</span>
                    </>
                  ) : (
                    <>
                      <Wand2 className="w-5 h-5" />
                      <span>Generate Speech</span>
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
