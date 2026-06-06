import React, { useCallback, useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Volume2,
  BookmarkPlus,
  Check,
  ChevronDown,
  ChevronUp,
  Download,
  Layers,
  Sparkles,
  Wand2,
  Zap,
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import Layout from '../layout/Layout';
import Slider from '../ui/Slider';
import Button from '../ui/Button';
import SfxLibraryPanel from './SfxLibraryPanel';
import { generateSfx, ensureAudioWavBlob } from '../../services/sfxApi';
import { saveSfxBlob } from '../../services/sfxAudioDb';
import {
  cn,
  downloadBlob,
  formatDuration,
  generateId,
} from '../../lib/utils';
import {
  defaultSfxName,
  SfxParams,
  useSfxStore,
} from '../../stores/sfxStore';

const PROMPT_PRESETS = [
  'A dog barking loudly in a park.',
  'Busy coffee shop ambience with espresso machine hiss.',
  'Ocean waves crashing on a rocky shore.',
  'Fire crackling warmly in a fireplace.',
  'Birds chirping in a forest at dawn.',
  'Car engine starting and idling.',
];

interface SfxCandidate {
  id: string;
  prompt: string;
  params: Omit<SfxParams, 'batchCount'>;
  duration: number;
  genTime: number;
  blob: Blob;
  audioUrl: string;
  seed?: number;
}

const BackgroundOrbs: React.FC = () => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none">
    <motion.div
      animate={{ x: [0, 80, 0], y: [0, -40, 0], scale: [1, 1.15, 1] }}
      transition={{ duration: 22, repeat: Infinity, ease: 'linear' }}
      className="absolute -top-32 -right-32 w-96 h-96 rounded-full"
      style={{
        background:
          'radial-gradient(circle, rgba(244, 63, 94, 0.14) 0%, transparent 70%)',
        filter: 'blur(40px)',
      }}
    />
    <motion.div
      animate={{ x: [0, -60, 0], y: [0, 60, 0] }}
      transition={{ duration: 28, repeat: Infinity, ease: 'linear' }}
      className="absolute -bottom-24 -left-24 w-80 h-80 rounded-full"
      style={{
        background:
          'radial-gradient(circle, rgba(249, 115, 22, 0.12) 0%, transparent 70%)',
        filter: 'blur(40px)',
      }}
    />
  </div>
);

function downloadFilename(name: string): string {
  const safe = name
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '_')
    .slice(0, 60);
  return `${safe || 'sound_effect'}.wav`;
}

const SoundEffectsPage: React.FC = () => {
  const {
    prompt,
    setPrompt,
    params,
    setParam,
    addToLibrary,
  } = useSfxStore();

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState({ current: 0, total: 0 });
  const [candidates, setCandidates] = useState<SfxCandidate[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const candidateUrlsRef = useRef<string[]>([]);

  const revokeCandidateUrls = useCallback(() => {
    candidateUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    candidateUrlsRef.current = [];
  }, []);

  useEffect(() => {
    return () => {
      revokeCandidateUrls();
    };
  }, [revokeCandidateUrls]);

  const requestParams = (): Omit<SfxParams, 'batchCount'> => ({
    seconds: params.seconds,
    num_inference_steps: params.num_inference_steps,
    cfg_scale: params.cfg_scale,
    sigma_shift: params.sigma_shift,
    seed: params.seed,
  });

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Describe the sound you want to generate');
      return;
    }

    revokeCandidateUrls();
    setCandidates([]);
    setSelectedId(null);
    setIsGenerating(true);
    setGenProgress({ current: 0, total: params.batchCount });

    const nextCandidates: SfxCandidate[] = [];
    const baseParams = requestParams();

    try {
      for (let i = 0; i < params.batchCount; i++) {
        setGenProgress({ current: i + 1, total: params.batchCount });
        const seed =
          baseParams.seed != null ? baseParams.seed + i : undefined;
        const result = await generateSfx({
          prompt: prompt.trim(),
          seconds: baseParams.seconds,
          num_inference_steps: baseParams.num_inference_steps,
          cfg_scale: baseParams.cfg_scale,
          sigma_shift: baseParams.sigma_shift,
          seed,
        });

        const wavBlob = ensureAudioWavBlob(result.blob);
        const url = URL.createObjectURL(wavBlob);
        candidateUrlsRef.current.push(url);
        const candidate: SfxCandidate = {
          id: generateId(),
          prompt: prompt.trim(),
          params: { ...baseParams, seed: seed ?? null },
          duration: result.duration,
          genTime: result.genTime,
          blob: wavBlob,
          audioUrl: url,
          seed,
        };
        nextCandidates.push(candidate);
        setCandidates([...nextCandidates]);
      }

      if (nextCandidates.length === 1) {
        setSelectedId(nextCandidates[0].id);
      }
      toast.success(`Generated ${nextCandidates.length} variant(s)`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Generation failed');
    } finally {
      setIsGenerating(false);
      setGenProgress({ current: 0, total: 0 });
    }
  };

  const handleSaveSelected = async () => {
    const selected = candidates.find((c) => c.id === selectedId);
    if (!selected) {
      toast.error('Select a variant to save');
      return;
    }

    const libraryId = generateId();
    const name = defaultSfxName(selected.prompt);
    try {
      await saveSfxBlob(libraryId, selected.blob);
      addToLibrary({
        id: libraryId,
        name,
        prompt: selected.prompt,
        params: selected.params,
        duration: selected.duration,
        createdAt: Date.now(),
      });
      toast.success(`Saved "${name}" to library`);
    } catch {
      toast.error('Failed to save to library');
    }
  };

  const handleDownloadSelected = () => {
    const selected = candidates.find((c) => c.id === selectedId);
    if (!selected) {
      toast.error('Select a variant to download');
      return;
    }
    downloadBlob(
      selected.blob,
      downloadFilename(defaultSfxName(selected.prompt))
    );
  };

  const handleSaveCandidate = async (candidate: SfxCandidate) => {
    const libraryId = generateId();
    const name = defaultSfxName(candidate.prompt);
    try {
      await saveSfxBlob(libraryId, candidate.blob);
      addToLibrary({
        id: libraryId,
        name,
        prompt: candidate.prompt,
        params: candidate.params,
        duration: candidate.duration,
        createdAt: Date.now(),
      });
      toast.success(`Saved variant to library`);
    } catch {
      toast.error('Failed to save');
    }
  };

  return (
    <Layout rightPanel={<SfxLibraryPanel />}>
      <div className="h-full flex flex-col relative overflow-hidden aurora-bg noise">
        <BackgroundOrbs />

        <div className="relative z-10 flex flex-col h-full">
          {/* Header */}
          <div className="px-8 py-6 border-b border-white/5">
            <div className="flex items-center gap-4">
              <motion.div
                whileHover={{ scale: 1.05, rotate: -3 }}
                className="relative"
              >
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-rose-500 to-orange-500 blur-xl opacity-40" />
                <div className="relative w-14 h-14 rounded-2xl bg-gradient-to-br from-rose-500 to-orange-500 flex items-center justify-center shadow-2xl">
                  <Volume2 className="w-7 h-7 text-white" />
                </div>
              </motion.div>
              <div>
                <h1 className="text-2xl font-bold gradient-text">
                  Sound Effects
                </h1>
                <p className="text-sm text-text-tertiary mt-0.5">
                  MOSS-SoundEffect v2 · 48 kHz · up to 30s
                </p>
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6">
            {/* Prompt */}
            <div className="glow-card rounded-2xl p-6 border border-white/5">
              <label className="flex items-center gap-2 text-sm font-medium text-text-secondary mb-3">
                <Wand2 className="w-4 h-4 text-rose-400" />
                Sound description
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe the sound effect you want — environment, action, mood..."
                rows={4}
                className="w-full bg-bg-tertiary border border-white/10 rounded-xl px-4 py-3 text-text-primary placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-rose-500/40 resize-none"
              />

              <div className="flex flex-wrap gap-2 mt-4">
                {PROMPT_PRESETS.map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => setPrompt(preset)}
                    className="text-xs px-3 py-1.5 rounded-full bg-white/5 border border-white/5 text-text-tertiary hover:text-white hover:border-rose-500/30 hover:bg-rose-500/10 transition-colors"
                  >
                    {preset.length > 42 ? `${preset.slice(0, 42)}…` : preset}
                  </button>
                ))}
              </div>
            </div>

            {/* Controls row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glow-card rounded-2xl p-6 border border-white/5 space-y-5">
                <Slider
                  label="Duration"
                  value={params.seconds}
                  onChange={(v) => setParam('seconds', v)}
                  min={1}
                  max={30}
                  step={0.5}
                  formatValue={(v) => `${v}s`}
                  description="Length of generated audio (max 30s)"
                />

                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-text-secondary mb-3">
                    <Layers className="w-4 h-4 text-orange-400" />
                    Variants to generate
                  </label>
                  <div className="flex gap-2">
                    {[1, 2, 3, 4].map((n) => (
                      <button
                        key={n}
                        type="button"
                        onClick={() => setParam('batchCount', n)}
                        className={cn(
                          'flex-1 py-2.5 rounded-xl text-sm font-medium border transition-all',
                          params.batchCount === n
                            ? 'bg-gradient-to-br from-rose-500/20 to-orange-500/10 border-rose-500/40 text-white'
                            : 'bg-white/5 border-white/5 text-text-tertiary hover:border-white/15'
                        )}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                  <p className="text-xs text-text-tertiary mt-2">
                    Generate multiple takes and pick your favorite
                  </p>
                </div>
              </div>

              <div className="glow-card rounded-2xl p-6 border border-white/5">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center justify-between w-full text-sm font-medium text-text-secondary mb-4"
                >
                  <span className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-rose-400" />
                    Advanced sampling
                  </span>
                  {showAdvanced ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </button>

                <AnimatePresence>
                  {showAdvanced && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="space-y-5 overflow-hidden"
                    >
                      <Slider
                        label="Inference steps"
                        value={params.num_inference_steps}
                        onChange={(v) => setParam('num_inference_steps', v)}
                        min={20}
                        max={150}
                        step={5}
                      />
                      <Slider
                        label="CFG scale"
                        value={params.cfg_scale}
                        onChange={(v) => setParam('cfg_scale', v)}
                        min={1}
                        max={8}
                        step={0.1}
                        formatValue={(v) => v.toFixed(1)}
                      />
                      <Slider
                        label="Sigma shift"
                        value={params.sigma_shift}
                        onChange={(v) => setParam('sigma_shift', v)}
                        min={0}
                        max={10}
                        step={0.1}
                        formatValue={(v) => v.toFixed(1)}
                      />
                      <div>
                        <label className="text-sm font-medium text-text-secondary">
                          Seed (optional)
                        </label>
                        <input
                          type="number"
                          value={params.seed ?? ''}
                          placeholder="Random"
                          onChange={(e) => {
                            const v = e.target.value;
                            setParam(
                              'seed',
                              v === '' ? null : parseInt(v, 10)
                            );
                          }}
                          className="mt-2 w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-rose-500/40"
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {!showAdvanced && (
                  <p className="text-xs text-text-tertiary">
                    Steps {params.num_inference_steps} · CFG{' '}
                    {params.cfg_scale.toFixed(1)} · σ{' '}
                    {params.sigma_shift.toFixed(1)}
                  </p>
                )}
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-wrap items-center gap-3">
              <Button
                size="lg"
                loading={isGenerating}
                onClick={handleGenerate}
                className="bg-gradient-to-r from-rose-500 to-orange-500 hover:from-rose-400 hover:to-orange-400 border-0 shadow-lg shadow-rose-500/25"
              >
                <Zap className="w-5 h-5" />
                {isGenerating
                  ? `Generating ${genProgress.current}/${genProgress.total}…`
                  : `Generate ${params.batchCount} variant${params.batchCount > 1 ? 's' : ''}`}
              </Button>

              {selectedId && (
                <>
                  <Button variant="secondary" onClick={handleSaveSelected}>
                    <BookmarkPlus className="w-4 h-4" />
                    Save selected
                  </Button>
                  <Button variant="ghost" onClick={handleDownloadSelected}>
                    <Download className="w-4 h-4" />
                    Download WAV
                  </Button>
                </>
              )}
            </div>

            {/* Candidates */}
            <AnimatePresence>
              {candidates.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-4"
                >
                  <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wider">
                    Generated variants — click to select
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                    {candidates.map((candidate, index) => {
                      const isSelected = selectedId === candidate.id;
                      return (
                        <motion.button
                          key={candidate.id}
                          type="button"
                          layout
                          onClick={() => setSelectedId(candidate.id)}
                          className={cn(
                            'text-left glow-card rounded-2xl p-5 border transition-all duration-300',
                            isSelected
                              ? 'border-rose-500/60 ring-2 ring-rose-500/30 bg-rose-500/5'
                              : 'border-white/5 hover:border-white/15 bg-bg-secondary/30'
                          )}
                        >
                          <div className="flex items-center justify-between mb-3">
                            <span className="text-xs font-mono text-text-tertiary">
                              Variant {index + 1}
                              {candidate.seed != null && ` · seed ${candidate.seed}`}
                            </span>
                            {isSelected && (
                              <span className="flex items-center gap-1 text-xs text-rose-300">
                                <Check className="w-3.5 h-3.5" /> Selected
                              </span>
                            )}
                          </div>

                          <audio
                            src={candidate.audioUrl}
                            controls
                            preload="metadata"
                            className="w-full h-9 mb-4 rounded-lg opacity-90"
                            onClick={(e) => e.stopPropagation()}
                            onError={() => toast.error('Could not decode audio — try Download WAV')}
                          />

                          <div className="flex items-center justify-between">
                            <div className="text-xs text-text-tertiary">
                              <span className="font-mono text-text-secondary">
                                {formatDuration(candidate.duration)}
                              </span>
                              {' · '}
                              {candidate.genTime.toFixed(1)}s gen
                            </div>
                            <div
                              className="flex gap-1"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <button
                                type="button"
                                onClick={() =>
                                  downloadBlob(
                                    candidate.blob,
                                    downloadFilename(
                                      `variant_${index + 1}_${defaultSfxName(candidate.prompt)}`
                                    )
                                  )
                                }
                                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-text-secondary hover:text-white"
                                title="Download"
                              >
                                <Download className="w-4 h-4" />
                              </button>
                              <button
                                type="button"
                                onClick={() => handleSaveCandidate(candidate)}
                                className="p-2 rounded-lg bg-white/5 hover:bg-rose-500/20 text-text-secondary hover:text-rose-300"
                                title="Save to library"
                              >
                                <BookmarkPlus className="w-4 h-4" />
                              </button>
                            </div>
                          </div>
                        </motion.button>
                      );
                    })}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default SoundEffectsPage;
