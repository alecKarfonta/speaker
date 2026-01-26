import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Mic2,
    Wand2,
    Upload,
    User,
    Globe,
    Loader2,
    Play,
    Pause,
    Check,
    ChevronDown,
    Sparkles,
    X,
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import { cn } from '../../lib/utils';
import Button from '../ui/Button';

// ============================================================================
// Types
// ============================================================================

export interface QwenVoice {
    id: string;
    name: string;
    description: string;
    native_language: string;
    supported_languages: string[];
    gender: string;
    age_range: string;
    style_tags: string[];
    backend: string;
}

export type QwenTTSMode = 'custom_voice' | 'voice_design' | 'voice_clone';

interface QwenVoiceSelectorProps {
    onVoiceSelect?: (voice: QwenVoice) => void;
    onModeChange?: (mode: QwenTTSMode) => void;
    onSynthesize?: (params: SynthesizeParams) => void;
    selectedVoice?: QwenVoice | null;
    selectedMode?: QwenTTSMode;
    className?: string;
}

interface SynthesizeParams {
    mode: QwenTTSMode;
    speaker?: string;
    instruct?: string;
    refAudio?: File;
    refText?: string;
}

// ============================================================================
// Main Component
// ============================================================================

export const QwenVoiceSelector: React.FC<QwenVoiceSelectorProps> = ({
    onVoiceSelect,
    onModeChange,
    onSynthesize,
    selectedVoice: externalSelectedVoice,
    selectedMode: externalSelectedMode,
    className,
}) => {
    const [voices, setVoices] = useState<QwenVoice[]>([]);
    const [languages, setLanguages] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const [mode, setMode] = useState<QwenTTSMode>(externalSelectedMode || 'custom_voice');
    const [selectedVoice, setSelectedVoice] = useState<QwenVoice | null>(externalSelectedVoice || null);

    // Voice design state
    const [voiceDescription, setVoiceDescription] = useState('');

    // Voice clone state
    const [refAudio, setRefAudio] = useState<File | null>(null);
    const [refText, setRefText] = useState('');

    // Filter state
    const [languageFilter, setLanguageFilter] = useState<string>('all');
    const [genderFilter, setGenderFilter] = useState<string>('all');

    // Load speakers and languages
    const loadData = useCallback(async () => {
        setLoading(true);
        try {
            const [speakersRes, languagesRes] = await Promise.all([
                fetch('/api/v1/qwen/speakers'),
                fetch('/api/v1/qwen/languages'),
            ]);

            if (speakersRes.ok) {
                const data = await speakersRes.json();
                setVoices(data.speakers || []);
            }

            if (languagesRes.ok) {
                const data = await languagesRes.json();
                setLanguages(data.languages || []);
            }
        } catch (error) {
            console.error('Failed to load Qwen voices:', error);
            toast.error('Failed to load Qwen voices');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadData();
    }, [loadData]);

    // Sync with external props
    useEffect(() => {
        if (externalSelectedMode) setMode(externalSelectedMode);
    }, [externalSelectedMode]);

    useEffect(() => {
        if (externalSelectedVoice) setSelectedVoice(externalSelectedVoice);
    }, [externalSelectedVoice]);

    // Handle mode change
    const handleModeChange = (newMode: QwenTTSMode) => {
        setMode(newMode);
        onModeChange?.(newMode);
    };

    // Handle voice select
    const handleVoiceSelect = (voice: QwenVoice) => {
        setSelectedVoice(voice);
        onVoiceSelect?.(voice);
    };

    // Handle file upload
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setRefAudio(file);
        }
    };

    // Get synthesis params based on mode
    const getSynthesizeParams = (): SynthesizeParams => {
        switch (mode) {
            case 'custom_voice':
                return {
                    mode: 'custom_voice',
                    speaker: selectedVoice?.name || 'Vivian',
                };
            case 'voice_design':
                return {
                    mode: 'voice_design',
                    instruct: voiceDescription,
                };
            case 'voice_clone':
                return {
                    mode: 'voice_clone',
                    refAudio: refAudio || undefined,
                    refText: refText || undefined,
                };
        }
    };

    // Filter voices
    const filteredVoices = voices.filter(voice => {
        if (languageFilter !== 'all' && voice.native_language !== languageFilter) return false;
        if (genderFilter !== 'all' && voice.gender !== genderFilter) return false;
        return true;
    });

    const modeConfig = [
        { id: 'custom_voice' as const, label: 'Preset Voices', icon: User, description: '9 built-in speakers' },
        { id: 'voice_design' as const, label: 'Design Voice', icon: Wand2, description: 'Create from description' },
        { id: 'voice_clone' as const, label: 'Clone Voice', icon: Mic2, description: 'Clone from audio' },
    ];

    return (
        <div className={cn('space-y-6', className)}>
            {/* Mode Selection Tabs */}
            <div className="flex gap-2 p-1 bg-bg-secondary/50 rounded-xl">
                {modeConfig.map(({ id, label, icon: Icon }) => (
                    <button
                        key={id}
                        onClick={() => handleModeChange(id)}
                        className={cn(
                            'flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-lg transition-all',
                            mode === id
                                ? 'bg-accent text-white shadow-lg'
                                : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary'
                        )}
                    >
                        <Icon className="w-4 h-4" />
                        <span className="font-medium">{label}</span>
                    </button>
                ))}
            </div>

            {/* Loading State */}
            {loading && (
                <div className="flex items-center justify-center h-48">
                    <Loader2 className="w-8 h-8 animate-spin text-accent" />
                </div>
            )}

            {/* Custom Voice Mode - Voice Grid */}
            <AnimatePresence mode="wait">
                {!loading && mode === 'custom_voice' && (
                    <motion.div
                        key="custom_voice"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-4"
                    >
                        {/* Filters */}
                        <div className="flex gap-4">
                            <div className="flex items-center gap-2">
                                <Globe className="w-4 h-4 text-text-tertiary" />
                                <select
                                    value={languageFilter}
                                    onChange={(e) => setLanguageFilter(e.target.value)}
                                    className="bg-bg-secondary rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent/50"
                                >
                                    <option value="all">All Languages</option>
                                    {['Chinese', 'English', 'Japanese', 'Korean'].map(lang => (
                                        <option key={lang} value={lang}>{lang}</option>
                                    ))}
                                </select>
                            </div>

                            <div className="flex items-center gap-2">
                                <User className="w-4 h-4 text-text-tertiary" />
                                <select
                                    value={genderFilter}
                                    onChange={(e) => setGenderFilter(e.target.value)}
                                    className="bg-bg-secondary rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent/50"
                                >
                                    <option value="all">All Genders</option>
                                    <option value="female">Female</option>
                                    <option value="male">Male</option>
                                </select>
                            </div>
                        </div>

                        {/* Voice Grid */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                            {filteredVoices.map((voice, index) => (
                                <motion.div
                                    key={voice.id}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: index * 0.05 }}
                                    onClick={() => handleVoiceSelect(voice)}
                                    className={cn(
                                        'relative p-4 rounded-xl cursor-pointer transition-all',
                                        'bg-bg-secondary/50 hover:bg-bg-secondary border border-transparent',
                                        selectedVoice?.id === voice.id && 'border-accent ring-2 ring-accent/20'
                                    )}
                                >
                                    {/* Selected indicator */}
                                    {selectedVoice?.id === voice.id && (
                                        <div className="absolute top-3 right-3 w-6 h-6 rounded-full bg-accent flex items-center justify-center">
                                            <Check className="w-4 h-4 text-white" />
                                        </div>
                                    )}

                                    {/* Voice info */}
                                    <div className="flex items-start gap-3">
                                        <div className={cn(
                                            'w-10 h-10 rounded-xl flex items-center justify-center',
                                            voice.gender === 'female' ? 'bg-pink-500/20' : 'bg-blue-500/20'
                                        )}>
                                            <User className={cn(
                                                'w-5 h-5',
                                                voice.gender === 'female' ? 'text-pink-400' : 'text-blue-400'
                                            )} />
                                        </div>

                                        <div className="flex-1 min-w-0">
                                            <h4 className="font-semibold text-text-primary truncate">
                                                {voice.name}
                                            </h4>
                                            <p className="text-sm text-text-tertiary truncate">
                                                {voice.native_language} · {voice.gender}
                                            </p>
                                        </div>
                                    </div>

                                    {/* Style tags */}
                                    <div className="mt-3 flex flex-wrap gap-1">
                                        {voice.style_tags.slice(0, 3).map(tag => (
                                            <span
                                                key={tag}
                                                className="px-2 py-0.5 text-xs rounded-full bg-accent/10 text-accent"
                                            >
                                                {tag}
                                            </span>
                                        ))}
                                    </div>

                                    {/* Description */}
                                    <p className="mt-2 text-xs text-text-tertiary line-clamp-2">
                                        {voice.description}
                                    </p>
                                </motion.div>
                            ))}
                        </div>

                        {filteredVoices.length === 0 && (
                            <div className="text-center py-12 text-text-tertiary">
                                No voices match your filters
                            </div>
                        )}
                    </motion.div>
                )}

                {/* Voice Design Mode */}
                {!loading && mode === 'voice_design' && (
                    <motion.div
                        key="voice_design"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-4"
                    >
                        <div className="card p-6">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500">
                                    <Wand2 className="w-5 h-5 text-white" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-text-primary">Design Your Voice</h3>
                                    <p className="text-sm text-text-tertiary">
                                        Describe the voice you want to create
                                    </p>
                                </div>
                            </div>

                            <textarea
                                value={voiceDescription}
                                onChange={(e) => setVoiceDescription(e.target.value)}
                                placeholder="e.g., A warm, friendly female voice with a slight British accent, speaking at a moderate pace with gentle intonation..."
                                className="w-full min-h-[120px] bg-bg-tertiary rounded-xl p-4 text-text-primary placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/50 resize-none"
                            />

                            <div className="mt-4 p-3 bg-accent/10 rounded-lg text-sm text-text-secondary">
                                <Sparkles className="w-4 h-4 inline-block mr-2 text-accent" />
                                <strong>Tip:</strong> Include details about accent, tone, speaking pace, and emotional quality for best results.
                            </div>
                        </div>
                    </motion.div>
                )}

                {/* Voice Clone Mode */}
                {!loading && mode === 'voice_clone' && (
                    <motion.div
                        key="voice_clone"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-4"
                    >
                        <div className="card p-6">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-2 rounded-xl bg-gradient-to-br from-green-500 to-emerald-500">
                                    <Mic2 className="w-5 h-5 text-white" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-text-primary">Clone a Voice</h3>
                                    <p className="text-sm text-text-tertiary">
                                        Upload 3+ seconds of clear reference audio
                                    </p>
                                </div>
                            </div>

                            {/* File upload area */}
                            <label className={cn(
                                'block border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all',
                                refAudio
                                    ? 'border-accent bg-accent/5'
                                    : 'border-border hover:border-accent/50 hover:bg-bg-tertiary'
                            )}>
                                <input
                                    type="file"
                                    accept="audio/*"
                                    onChange={handleFileChange}
                                    className="hidden"
                                />

                                {refAudio ? (
                                    <div className="flex items-center justify-center gap-3">
                                        <Check className="w-6 h-6 text-accent" />
                                        <span className="text-text-primary font-medium">{refAudio.name}</span>
                                        <button
                                            onClick={(e) => {
                                                e.preventDefault();
                                                setRefAudio(null);
                                            }}
                                            className="text-text-tertiary hover:text-red-400"
                                        >
                                            <X className="w-5 h-5" />
                                        </button>
                                    </div>
                                ) : (
                                    <>
                                        <Upload className="w-8 h-8 mx-auto mb-3 text-text-tertiary" />
                                        <p className="text-text-secondary font-medium">
                                            Click to upload or drag and drop
                                        </p>
                                        <p className="text-sm text-text-tertiary mt-1">
                                            WAV, MP3, or other audio formats
                                        </p>
                                    </>
                                )}
                            </label>

                            {/* Reference text (optional) */}
                            <div className="mt-4">
                                <label className="block text-sm font-medium text-text-secondary mb-2">
                                    Reference Text (Optional)
                                </label>
                                <textarea
                                    value={refText}
                                    onChange={(e) => setRefText(e.target.value)}
                                    placeholder="Transcript of the reference audio for better quality..."
                                    className="w-full min-h-[80px] bg-bg-tertiary rounded-xl p-4 text-text-primary placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/50 resize-none"
                                />
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Selected voice summary / action button */}
            {!loading && (
                <div className="flex items-center justify-between p-4 bg-bg-secondary/50 rounded-xl">
                    <div className="text-sm">
                        {mode === 'custom_voice' && selectedVoice && (
                            <span className="text-text-secondary">
                                Selected: <span className="text-text-primary font-medium">{selectedVoice.name}</span>
                            </span>
                        )}
                        {mode === 'voice_design' && (
                            <span className="text-text-secondary">
                                Mode: <span className="text-text-primary font-medium">Voice Design</span>
                            </span>
                        )}
                        {mode === 'voice_clone' && (
                            <span className="text-text-secondary">
                                Mode: <span className="text-text-primary font-medium">Voice Clone</span>
                                {refAudio && <span className="text-accent ml-2">✓ Audio ready</span>}
                            </span>
                        )}
                    </div>

                    {onSynthesize && (
                        <Button
                            onClick={() => onSynthesize(getSynthesizeParams())}
                            disabled={
                                (mode === 'custom_voice' && !selectedVoice) ||
                                (mode === 'voice_design' && !voiceDescription.trim()) ||
                                (mode === 'voice_clone' && !refAudio)
                            }
                        >
                            <Sparkles className="w-4 h-4" />
                            Use This Voice
                        </Button>
                    )}
                </div>
            )}
        </div>
    );
};

export default QwenVoiceSelector;
