import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Mic,
    Wand2,
    Upload,
    Layers,
    Play,
    Pause,
    Download,
    Volume2,
    User,
    Sparkles,
    Copy,
    ChevronDown,
    RefreshCw,
    X,
    Check,
    Music,
    Settings2,
    Save,
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import Layout from '../layout/Layout';
import Button from '../ui/Button';

// API Base URL - use relative path for nginx proxy (empty string = relative to current host)
const API_BASE = '';

// Types
interface QwenSpeaker {
    id: string;
    name: string;
    description: string;
    native_language: string;
    gender: string;
}

interface GeneratedAudio {
    blob: Blob;
    url: string;
    duration: number;
}

interface GenerationHistoryItem {
    id: string;
    blob: Blob;
    url: string;
    duration: number;
    timestamp: Date;
    mode: 'speaker' | 'design' | 'clone';
    text: string;
    speaker?: string;
    instruct?: string;
    savedToCatalog?: boolean;
    savedVoiceName?: string;
}

// Tab definitions
type StudioTab = 'speakers' | 'design' | 'clone';

const TABS: { id: StudioTab; label: string; icon: React.ReactNode; description: string }[] = [
    { id: 'speakers', label: 'Speakers', icon: <User size={18} />, description: 'Built-in voices with style control' },
    { id: 'design', label: 'Design', icon: <Wand2 size={18} />, description: 'Create voice from description' },
    { id: 'clone', label: 'Clone', icon: <Copy size={18} />, description: 'Clone from audio sample' },
];

// Voice design presets
const DESIGN_PRESETS = [
    'A warm, friendly female voice with a moderate pace',
    'A deep, authoritative male voice like a news anchor',
    'An energetic young voice with enthusiasm',
    'A calm, soothing voice perfect for meditation',
    'A playful, animated voice for storytelling',
    'A professional, clear voice for presentations',
];

// Default speakers (fallback if API fails)
const DEFAULT_SPEAKERS: QwenSpeaker[] = [
    { id: 'vivian', name: 'Vivian', description: 'Bright, edgy young female', native_language: 'Chinese', gender: 'female' },
    { id: 'serena', name: 'Serena', description: 'Warm, gentle young female', native_language: 'Chinese', gender: 'female' },
    { id: 'ryan', name: 'Ryan', description: 'Dynamic male with rhythm', native_language: 'English', gender: 'male' },
    { id: 'aiden', name: 'Aiden', description: 'Sunny American male', native_language: 'English', gender: 'male' },
    { id: 'uncle_fu', name: 'Uncle Fu', description: 'Deep, mellow mature male', native_language: 'Chinese', gender: 'male' },
    { id: 'dylan', name: 'Dylan', description: 'Youthful Beijing male', native_language: 'Chinese', gender: 'male' },
    { id: 'eric', name: 'Eric', description: 'Lively Chengdu male', native_language: 'Chinese', gender: 'male' },
    { id: 'ono_anna', name: 'Ono Anna', description: 'Playful Japanese female', native_language: 'Japanese', gender: 'female' },
    { id: 'sohee', name: 'Sohee', description: 'Warm Korean female', native_language: 'Korean', gender: 'female' },
];

const VoiceStudio: React.FC = () => {
    const [activeTab, setActiveTab] = useState<StudioTab>('speakers');
    const [isLoading, setIsLoading] = useState(false);
    const [generatedAudio, setGeneratedAudio] = useState<GeneratedAudio | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [generationHistory, setGenerationHistory] = useState<GenerationHistoryItem[]>([]);
    const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    // Speakers tab state - start with defaults, update from API if available
    const [speakers, setSpeakers] = useState<QwenSpeaker[]>(DEFAULT_SPEAKERS);
    const [selectedSpeaker, setSelectedSpeaker] = useState('Vivian');
    const [speakerInstruct, setSpeakerInstruct] = useState('');
    const [speakerText, setSpeakerText] = useState('Hello! This is a test of the Qwen text to speech system.');

    // Design tab state
    const [designInstruct, setDesignInstruct] = useState('');
    const [designText, setDesignText] = useState('Welcome to the voice design studio where you can create any voice you imagine.');

    // Clone tab state
    const [cloneAudioFile, setCloneAudioFile] = useState<File | null>(null);
    const [cloneAudioUrl, setCloneAudioUrl] = useState<string | null>(null);
    const [cloneText, setCloneText] = useState('Testing voice cloning with this sample text.');
    const [useXvectorOnly, setUseXvectorOnly] = useState(true);
    const [cloneVoiceName, setCloneVoiceName] = useState('');
    const [saveToVoiceCatalog, setSaveToVoiceCatalog] = useState(true);

    const [language, setLanguage] = useState('English');

    // Fetch speakers on mount (optional - we have defaults)
    useEffect(() => {
        fetchSpeakers();
    }, []);

    const fetchSpeakers = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/v1/qwen/speakers`);
            if (response.ok) {
                const data = await response.json();
                if (data.speakers && data.speakers.length > 0) {
                    setSpeakers(data.speakers);
                }
            }
        } catch (error) {
            console.log('Using default speakers (API not reachable)');
        }
    };

    // Audio playback - update audio source when URL changes
    const handlePlay = () => {
        if (!generatedAudio?.url) return;

        // Always update the audio source to match current generatedAudio
        if (!audioRef.current) {
            audioRef.current = new Audio(generatedAudio.url);
            audioRef.current.onended = () => setIsPlaying(false);
        } else if (audioRef.current.src !== generatedAudio.url) {
            // URL changed - update the source
            audioRef.current.pause();
            audioRef.current.src = generatedAudio.url;
            audioRef.current.load();
        }

        if (isPlaying) {
            audioRef.current.pause();
            setIsPlaying(false);
        } else {
            audioRef.current.play();
            setIsPlaying(true);
        }
    };

    // Select a history item for playback
    const selectHistoryItem = (item: GenerationHistoryItem) => {
        // Stop current playback
        if (audioRef.current) {
            audioRef.current.pause();
            setIsPlaying(false);
        }

        setGeneratedAudio({ blob: item.blob, url: item.url, duration: item.duration });
        setSelectedHistoryId(item.id);
    };

    // Save a generated voice to the catalog (for review-first workflow)
    const saveHistoryItemToCatalog = async (item: GenerationHistoryItem, voiceName: string) => {
        if (!voiceName.trim()) {
            toast.error('Please enter a voice name');
            return;
        }

        try {
            // Create a file from the blob
            const file = new File([item.blob], 'voice_sample.wav', { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('ref_audio', file);

            const params = new URLSearchParams({
                voice_id: voiceName.trim(),
            });

            const response = await fetch(`${API_BASE}/api/v1/qwen/voices/create-prompt?${params}`, {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                // Update the history item to mark as saved
                setGenerationHistory(prev => prev.map(h =>
                    h.id === item.id
                        ? { ...h, savedToCatalog: true, savedVoiceName: voiceName }
                        : h
                ));
                toast.success(`Voice "${voiceName}" saved to catalog!`);
                // Refresh speakers list
                fetchSpeakers();
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to save voice');
            }
        } catch (error: any) {
            toast.error(error.message || 'Failed to save voice to catalog');
        }
    };

    const handleDownload = () => {
        if (!generatedAudio?.blob) return;
        const url = URL.createObjectURL(generatedAudio.blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `voice_studio_${Date.now()}.wav`;
        a.click();
        URL.revokeObjectURL(url);
    };

    // Generate with built-in speaker
    const generateSpeaker = async () => {
        if (!speakerText.trim()) {
            toast.error('Please enter text to synthesize');
            return;
        }

        setIsLoading(true);
        try {
            const response = await fetch(`${API_BASE}/api/v1/qwen/synthesize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: speakerText,
                    mode: 'custom_voice',
                    speaker: selectedSpeaker,
                    language,
                    instruct: speakerInstruct || undefined,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || error.message || 'Synthesis failed');
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const duration = parseFloat(response.headers.get('X-Audio-Duration') || '0');

            const newAudio = { blob, url, duration };
            setGeneratedAudio(newAudio);

            // Add to history
            const historyItem: GenerationHistoryItem = {
                id: `gen-${Date.now()}`,
                ...newAudio,
                timestamp: new Date(),
                mode: 'speaker',
                text: speakerText,
                speaker: selectedSpeaker,
                instruct: speakerInstruct || undefined,
            };
            setGenerationHistory(prev => [historyItem, ...prev]);
            setSelectedHistoryId(historyItem.id);

            toast.success('Audio generated!');
        } catch (error: any) {
            toast.error(error.message || 'Failed to generate audio');
        } finally {
            setIsLoading(false);
        }
    };

    // Generate with voice design
    const generateDesign = async () => {
        if (!designText.trim() || !designInstruct.trim()) {
            toast.error('Please enter both text and voice description');
            return;
        }

        setIsLoading(true);
        try {
            const response = await fetch(`${API_BASE}/api/v1/qwen/synthesize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: designText,
                    mode: 'voice_design',
                    instruct: designInstruct,
                    language,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || error.message || 'Voice design failed');
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const duration = parseFloat(response.headers.get('X-Audio-Duration') || '0');

            const newAudio = { blob, url, duration };
            setGeneratedAudio(newAudio);

            // Add to history
            const historyItem: GenerationHistoryItem = {
                id: `gen-${Date.now()}`,
                ...newAudio,
                timestamp: new Date(),
                mode: 'design',
                text: designText,
                instruct: designInstruct,
            };
            setGenerationHistory(prev => [historyItem, ...prev]);
            setSelectedHistoryId(historyItem.id);

            toast.success('Voice designed and generated!');
        } catch (error: any) {
            toast.error(error.message || 'Failed to design voice');
        } finally {
            setIsLoading(false);
        }
    };

    // Generate with voice clone
    const generateClone = async () => {
        if (!cloneAudioFile || !cloneText.trim()) {
            toast.error('Please upload an audio file and enter text');
            return;
        }

        // Validate voice name if saving
        if (saveToVoiceCatalog && !cloneVoiceName.trim()) {
            toast.error('Please enter a name for the cloned voice');
            return;
        }

        setIsLoading(true);
        try {
            // First, save the voice if requested
            if (saveToVoiceCatalog && cloneVoiceName.trim()) {
                const saveFormData = new FormData();
                saveFormData.append('ref_audio', cloneAudioFile);

                const saveParams = new URLSearchParams({
                    voice_id: cloneVoiceName.trim(),
                });

                const saveResponse = await fetch(`${API_BASE}/api/v1/qwen/voices/create-prompt?${saveParams}`, {
                    method: 'POST',
                    body: saveFormData,
                });

                if (saveResponse.ok) {
                    toast.success(`Voice "${cloneVoiceName}" saved to catalog!`);
                    // Refresh speakers list
                    fetchSpeakers();
                }
            }

            // Now generate the audio
            const formData = new FormData();
            formData.append('ref_audio', cloneAudioFile);

            const params = new URLSearchParams({
                text: cloneText,
                language,
                use_xvector_only: String(useXvectorOnly),
            });

            const response = await fetch(`${API_BASE}/api/v1/qwen/clone?${params}`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || error.message || 'Voice cloning failed');
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const duration = parseFloat(response.headers.get('X-Audio-Duration') || '0');

            const newAudio = { blob, url, duration };
            setGeneratedAudio(newAudio);

            // Add to history
            const historyItem: GenerationHistoryItem = {
                id: `gen-${Date.now()}`,
                ...newAudio,
                timestamp: new Date(),
                mode: 'clone',
                text: cloneText,
            };
            setGenerationHistory(prev => [historyItem, ...prev]);
            setSelectedHistoryId(historyItem.id);

            toast.success('Voice cloned and generated!');
        } catch (error: any) {
            toast.error(error.message || 'Failed to clone voice');
        } finally {
            setIsLoading(false);
        }
    };

    // Handle audio file upload
    const handleAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setCloneAudioFile(file);
            setCloneAudioUrl(URL.createObjectURL(file));
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('audio/')) {
            setCloneAudioFile(file);
            setCloneAudioUrl(URL.createObjectURL(file));
        }
    };

    return (
        <Layout>
            <div className="h-full flex flex-col bg-background">
                {/* Header */}
                <div className="px-6 py-4 border-b border-border-default">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600">
                            <Sparkles className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold text-text-primary">Voice Studio</h1>
                            <p className="text-sm text-text-secondary">Create, design, and clone voices with Qwen3-TTS</p>
                        </div>
                    </div>
                </div>

                {/* Main Content */}
                <div className="flex-1 flex overflow-hidden">
                    {/* Left Panel - Controls */}
                    <div className="w-1/2 border-r border-border-default flex flex-col overflow-hidden">
                        {/* Tabs */}
                        <div className="flex border-b border-border-default">
                            {TABS.map((tab) => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`flex-1 px-4 py-3 flex items-center justify-center gap-2 text-sm font-medium transition-colors ${activeTab === tab.id
                                        ? 'text-accent-primary border-b-2 border-accent-primary bg-bg-secondary'
                                        : 'text-text-secondary hover:text-text-primary hover:bg-bg-secondary/50'
                                        }`}
                                >
                                    {tab.icon}
                                    {tab.label}
                                </button>
                            ))}
                        </div>

                        {/* Tab Content */}
                        <div className="flex-1 overflow-y-auto p-6 space-y-6">
                            {/* Speakers Tab */}
                            {activeTab === 'speakers' && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="space-y-6"
                                >
                                    <div>
                                        <label className="block text-sm font-medium text-text-primary mb-2">
                                            Select Speaker
                                        </label>
                                        <div className="grid grid-cols-3 gap-2">
                                            {speakers.map((speaker) => (
                                                <button
                                                    key={speaker.name}
                                                    onClick={() => setSelectedSpeaker(speaker.name)}
                                                    className={`p-3 rounded-lg border text-left transition-all ${selectedSpeaker === speaker.name
                                                        ? 'border-accent-primary bg-accent-primary/10 ring-1 ring-accent-primary'
                                                        : 'border-border-default hover:border-border-hover bg-bg-secondary'
                                                        }`}
                                                >
                                                    <div className="font-medium text-text-primary text-sm">{speaker.name}</div>
                                                    <div className="text-xs text-text-secondary truncate">{speaker.description}</div>
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    <div>
                                        <label className="block text-sm font-medium text-text-primary mb-2">
                                            Style Instruction (Optional)
                                        </label>
                                        <input
                                            type="text"
                                            value={speakerInstruct}
                                            onChange={(e) => setSpeakerInstruct(e.target.value)}
                                            placeholder="e.g., Speak angrily, Use a calm tone, Very happy..."
                                            className="w-full px-4 py-3 rounded-lg border border-border-default bg-bg-secondary text-text-primary placeholder-text-tertiary focus:border-accent-primary focus:ring-1 focus:ring-accent-primary"
                                        />
                                        <p className="mt-1.5 text-xs text-text-tertiary">
                                            💡 <strong>Requires 1.7B model</strong> for best results. The 0.6B model has limited style control.
                                            Try prompts like "Speak angrily", "Very happy", or "Whisper softly".
                                        </p>
                                    </div>

                                    <div>
                                        <label className="block text-sm font-medium text-text-primary mb-2">
                                            Text to Speak
                                        </label>
                                        <textarea
                                            value={speakerText}
                                            onChange={(e) => setSpeakerText(e.target.value)}
                                            rows={4}
                                            className="w-full px-4 py-3 rounded-lg border border-border-default bg-bg-secondary text-text-primary placeholder-text-tertiary focus:border-accent-primary focus:ring-1 focus:ring-accent-primary resize-none"
                                        />
                                    </div>

                                    <Button
                                        onClick={generateSpeaker}
                                        disabled={isLoading || !speakerText.trim()}
                                        className="w-full"
                                        variant="primary"
                                    >
                                        {isLoading ? (
                                            <RefreshCw className="w-4 h-4 animate-spin mr-2" />
                                        ) : (
                                            <Play className="w-4 h-4 mr-2" />
                                        )}
                                        Generate Speech
                                    </Button>
                                </motion.div>
                            )}

                            {/* Design Tab */}
                            {activeTab === 'design' && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="space-y-6"
                                >
                                    <div>
                                        <label className="block text-sm font-medium text-text-primary mb-2">
                                            Describe Your Voice
                                        </label>
                                        <textarea
                                            value={designInstruct}
                                            onChange={(e) => setDesignInstruct(e.target.value)}
                                            rows={3}
                                            placeholder="Describe the voice you want to create..."
                                            className="w-full px-4 py-3 rounded-lg border border-border-default bg-bg-secondary text-text-primary placeholder-text-tertiary focus:border-accent-primary focus:ring-1 focus:ring-accent-primary resize-none"
                                        />
                                        <div className="mt-2 flex flex-wrap gap-2">
                                            {DESIGN_PRESETS.slice(0, 4).map((preset, i) => (
                                                <button
                                                    key={i}
                                                    onClick={() => setDesignInstruct(preset)}
                                                    className="px-2 py-1 text-xs rounded-full bg-accent-primary/10 text-accent-primary hover:bg-accent-primary/20 transition-colors"
                                                >
                                                    {preset.slice(0, 30)}...
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    <div>
                                        <label className="block text-sm font-medium text-text-primary mb-2">
                                            Text to Speak
                                        </label>
                                        <textarea
                                            value={designText}
                                            onChange={(e) => setDesignText(e.target.value)}
                                            rows={4}
                                            className="w-full px-4 py-3 rounded-lg border border-border-default bg-bg-secondary text-text-primary placeholder-text-tertiary focus:border-accent-primary focus:ring-1 focus:ring-accent-primary resize-none"
                                        />
                                    </div>

                                    <Button
                                        onClick={generateDesign}
                                        disabled={isLoading || !designText.trim() || !designInstruct.trim()}
                                        className="w-full"
                                        variant="primary"
                                    >
                                        {isLoading ? (
                                            <RefreshCw className="w-4 h-4 animate-spin mr-2" />
                                        ) : (
                                            <Wand2 className="w-4 h-4 mr-2" />
                                        )}
                                        Design & Generate
                                    </Button>
                                </motion.div>
                            )}

                            {/* Clone Tab */}
                            {activeTab === 'clone' && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="space-y-6"
                                >
                                    <div>
                                        <label className="block text-sm font-medium text-text-primary mb-2">
                                            Reference Audio (3+ seconds)
                                        </label>
                                        <div
                                            onDrop={handleDrop}
                                            onDragOver={(e) => e.preventDefault()}
                                            className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${cloneAudioFile
                                                ? 'border-success bg-success/5'
                                                : 'border-border-default hover:border-accent-primary'
                                                }`}
                                        >
                                            {cloneAudioFile ? (
                                                <div className="space-y-2">
                                                    <Check className="w-8 h-8 mx-auto text-success" />
                                                    <p className="text-sm font-medium text-text-primary">{cloneAudioFile.name}</p>
                                                    <p className="text-xs text-text-secondary">
                                                        {(cloneAudioFile.size / 1024).toFixed(1)} KB
                                                    </p>
                                                    {cloneAudioUrl && (
                                                        <audio controls src={cloneAudioUrl} className="w-full mt-2" />
                                                    )}
                                                    <button
                                                        onClick={() => {
                                                            setCloneAudioFile(null);
                                                            setCloneAudioUrl(null);
                                                        }}
                                                        className="text-xs text-error hover:underline"
                                                    >
                                                        Remove
                                                    </button>
                                                </div>
                                            ) : (
                                                <div className="space-y-2">
                                                    <Upload className="w-8 h-8 mx-auto text-text-tertiary" />
                                                    <p className="text-sm text-text-secondary">
                                                        Drag & drop audio file or{' '}
                                                        <label className="text-accent-primary cursor-pointer hover:underline">
                                                            browse
                                                            <input
                                                                type="file"
                                                                accept="audio/*"
                                                                onChange={handleAudioUpload}
                                                                className="hidden"
                                                            />
                                                        </label>
                                                    </p>
                                                    <p className="text-xs text-text-tertiary">
                                                        WAV, MP3, or other audio formats
                                                    </p>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-2">
                                        <input
                                            type="checkbox"
                                            id="xvector"
                                            checked={useXvectorOnly}
                                            onChange={(e) => setUseXvectorOnly(e.target.checked)}
                                            className="rounded border-border-default"
                                        />
                                        <label htmlFor="xvector" className="text-sm text-text-secondary">
                                            X-vector only mode (faster, captures voice timbre)
                                        </label>
                                    </div>

                                    {/* Voice Name & Save to Catalog */}
                                    <div className="p-4 bg-bg-tertiary/50 rounded-lg border border-border-default space-y-3">
                                        <div className="flex items-center gap-2">
                                            <input
                                                type="checkbox"
                                                id="saveToCatalog"
                                                checked={saveToVoiceCatalog}
                                                onChange={(e) => setSaveToVoiceCatalog(e.target.checked)}
                                                className="rounded border-border-default"
                                            />
                                            <label htmlFor="saveToCatalog" className="text-sm font-medium text-text-primary">
                                                Save voice to catalog
                                            </label>
                                        </div>

                                        {saveToVoiceCatalog && (
                                            <div>
                                                <label className="block text-xs text-text-secondary mb-1">
                                                    Voice Name
                                                </label>
                                                <input
                                                    type="text"
                                                    value={cloneVoiceName}
                                                    onChange={(e) => setCloneVoiceName(e.target.value)}
                                                    placeholder="e.g., My Custom Voice"
                                                    className="w-full px-3 py-2 rounded-lg border border-border-default bg-bg-secondary text-text-primary placeholder-text-tertiary focus:border-accent-primary focus:ring-1 focus:ring-accent-primary text-sm"
                                                />
                                                <p className="text-xs text-text-tertiary mt-1">
                                                    Saved voices appear in the Speakers tab
                                                </p>
                                            </div>
                                        )}
                                    </div>

                                    <div>
                                        <label className="block text-sm font-medium text-text-primary mb-2">
                                            Text to Speak
                                        </label>
                                        <textarea
                                            value={cloneText}
                                            onChange={(e) => setCloneText(e.target.value)}
                                            rows={4}
                                            className="w-full px-4 py-3 rounded-lg border border-border-default bg-bg-secondary text-text-primary placeholder-text-tertiary focus:border-accent-primary focus:ring-1 focus:ring-accent-primary resize-none"
                                        />
                                    </div>

                                    <Button
                                        onClick={generateClone}
                                        disabled={isLoading || !cloneAudioFile || !cloneText.trim() || (saveToVoiceCatalog && !cloneVoiceName.trim())}
                                        className="w-full"
                                        variant="primary"
                                    >
                                        {isLoading ? (
                                            <RefreshCw className="w-4 h-4 animate-spin mr-2" />
                                        ) : (
                                            <Copy className="w-4 h-4 mr-2" />
                                        )}
                                        {saveToVoiceCatalog ? 'Clone, Save & Generate' : 'Clone & Generate'}
                                    </Button>
                                </motion.div>
                            )}
                        </div>
                    </div>

                    {/* Right Panel - Preview & History */}
                    <div className="w-1/2 flex flex-col bg-bg-secondary/30">
                        <div className="p-6 border-b border-border-default">
                            <h2 className="text-lg font-semibold text-text-primary flex items-center gap-2">
                                <Volume2 className="w-5 h-5" />
                                Preview
                            </h2>
                        </div>

                        <div className="flex-1 flex flex-col overflow-hidden">
                            {/* Current Audio Preview */}
                            <div className="p-6 border-b border-border-default">
                                {generatedAudio ? (
                                    <motion.div
                                        initial={{ opacity: 0, scale: 0.95 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        className="space-y-4"
                                    >
                                        {/* Audio Visualizer */}
                                        <div className="h-24 bg-gradient-to-br from-purple-900/30 to-pink-900/30 rounded-xl flex items-center justify-center border border-border-default">
                                            <div className="flex items-end gap-1">
                                                {[...Array(16)].map((_, i) => (
                                                    <motion.div
                                                        key={i}
                                                        className="w-1.5 bg-gradient-to-t from-purple-500 to-pink-500 rounded-full"
                                                        animate={{
                                                            height: isPlaying ? [8, 24 + Math.random() * 32, 8] : 8,
                                                        }}
                                                        transition={{
                                                            duration: 0.4,
                                                            repeat: isPlaying ? Infinity : 0,
                                                            delay: i * 0.04,
                                                        }}
                                                    />
                                                ))}
                                            </div>
                                        </div>

                                        {/* Controls */}
                                        <div className="flex items-center justify-center gap-3">
                                            <Button
                                                onClick={handlePlay}
                                                variant="primary"
                                                className="rounded-full w-12 h-12"
                                            >
                                                {isPlaying ? (
                                                    <Pause className="w-5 h-5" />
                                                ) : (
                                                    <Play className="w-5 h-5 ml-0.5" />
                                                )}
                                            </Button>
                                            <Button
                                                onClick={handleDownload}
                                                variant="secondary"
                                                className="rounded-full w-10 h-10"
                                            >
                                                <Download className="w-4 h-4" />
                                            </Button>
                                        </div>

                                        {generatedAudio.duration > 0 && (
                                            <p className="text-center text-sm text-text-secondary">
                                                Duration: {generatedAudio.duration.toFixed(1)}s
                                            </p>
                                        )}
                                    </motion.div>
                                ) : (
                                    <div className="text-center text-text-tertiary py-8">
                                        <Music className="w-12 h-12 mx-auto mb-3 opacity-30" />
                                        <p className="text-sm">Generate audio to preview</p>
                                    </div>
                                )}
                            </div>

                            {/* Generation History */}
                            <div className="flex-1 overflow-y-auto">
                                <div className="p-4 border-b border-border-default sticky top-0 bg-bg-secondary/50 backdrop-blur-sm">
                                    <h3 className="text-sm font-medium text-text-secondary flex items-center gap-2">
                                        <Layers className="w-4 h-4" />
                                        Generation History ({generationHistory.length})
                                    </h3>
                                </div>

                                {generationHistory.length === 0 ? (
                                    <div className="p-6 text-center text-text-tertiary">
                                        <p className="text-sm">No generations yet</p>
                                    </div>
                                ) : (
                                    <div className="divide-y divide-border-default">
                                        {generationHistory.map((item) => (
                                            <motion.div
                                                key={item.id}
                                                className={`p-4 transition-colors hover:bg-bg-secondary/50 ${selectedHistoryId === item.id ? 'bg-accent-primary/10 border-l-2 border-accent-primary' : ''}`}
                                                initial={{ opacity: 0, y: -10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                            >
                                                <button
                                                    onClick={() => selectHistoryItem(item)}
                                                    className="w-full text-left"
                                                >
                                                    <div className="flex items-start justify-between gap-3">
                                                        <div className="flex-1 min-w-0">
                                                            <div className="flex items-center gap-2 mb-1 flex-wrap">
                                                                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${item.mode === 'speaker' ? 'bg-blue-500/20 text-blue-400' :
                                                                    item.mode === 'design' ? 'bg-purple-500/20 text-purple-400' :
                                                                        'bg-green-500/20 text-green-400'
                                                                    }`}>
                                                                    {item.mode === 'speaker' ? 'Speaker' : item.mode === 'design' ? 'Design' : 'Clone'}
                                                                </span>
                                                                {item.speaker && (
                                                                    <span className="text-xs text-text-secondary">{item.speaker}</span>
                                                                )}
                                                                {item.savedToCatalog && (
                                                                    <span className="text-xs px-2 py-0.5 rounded-full bg-green-500/20 text-green-400 flex items-center gap-1">
                                                                        <Check className="w-3 h-3" /> {item.savedVoiceName || 'Saved'}
                                                                    </span>
                                                                )}
                                                            </div>
                                                            <p className="text-sm text-text-primary truncate">
                                                                {item.text.substring(0, 60)}{item.text.length > 60 ? '...' : ''}
                                                            </p>
                                                            <div className="flex items-center gap-3 mt-1 text-xs text-text-tertiary">
                                                                <span>{item.duration.toFixed(1)}s</span>
                                                                <span>{item.timestamp.toLocaleTimeString()}</span>
                                                            </div>
                                                        </div>
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                selectHistoryItem(item);
                                                                setTimeout(handlePlay, 100);
                                                            }}
                                                            className="p-2 rounded-full bg-accent-primary/20 hover:bg-accent-primary/30 text-accent-primary shrink-0"
                                                        >
                                                            <Play className="w-3 h-3" />
                                                        </button>
                                                    </div>
                                                </button>

                                                {/* Save to Catalog button - only for design/clone modes that haven't been saved */}
                                                {(item.mode === 'design' || item.mode === 'clone') && !item.savedToCatalog && (
                                                    <div className="mt-3 pt-3 border-t border-border-default">
                                                        <div className="flex items-center gap-2">
                                                            <input
                                                                type="text"
                                                                placeholder="Voice name..."
                                                                className="flex-1 px-2 py-1.5 text-xs rounded border border-border-default bg-bg-secondary text-text-primary placeholder-text-tertiary"
                                                                onClick={(e) => e.stopPropagation()}
                                                                onKeyDown={(e) => {
                                                                    if (e.key === 'Enter') {
                                                                        e.stopPropagation();
                                                                        saveHistoryItemToCatalog(item, (e.target as HTMLInputElement).value);
                                                                    }
                                                                }}
                                                                id={`voice-name-${item.id}`}
                                                            />
                                                            <button
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    const input = document.getElementById(`voice-name-${item.id}`) as HTMLInputElement;
                                                                    if (input) {
                                                                        saveHistoryItemToCatalog(item, input.value);
                                                                    }
                                                                }}
                                                                className="px-2 py-1.5 text-xs bg-accent-primary/20 hover:bg-accent-primary/30 text-accent-primary rounded flex items-center gap-1 whitespace-nowrap"
                                                            >
                                                                <Save className="w-3 h-3" />
                                                                Save
                                                            </button>
                                                        </div>
                                                    </div>
                                                )}
                                            </motion.div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </Layout>
    );
};

export default VoiceStudio;
