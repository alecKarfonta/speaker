import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    X, User, Eye, Palette, Shirt, Ruler, Mic, RefreshCw,
    Play, Pause, Loader2, Image, Plus, Check, Edit3, Save, Sparkles,
    Wand2, Volume2,
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import type { CharacterRef, ProjectDetail } from '../../services/audiobookApi';
import {
    getPortraitUrl,
    generatePortraitVariant,
    selectPortrait,
    updateCharacter,
    designVoice,
} from '../../services/audiobookApi';

interface CharacterProfileModalProps {
    character: CharacterRef;
    projectId: string;
    voiceMap: Record<string, string>;
    availableVoices: string[];
    onVoiceChange: (name: string, voice: string) => void;
    onProjectUpdate: (project: ProjectDetail) => void;
    onClose: () => void;
}

const PROFILE_ICONS: Record<string, React.ReactNode> = {
    face: <Eye size={12} />,
    skin: <Palette size={12} />,
    build: <Ruler size={12} />,
    clothing: <Shirt size={12} />,
};

const PROFILE_COLORS: Record<string, string> = {
    face: 'from-blue-500/20 to-cyan-500/20 border-blue-500/15 text-blue-300',
    skin: 'from-amber-500/20 to-orange-500/20 border-amber-500/15 text-amber-300',
    build: 'from-emerald-500/20 to-green-500/20 border-emerald-500/15 text-emerald-300',
    clothing: 'from-violet-500/20 to-purple-500/20 border-violet-500/15 text-violet-300',
};

const CharacterProfileModal: React.FC<CharacterProfileModalProps> = ({
    character,
    projectId,
    voiceMap,
    availableVoices,
    onVoiceChange,
    onProjectUpdate,
    onClose,
}) => {
    const [selectedVariant, setSelectedVariant] = useState(-1);
    const [generatingPortrait, setGeneratingPortrait] = useState(false);
    const [designingVoice, setDesigningVoice] = useState(false);

    // Editing states
    const [editingDesc, setEditingDesc] = useState(false);
    const [descText, setDescText] = useState(character.description || '');
    const [editingPortraitPrompt, setEditingPortraitPrompt] = useState(false);
    const [portraitPromptText, setPortraitPromptText] = useState(character.portrait_prompt || '');
    const [editingVoicePrompt, setEditingVoicePrompt] = useState(false);
    const [voicePromptText, setVoicePromptText] = useState(character.voice_prompt || '');
    const [saving, setSaving] = useState(false);

    const [playingVoice, setPlayingVoice] = useState(false);
    const audioRef = useRef<HTMLAudioElement>(null);

    const voiceName = voiceMap[character.name] || '';
    const variants = character.portrait_variants || [];
    const hasPortrait = !!character.portrait_path;
    const profile = character.visual_profile;

    const portraitSrc = hasPortrait
        ? getPortraitUrl(projectId, character.name, selectedVariant)
        : null;

    // ── Handlers ──

    const handleGenerateVariant = async () => {
        setGeneratingPortrait(true);
        try {
            const updated = await generatePortraitVariant(projectId, character.name);
            onProjectUpdate(updated);
            toast.success('New portrait variant generated!');
        } catch (e: any) {
            toast.error(e.message || 'Portrait generation failed');
        } finally {
            setGeneratingPortrait(false);
        }
    };

    const handleSelectVariant = async (idx: number) => {
        setSelectedVariant(idx);
        try {
            const updated = await selectPortrait(projectId, character.name, idx);
            onProjectUpdate(updated);
        } catch (e: any) {
            toast.error(e.message || 'Failed to select portrait');
        }
    };

    const handleSaveField = async (field: 'description' | 'portrait_prompt' | 'voice_prompt') => {
        setSaving(true);
        try {
            const fields: any = {};
            if (field === 'description') fields.description = descText;
            if (field === 'portrait_prompt') fields.portrait_prompt = portraitPromptText;
            if (field === 'voice_prompt') fields.voice_prompt = voicePromptText;

            const updated = await updateCharacter(projectId, character.name, fields);
            onProjectUpdate(updated);

            if (field === 'description') setEditingDesc(false);
            if (field === 'portrait_prompt') setEditingPortraitPrompt(false);
            if (field === 'voice_prompt') setEditingVoicePrompt(false);

            const labels = { description: 'Description', portrait_prompt: 'Portrait prompt', voice_prompt: 'Voice prompt' };
            toast.success(`${labels[field]} updated!`);
        } catch (e: any) {
            toast.error(e.message || 'Failed to save');
        } finally {
            setSaving(false);
        }
    };

    const handleDesignVoice = async () => {
        setDesigningVoice(true);
        try {
            const updated = await designVoice(projectId, character.name);
            onProjectUpdate(updated);
            toast.success('Voice designed and assigned!');
        } catch (e: any) {
            toast.error(e.message || 'Voice design failed');
        } finally {
            setDesigningVoice(false);
        }
    };

    const handlePreviewVoice = async () => {
        if (!audioRef.current) return;
        if (playingVoice) {
            audioRef.current.pause();
            audioRef.current.currentTime = 0;
            setPlayingVoice(false);
            return;
        }

        setPlayingVoice(true);
        const url = `/audiobook/projects/${projectId}/characters/${encodeURIComponent(character.name)}/preview-voice`;
        try {
            const res = await fetch(url, { method: 'POST' });
            if (!res.ok) throw new Error('Voice preview failed');
            const blob = await res.blob();
            audioRef.current.src = URL.createObjectURL(blob);
            audioRef.current.play();
            audioRef.current.onended = () => setPlayingVoice(false);
        } catch (e: any) {
            toast.error(e.message || 'Voice preview failed');
            setPlayingVoice(false);
        }
    };

    // ── Editable prompt section helper ──
    const renderEditablePrompt = (
        label: string,
        icon: React.ReactNode,
        value: string,
        editing: boolean,
        setEditing: (v: boolean) => void,
        text: string,
        setText: (v: string) => void,
        field: 'description' | 'portrait_prompt' | 'voice_prompt',
        colorClass: string,
        placeholder: string,
    ) => (
        <div className="px-8 py-3">
            <div className="flex items-center justify-between mb-2">
                <h3 className="text-[9px] font-bold text-white/25 uppercase tracking-[0.15em] flex items-center gap-1.5">
                    {icon} {label}
                </h3>
                {!editing && (
                    <button
                        onClick={() => setEditing(true)}
                        className="text-[10px] text-white/25 hover:text-white/50 flex items-center gap-1 transition-colors"
                    >
                        <Edit3 size={9} /> Edit
                    </button>
                )}
            </div>

            {editing ? (
                <div className="space-y-2.5">
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        className="w-full bg-white/[0.04] border border-white/[0.08] rounded-xl px-4 py-3 text-[12px] text-white/80 placeholder:text-white/20 focus:border-violet-500/40 focus:ring-1 focus:ring-violet-500/20 focus:outline-none transition-all resize-y min-h-[80px] font-mono leading-relaxed"
                        rows={3}
                        autoFocus
                        placeholder={placeholder}
                    />
                    <div className="flex items-center justify-end gap-2">
                        <button
                            onClick={() => { setEditing(false); setText(value); }}
                            className="px-3 py-1.5 rounded-lg text-[10px] text-white/30 hover:text-white/60 transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={() => handleSaveField(field)}
                            disabled={saving}
                            className="px-4 py-1.5 rounded-lg text-[10px] font-semibold bg-violet-500/20 text-violet-300 hover:bg-violet-500/30 border border-violet-500/20 transition-all flex items-center gap-1.5 disabled:opacity-40"
                        >
                            {saving ? <Loader2 size={10} className="animate-spin" /> : <Save size={10} />}
                            Save
                        </button>
                    </div>
                </div>
            ) : (
                <div className={`bg-gradient-to-br ${colorClass} rounded-xl p-4 border`}>
                    <p className="text-[12px] text-white/50 leading-relaxed italic">
                        {value || placeholder}
                    </p>
                </div>
            )}
        </div>
    );

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-md"
                onClick={onClose}
            >
                <motion.div
                    initial={{ opacity: 0, scale: 0.92, y: 30 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.92, y: 30 }}
                    transition={{ type: 'spring', damping: 28, stiffness: 350 }}
                    className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-3xl bg-gradient-to-br from-[#0e0e1a] via-[#121220] to-[#0d0d18] border border-white/[0.08] shadow-2xl shadow-black/60"
                    onClick={(e) => e.stopPropagation()}
                >
                    <audio ref={audioRef} />

                    {/* Close button */}
                    <button
                        onClick={onClose}
                        className="absolute top-5 right-5 z-10 w-8 h-8 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 flex items-center justify-center text-white/40 hover:text-white/70 transition-all"
                    >
                        <X size={14} />
                    </button>

                    {/* Hero section */}
                    <div className="relative p-8 pb-4">
                        <div className="flex gap-6">
                            {/* Portrait */}
                            <div className="flex-shrink-0 space-y-3">
                                <div className="relative w-40 h-40 rounded-2xl overflow-hidden border-2 border-white/10 bg-gradient-to-br from-violet-500/10 to-pink-500/10 shadow-xl shadow-violet-500/5">
                                    {portraitSrc ? (
                                        <img
                                            src={portraitSrc}
                                            alt={character.name}
                                            className="w-full h-full object-cover object-top"
                                            key={selectedVariant}
                                        />
                                    ) : (
                                        <div className="w-full h-full flex items-center justify-center">
                                            <span className="text-5xl font-black text-violet-300/30">{character.name[0]}</span>
                                        </div>
                                    )}
                                    {generatingPortrait && (
                                        <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                                            <Loader2 size={24} className="animate-spin text-violet-400" />
                                        </div>
                                    )}
                                </div>

                                {/* Variant thumbnails */}
                                {variants.length > 0 && (
                                    <div className="flex gap-1.5 flex-wrap justify-center">
                                        {variants.map((_, idx) => (
                                            <button
                                                key={idx}
                                                onClick={() => handleSelectVariant(idx)}
                                                className={`w-9 h-9 rounded-lg overflow-hidden border-2 transition-all hover:scale-105 ${selectedVariant === idx
                                                    ? 'border-violet-400 shadow-md shadow-violet-500/30'
                                                    : 'border-white/10 hover:border-white/20'
                                                    }`}
                                            >
                                                <img
                                                    src={getPortraitUrl(projectId, character.name, idx)}
                                                    alt={`Variant ${idx + 1}`}
                                                    className="w-full h-full object-cover object-top"
                                                />
                                            </button>
                                        ))}
                                        <button
                                            onClick={handleGenerateVariant}
                                            disabled={generatingPortrait}
                                            className="w-9 h-9 rounded-lg border-2 border-dashed border-white/10 hover:border-violet-400/40 flex items-center justify-center text-white/20 hover:text-violet-400/60 transition-all disabled:opacity-30"
                                            title="Generate new variant"
                                        >
                                            {generatingPortrait ? (
                                                <Loader2 size={11} className="animate-spin" />
                                            ) : (
                                                <Plus size={11} />
                                            )}
                                        </button>
                                    </div>
                                )}

                                {/* Generate first portrait */}
                                {variants.length === 0 && (
                                    <button
                                        onClick={handleGenerateVariant}
                                        disabled={generatingPortrait}
                                        className="w-full px-3 py-2 rounded-xl text-[10px] font-semibold bg-gradient-to-r from-violet-500/15 to-pink-500/15 hover:from-violet-500/25 hover:to-pink-500/25 border border-violet-500/15 text-violet-300 disabled:opacity-40 flex items-center justify-center gap-1.5 transition-all"
                                    >
                                        {generatingPortrait ? (
                                            <Loader2 size={10} className="animate-spin" />
                                        ) : (
                                            <Image size={10} />
                                        )}
                                        Generate Portrait
                                    </button>
                                )}
                            </div>

                            {/* Name + Voice */}
                            <div className="flex-1 min-w-0 pt-1">
                                <h2 className="text-2xl font-bold text-white/90 tracking-tight">{character.name}</h2>

                                {/* Voice section */}
                                <div className="mt-4 space-y-2">
                                    <label className="text-[9px] font-bold text-white/25 uppercase tracking-[0.15em] flex items-center gap-1">
                                        <Mic size={9} /> Voice
                                    </label>
                                    <div className="flex items-center gap-2">
                                        <select
                                            value={voiceName}
                                            onChange={(e) => onVoiceChange(character.name, e.target.value)}
                                            className="flex-1 bg-white/[0.04] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white/80 focus:border-violet-500/40 focus:outline-none transition-all"
                                        >
                                            <option value="" className="bg-[#0f0f1a]">Default (Narrator)</option>
                                            {availableVoices.map((v) => (
                                                <option key={v} value={v} className="bg-[#0f0f1a]">{v}</option>
                                            ))}
                                        </select>
                                        <button
                                            onClick={handlePreviewVoice}
                                            disabled={!voiceName}
                                            className={`w-9 h-9 rounded-xl flex items-center justify-center border transition-all ${playingVoice
                                                ? 'bg-violet-500/20 border-violet-500/30 text-violet-300'
                                                : 'bg-white/[0.04] border-white/[0.08] text-white/40 hover:text-white/70 hover:bg-white/[0.06]'
                                                } disabled:opacity-30`}
                                            title="Preview voice"
                                        >
                                            {playingVoice ? <Pause size={13} /> : <Play size={13} />}
                                        </button>
                                    </div>

                                    {/* Design Voice button */}
                                    {character.voice_prompt && (
                                        <button
                                            onClick={handleDesignVoice}
                                            disabled={designingVoice}
                                            className="w-full mt-1 px-3 py-2.5 rounded-xl text-[10px] font-semibold bg-gradient-to-r from-cyan-500/15 to-teal-500/15 hover:from-cyan-500/25 hover:to-teal-500/25 border border-cyan-500/15 text-cyan-300 disabled:opacity-40 flex items-center justify-center gap-1.5 transition-all"
                                        >
                                            {designingVoice ? (
                                                <>
                                                    <Loader2 size={10} className="animate-spin" />
                                                    Designing Voice...
                                                </>
                                            ) : (
                                                <>
                                                    <Wand2 size={10} />
                                                    Design Voice from Prompt
                                                </>
                                            )}
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Visual Profile Grid */}
                    {profile && (
                        <div className="px-8 pb-2">
                            <h3 className="text-[9px] font-bold text-white/25 uppercase tracking-[0.15em] mb-3 flex items-center gap-1.5">
                                <User size={9} /> Visual Profile
                            </h3>
                            <div className="grid grid-cols-2 gap-3">
                                {(['face', 'skin', 'build', 'clothing'] as const).map((key) => {
                                    const val = profile[key];
                                    if (!val) return null;
                                    return (
                                        <motion.div
                                            key={key}
                                            initial={{ opacity: 0, y: 8 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: 0.05 * (['face', 'skin', 'build', 'clothing'].indexOf(key)) }}
                                            className={`rounded-xl p-3.5 bg-gradient-to-br ${PROFILE_COLORS[key]} border`}
                                        >
                                            <div className="flex items-center gap-1.5 mb-1.5">
                                                {PROFILE_ICONS[key]}
                                                <span className="text-[9px] font-bold uppercase tracking-[0.12em] opacity-70">
                                                    {key}
                                                </span>
                                            </div>
                                            <p className="text-[11px] leading-relaxed text-white/70">{val}</p>
                                        </motion.div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Description */}
                    {renderEditablePrompt(
                        'Description', <Sparkles size={9} />,
                        character.description || '', editingDesc, setEditingDesc,
                        descText, setDescText, 'description',
                        'from-white/[0.02] to-white/[0.01] border-white/[0.04]',
                        'No description yet. Run character extraction to populate.',
                    )}

                    {/* Portrait Prompt */}
                    {renderEditablePrompt(
                        'Portrait Prompt', <Image size={9} />,
                        character.portrait_prompt || '', editingPortraitPrompt, setEditingPortraitPrompt,
                        portraitPromptText, setPortraitPromptText, 'portrait_prompt',
                        'from-violet-500/[0.06] to-pink-500/[0.06] border-violet-500/[0.08]',
                        'No portrait prompt yet. Run character extraction to generate.',
                    )}

                    {/* Voice Prompt */}
                    {renderEditablePrompt(
                        'Voice Prompt', <Volume2 size={9} />,
                        character.voice_prompt || '', editingVoicePrompt, setEditingVoicePrompt,
                        voicePromptText, setVoicePromptText, 'voice_prompt',
                        'from-cyan-500/[0.06] to-teal-500/[0.06] border-cyan-500/[0.08]',
                        'No voice prompt yet. Run character extraction to generate.',
                    )}

                    {/* Bottom padding */}
                    <div className="h-4" />
                </motion.div>
            </motion.div>
        </AnimatePresence>
    );
};

export default CharacterProfileModal;
