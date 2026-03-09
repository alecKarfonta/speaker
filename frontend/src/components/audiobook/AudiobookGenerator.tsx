import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
    BookOpen, Plus, Trash2, Play, Pause, RefreshCw, Download,
    ChevronDown, ChevronRight, ChevronUp, Edit3, Check, X, Upload, Users,
    Volume2, Mic, Zap, AlertCircle, Loader2, FileText, Music,
    Sparkles, Settings2, Square, Radio, Wifi, WifiOff, Clock,
    Hash, Headphones, Layers, Image, Film, Sliders, Timer,
    Package, Palette, Maximize2,
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import Layout from '../layout/Layout';
import { useAudiobookStore } from '../../stores/audiobookStore';
import { useGenerationSocket } from '../../hooks/useGenerationSocket';
import { getSegmentAudioUrl, getSegmentVisualUrl, getChapterExportUrl, getFullExportUrl, getVideoUrl, getDownloadAllUrl, setVideoFillMode, getPortraitUrl, getVisualAssetFileUrl, getVisualCandidateUrl } from '../../services/audiobookApi';
import type { SegmentResponse, ChapterResponse, VisualParams, VisualAsset } from '../../services/audiobookApi';
import CharacterProfileModal from './CharacterProfileModal';
import VisualPlayer from './VisualPlayer';
import PresentationMode from './PresentationMode';
import QueuePanel from './QueuePanel';

// ── Extracted modules ───────────────────────────────────────────────────
import { glass, glassHover, cardStyle, inputStyle, selectStyle, btnPrimary, btnGhost } from './designTokens';
import { ANIMATION_STYLES, RESOLUTION_PRESETS, VISUAL_MODES, estimateTime, getCharacterColor, isBrowserVideo } from './utils';
import type { VisualMode } from './utils';
import { SidebarSection, StatusBadge, ProgressBar } from './SharedComponents';

const VisualSettingsPopover: React.FC<{
    segment: SegmentResponse;
    projectId: string;
    onClose: () => void;
}> = ({ segment, projectId, onClose }) => {
    const { generateVisual, visualMode, visualSettings, setVisualSettings, updateSegment, currentProject } = useAudiobookStore();

    const hasPortraits = !!(currentProject?.characters?.some(c => c.portrait_path));

    const computeFrames = useCallback((fpVal: number) => {
        if (segment.duration) {
            const raw = Math.round(segment.duration * fpVal);
            const rounded = ((raw + 7) >> 3) * 8 + 1;
            return Math.min(Math.max(rounded, 9), 449);
        }
        return visualSettings.frames;
    }, [segment.duration, visualSettings.frames]);

    const [localMode, setLocalMode] = useState<VisualMode>(visualMode as VisualMode);
    const [fps, setFps] = useState(visualSettings.fps);
    const [frames, setFrames] = useState(() => computeFrames(visualSettings.fps));
    const [resPick, setResPick] = useState(() =>
        RESOLUTION_PRESETS.findIndex(r => r.w === visualSettings.width && r.h === visualSettings.height)
    );
    const [scenePrompt, setScenePrompt] = useState(segment.scene_prompt || '');
    const [promptDirty, setPromptDirty] = useState(false);
    const [savingPrompt, setSavingPrompt] = useState(false);
    const [animationStyle, setAnimationStyle] = useState<string>(segment.animation_style || 'random');
    const [fillMode, setFillMode] = useState<'loop' | 'hold' | 'fade'>(
        (segment.video_fill_mode as 'loop' | 'hold' | 'fade') || 'hold'
    );

    // Portrait selector: auto-pick segment's character, fallback to first with portrait
    const charsWithPortraits = (currentProject?.characters || []).filter(c => c.portrait_path);
    const autoCharacter = charsWithPortraits.find(
        c => c.name.toLowerCase() === (segment.character || '').toLowerCase()
    )?.name || charsWithPortraits[0]?.name || '';
    const [refCharacter, setRefCharacter] = useState<string>(autoCharacter);

    const ref = useRef<HTMLDivElement>(null);

    const handleFillModeChange = async (mode: 'loop' | 'hold' | 'fade') => {
        setFillMode(mode);
        try { await setVideoFillMode(projectId, segment.id, mode); } catch { }
    };

    const res = RESOLUTION_PRESETS[resPick < 0 ? 1 : resPick];
    const estTime = estimateTime(localMode, frames, res.w, res.h);

    const handleFpsChange = (val: number) => {
        setFps(val);
        setFrames(computeFrames(val));
    };

    // Close on outside click
    useEffect(() => {
        const handler = (e: MouseEvent) => {
            if (ref.current && !ref.current.contains(e.target as Node)) onClose();
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [onClose]);

    const handleSavePrompt = async () => {
        if (!promptDirty) return;
        setSavingPrompt(true);
        try {
            await updateSegment(segment.id, { scene_prompt: scenePrompt });
            setPromptDirty(false);
        } catch { }
        setSavingPrompt(false);
    };

    const handleGenerate = async () => {
        // Persist fps and resolution globally (not frames — each segment computes from duration)
        setVisualSettings({ fps, width: res.w, height: res.h });
        if (promptDirty) await handleSavePrompt();
        onClose();
        const isRefMode = localMode === 'scene_image' || localMode === 'ref_video' || localMode === 'scene_video';
        const params: VisualParams = {
            mode: localMode,
            frames,
            fps,
            width: res.w,
            height: res.h,
            ...(localMode === 'image' || localMode === 'scene_image' ? { animation: animationStyle } : {}),
            ...(isRefMode && refCharacter ? { ref_character: refCharacter } : {}),
        };
        await generateVisual(segment.id, params);
    };

    return (
        <motion.div
            ref={ref}
            initial={{ opacity: 0, scale: 0.95, y: -4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -4 }}
            transition={{ duration: 0.12 }}
            className="absolute right-0 top-10 z-50 w-80 rounded-2xl border border-white/10 bg-[#0d0d1a]/95 backdrop-blur-xl shadow-2xl shadow-black/60 p-4 space-y-3.5"
        >
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Film size={13} className="text-emerald-400" />
                    <span className="text-xs font-semibold text-white/80">Visual Settings</span>
                </div>
                <button onClick={onClose} className="p-1 rounded-md text-white/25 hover:text-white/60 hover:bg-white/5 transition-colors">
                    <X size={12} />
                </button>
            </div>

            {/* Scene Prompt */}
            <div className="space-y-1.5">
                <label className="flex items-center justify-between text-[10px] font-medium text-white/40 uppercase tracking-wider">
                    <span>Scene Prompt</span>
                    {promptDirty && (
                        <button
                            onClick={handleSavePrompt}
                            disabled={savingPrompt}
                            className="flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/25 border border-emerald-500/20 transition-colors"
                        >
                            {savingPrompt ? <Loader2 size={8} className="animate-spin" /> : <Check size={8} />}
                            Save
                        </button>
                    )}
                </label>
                <textarea
                    value={scenePrompt}
                    onChange={e => { setScenePrompt(e.target.value); setPromptDirty(true); }}
                    placeholder="Describe the visual scene (auto-generated by LLM if empty)…"
                    rows={4}
                    className="w-full bg-white/[0.04] border border-white/[0.08] rounded-xl px-3 py-2 text-[11px] text-white/80 placeholder:text-white/20 focus:border-emerald-500/40 focus:ring-1 focus:ring-emerald-500/20 focus:outline-none resize-none transition-all leading-relaxed"
                />
            </div>

            {/* Mode selector — 5 pipelines */}
            <div className="space-y-1.5">
                <label className="text-[10px] font-medium text-white/40 uppercase tracking-wider">Pipeline</label>
                <div className="space-y-1">
                    {VISUAL_MODES.map(m => {
                        const disabled = m.needsRef && !hasPortraits;
                        const isActive = localMode === m.id;
                        const activeColor = m.icon === 'image' ? 'bg-sky-500/20 text-sky-300 border-sky-500/30' : 'bg-violet-500/20 text-violet-300 border-violet-500/30';
                        return (
                            <button
                                key={m.id}
                                onClick={() => !disabled && setLocalMode(m.id)}
                                disabled={disabled}
                                title={disabled ? 'Requires character portraits — run Extract Characters first' : m.desc}
                                className={`w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-left transition-all border ${isActive
                                    ? activeColor
                                    : disabled
                                        ? 'text-white/15 border-white/[0.03] cursor-not-allowed'
                                        : 'text-white/50 border-white/[0.06] hover:text-white/70 hover:bg-white/[0.04]'
                                    }`}
                            >
                                {m.icon === 'image' ? <Image size={12} /> : <Film size={12} />}
                                <div className="flex-1 min-w-0">
                                    <div className="text-[11px] font-medium leading-tight">{m.label}</div>
                                    <div className={`text-[9px] leading-tight ${isActive ? 'opacity-60' : 'opacity-40'}`}>{m.desc}</div>
                                </div>
                                {m.needsRef && (
                                    <span className={`flex-shrink-0 text-[8px] px-1.5 py-0.5 rounded-full border ${hasPortraits
                                        ? 'bg-emerald-500/10 text-emerald-400/70 border-emerald-500/15'
                                        : 'bg-red-500/10 text-red-400/50 border-red-500/15'
                                        }`}>
                                        {hasPortraits ? '✓ ref' : '⚠ no ref'}
                                    </span>
                                )}
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Portrait selector — reference modes only */}
            <AnimatePresence>
                {(localMode === 'scene_image' || localMode === 'ref_video' || localMode === 'scene_video') && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        className="overflow-hidden space-y-1.5"
                    >
                        <label className="text-[10px] font-medium text-white/40 uppercase tracking-wider block">
                            Reference Portrait
                        </label>
                        {charsWithPortraits.length > 0 ? (
                            <div className="space-y-1 max-h-32 overflow-y-auto pr-1">
                                {charsWithPortraits.map(c => (
                                    <button
                                        key={c.name}
                                        onClick={() => setRefCharacter(c.name)}
                                        className={`w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg transition-all border ${refCharacter === c.name
                                            ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/25'
                                            : 'text-white/50 border-white/[0.06] hover:text-white/70 hover:bg-white/[0.04]'
                                            }`}
                                    >
                                        <img
                                            src={`${getPortraitUrl(projectId, c.name)}?t=${Date.now()}`}
                                            alt={c.name}
                                            className="w-7 h-7 rounded-md object-cover border border-white/10 flex-shrink-0"
                                        />
                                        <span className="text-[11px] font-medium truncate">{c.name}</span>
                                        {c.name.toLowerCase() === (segment.character || '').toLowerCase() && (
                                            <span className="ml-auto text-[8px] px-1.5 py-0.5 rounded-full bg-sky-500/10 text-sky-400/70 border border-sky-500/15 flex-shrink-0">
                                                speaker
                                            </span>
                                        )}
                                    </button>
                                ))}
                            </div>
                        ) : (
                            <div className="text-[10px] text-red-400/60 px-2 py-2 rounded-lg border border-red-500/10 bg-red-500/[0.04]">
                                No character portraits available. Run <strong>Extract Characters</strong> and <strong>Generate Portraits</strong> first.
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Animation style — image output modes only */}
            <AnimatePresence>
                {(localMode === 'image' || localMode === 'scene_image') && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        className="overflow-hidden space-y-1.5"
                    >
                        <label className="text-[10px] font-medium text-white/40 uppercase tracking-wider block">
                            Animation <span className="normal-case text-white/20">(applied at export)</span>
                        </label>
                        <div className="grid grid-cols-3 gap-1">
                            {ANIMATION_STYLES.map(s => (
                                <button
                                    key={s.id}
                                    onClick={() => setAnimationStyle(s.id)}
                                    title={s.title}
                                    className={`py-1.5 px-1 rounded-lg text-[10px] font-medium transition-all border leading-tight ${animationStyle === s.id
                                        ? 'bg-sky-500/20 text-sky-300 border-sky-500/30'
                                        : 'text-white/35 border-white/[0.06] hover:text-white/60 hover:bg-white/[0.04]'
                                        }`}
                                >
                                    {s.label}
                                </button>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Video-only controls */}
            <AnimatePresence>
                {(localMode === 'video' || localMode === 'ref_video' || localMode === 'scene_video') && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        className="overflow-hidden space-y-3"
                    >
                        {/* Video fill mode (when clip is shorter than audio) */}
                        <div className="space-y-1.5">
                            <label className="text-[10px] font-medium text-white/40 uppercase tracking-wider block">
                                When clip ends early
                            </label>
                            <div className="flex rounded-xl border border-white/[0.08] overflow-hidden">
                                {([['loop', 'Loop'], ['hold', 'Hold frame'], ['fade', 'Fade out']] as const).map(([m, label]) => (
                                    <button
                                        key={m}
                                        onClick={() => handleFillModeChange(m)}
                                        className={`flex-1 py-1.5 text-[10px] font-medium transition-all ${fillMode === m
                                            ? 'bg-violet-500/20 text-violet-300'
                                            : 'text-white/35 hover:text-white/60 hover:bg-white/[0.04]'
                                            }`}
                                    >
                                        {label}
                                    </button>
                                ))}
                            </div>
                        </div>
                        {/* Frames */}
                        <div className="space-y-1.5">
                            <div className="flex items-center justify-between">
                                <label className="text-[10px] font-medium text-white/40 uppercase tracking-wider">Frames</label>
                                <div className="flex items-center gap-2">
                                    {segment.duration && (
                                        <span className="text-[9px] text-white/25 tabular-nums">({(frames / fps).toFixed(1)}s / {segment.duration.toFixed(1)}s audio)</span>
                                    )}
                                    <span className="text-[11px] text-white/60 tabular-nums font-mono">{frames}</span>
                                </div>
                            </div>
                            <input
                                type="range" min={9} max={449} step={8}
                                value={frames}
                                onChange={e => setFrames(Number(e.target.value))}
                                className="w-full h-1.5 accent-violet-500 bg-white/10 rounded-full appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between text-[9px] text-white/20">
                                <span>9</span><span>225</span><span>449</span>
                            </div>
                        </div>

                        {/* FPS */}
                        <div className="flex items-center gap-3">
                            <label className="text-[10px] font-medium text-white/40 uppercase tracking-wider w-8">FPS</label>
                            <input
                                type="range" min={8} max={30} step={1}
                                value={fps}
                                onChange={e => handleFpsChange(Number(e.target.value))}
                                className="flex-1 h-1.5 accent-violet-500 bg-white/10 rounded-full appearance-none cursor-pointer"
                            />
                            <span className="text-[11px] text-white/60 tabular-nums font-mono w-8 text-right">{fps}</span>
                        </div>

                        {/* Duration estimate from frames/fps */}
                        <div className="text-[10px] text-white/30 text-center">
                            Duration: ~{(frames / fps).toFixed(1)}s
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Resolution */}
            <div className="space-y-1.5">
                <label className="text-[10px] font-medium text-white/40 uppercase tracking-wider">Resolution</label>
                <div className="grid grid-cols-2 gap-1.5">
                    {RESOLUTION_PRESETS.map((r, i) => (
                        <button
                            key={r.label}
                            onClick={() => setResPick(i)}
                            className={`py-1.5 rounded-lg text-[10px] font-mono transition-all border ${resPick === i
                                ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                                : 'text-white/35 border-white/[0.06] hover:text-white/60 hover:bg-white/[0.04]'
                                }`}
                        >
                            {r.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Time estimate + Generate */}
            <div className="pt-1 border-t border-white/[0.05] flex items-center justify-between gap-3">
                <div className="flex items-center gap-1.5 text-white/40">
                    <Timer size={11} />
                    <span className="text-[11px]">{estTime}</span>
                </div>
                <button
                    onClick={handleGenerate}
                    className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-emerald-500/80 to-teal-600/80 text-white shadow-lg shadow-emerald-500/10 hover:from-emerald-500 hover:to-teal-600 transition-all active:scale-[0.98]"
                >
                    <Zap size={12} /> Generate
                </button>
            </div>
        </motion.div>
    );
};

// ============================================================
// Segment Row
// ============================================================
const SegmentRow: React.FC<{
    segment: SegmentResponse;
    projectId: string;
    voices: string[];
    index: number;
}> = ({ segment, projectId, voices, index }) => {
    const { generateSegment, updateSegment, splitSegment, mergeSegment, generating, playingSegmentId, setPlayingSegment } = useAudiobookStore();
    const [editing, setEditing] = useState(false);
    const [editText, setEditText] = useState(segment.text);
    const [editVoice, setEditVoice] = useState(segment.voice_name || '');
    const audioRef = useRef<HTMLAudioElement>(null);
    const isGenerating = generating.has(segment.id);
    const isPlaying = playingSegmentId === segment.id;
    const isStale = segment.status === 'pending' && !segment.has_audio;



    const speakerName = segment.character || null; // null = narrator
    const speakerLabel = segment.character || 'Narrator';
    const charColor = getCharacterColor(speakerName);
    const isNarrator = !segment.character;

    // For characters: compute inline HSL colors; for narrator: use Tailwind classes
    const borderColor = isNarrator ? undefined : `hsl(${charColor.hue}, 70%, 55%)`;
    const badgeBg = isNarrator ? undefined : `hsla(${charColor.hue}, 70%, 55%, 0.12)`;
    const badgeBorder = isNarrator ? undefined : `hsla(${charColor.hue}, 70%, 55%, 0.2)`;
    const badgeText = isNarrator ? undefined : `hsl(${charColor.hue}, 70%, 70%)`;

    const handlePlay = useCallback(() => {
        if (!segment.has_audio) return;
        if (isPlaying) {
            audioRef.current?.pause();
            setPlayingSegment(null);
        } else {
            setPlayingSegment(segment.id);
            if (audioRef.current) {
                audioRef.current.src = getSegmentAudioUrl(projectId, segment.id);
                audioRef.current.play().catch(() => { });
            }
        }
    }, [segment.id, segment.has_audio, isPlaying, projectId, setPlayingSegment]);

    const handleSaveEdit = useCallback(() => {
        const updates: { text?: string; voice_name?: string } = {};
        if (editText !== segment.text) updates.text = editText;
        if (editVoice !== segment.voice_name) updates.voice_name = editVoice;
        if (Object.keys(updates).length > 0) {
            updateSegment(segment.id, updates);
        }
        setEditing(false);
    }, [editText, editVoice, segment, updateSegment]);

    return (
        <motion.div
            layout
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.02 }}
            className={`group relative rounded-xl p-3.5 transition-all duration-200 border-l-[3px]
                ${isPlaying
                    ? 'bg-amber-500/[0.08] border border-amber-500/20 shadow-[0_0_20px_rgba(251,191,36,0.06)]'
                    : isGenerating
                        ? 'bg-sky-500/[0.05] border border-sky-500/15'
                        : `${isNarrator ? charColor.bg : ''} border border-transparent hover:bg-white/[0.04] hover:border-white/[0.06]`
                }`}
            style={{
                borderLeftColor: isPlaying ? undefined : (borderColor || (isNarrator ? 'rgba(251, 191, 36, 0.4)' : undefined)),
            }}
        >
            <audio ref={audioRef} onEnded={() => setPlayingSegment(null)} onError={() => setPlayingSegment(null)} />

            <div className="flex items-start gap-3">
                {/* Index + play */}
                <div className="flex flex-col items-center gap-1 flex-shrink-0 pt-0.5">
                    <span className="text-[10px] text-white/20 font-mono tabular-nums">{String(index + 1).padStart(2, '0')}</span>
                    <button
                        onClick={handlePlay}
                        disabled={!segment.has_audio}
                        className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all
                            ${segment.has_audio
                                ? isPlaying
                                    ? 'bg-amber-500 text-white shadow-lg shadow-amber-500/30 scale-105'
                                    : 'bg-white/[0.06] text-white/50 hover:bg-amber-500/20 hover:text-amber-400'
                                : 'bg-white/[0.02] text-white/10 cursor-not-allowed'
                            }`}
                    >
                        {isPlaying ? <Pause size={12} /> : <Play size={12} className="ml-0.5" />}
                    </button>
                </div>



                {/* Content */}
                <div className="flex-1 min-w-0">
                    {editing ? (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="space-y-2.5"
                        >
                            <textarea
                                value={editText}
                                onChange={(e) => setEditText(e.target.value)}
                                className={`${inputStyle} text-sm resize-y min-h-[80px] font-mono`}
                                rows={4}
                                autoFocus
                            />
                            <div className="flex items-center gap-2">
                                <select
                                    value={editVoice}
                                    onChange={(e) => setEditVoice(e.target.value)}
                                    className={selectStyle}
                                >
                                    {voices.map((v) => (
                                        <option key={v} value={v} className="bg-[#0f0f1a]">{v}</option>
                                    ))}
                                </select>
                                <div className="flex-1" />
                                <button onClick={() => setEditing(false)} className="px-3 py-1.5 rounded-lg text-xs text-white/40 hover:text-white/70 hover:bg-white/5 transition-colors">Cancel</button>
                                <button onClick={handleSaveEdit} className="px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 border border-emerald-500/20 transition-colors flex items-center gap-1">
                                    <Check size={12} /> Save
                                </button>
                            </div>
                        </motion.div>
                    ) : (
                        <>
                            {/* Speaker tag */}
                            <div className="flex items-center gap-2 mb-1">
                                <span
                                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-semibold border
                                        ${isNarrator
                                            ? 'bg-amber-500/10 text-amber-400/80 border-amber-500/15'
                                            : ''
                                        }`}
                                    style={!isNarrator ? {
                                        backgroundColor: badgeBg,
                                        borderColor: badgeBorder,
                                        color: badgeText,
                                    } : undefined}
                                >
                                    {isNarrator ? <Headphones size={9} /> : <Users size={9} />}
                                    {speakerLabel}
                                </span>
                                {segment.voice_name && (
                                    <span className="text-[10px] text-white/25 flex items-center gap-0.5">
                                        <Mic size={8} /> {segment.voice_name}
                                    </span>
                                )}
                            </div>
                            <p className="text-[13px] text-white/75 leading-[1.7]">
                                {segment.text}
                            </p>
                            <div className="flex items-center gap-2.5 mt-2 flex-wrap">
                                <StatusBadge status={segment.status} />
                                {isStale && (
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-500/10 text-amber-400 border border-amber-500/15">
                                        <AlertCircle size={8} /> needs re-gen
                                    </span>
                                )}
                                {segment.emotion && (
                                    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[10px] bg-pink-500/10 text-pink-400/80 border border-pink-500/10">
                                        {segment.emotion}
                                    </span>
                                )}
                                {segment.duration && (
                                    <span className="text-[11px] text-white/25 font-mono tabular-nums">{segment.duration.toFixed(1)}s</span>
                                )}

                                {segment.error_message && (
                                    <span className="text-[10px] text-red-400/70 truncate max-w-[200px]" title={segment.error_message}>
                                        ⚠ {segment.error_message}
                                    </span>
                                )}
                            </div>
                        </>
                    )}
                </div>

                {/* Actions toolbar */}
                {!editing && (
                    <div className="relative flex items-center gap-0.5 flex-shrink-0">
                        {[
                            { icon: <Edit3 size={13} />, title: 'Edit text', onClick: () => { setEditText(segment.text); setEditVoice(segment.voice_name || ''); setEditing(true); }, color: 'hover:text-white/80' },
                            { icon: <Sparkles size={13} />, title: 'Split segment', onClick: () => splitSegment(segment.id), color: 'hover:text-sky-400' },
                            { icon: <Layers size={13} />, title: 'Merge with next', onClick: () => mergeSegment(segment.id), color: 'hover:text-violet-400' },
                            { icon: <RefreshCw size={13} />, title: 'Re-generate audio', onClick: () => generateSegment(segment.id), color: 'hover:text-amber-400', disabled: isGenerating },
                        ].map((btn, i) => (
                            <button
                                key={i}
                                onClick={btn.onClick}
                                disabled={btn.disabled}
                                title={btn.title}
                                className={`p-1.5 rounded-lg text-white/25 ${btn.color} hover:bg-white/[0.06] transition-all disabled:opacity-30`}
                            >
                                {btn.icon}
                            </button>
                        ))}

                    </div>
                )}
            </div>
        </motion.div>
    );
};


// ============================================================
// Chapter Player (continuous playback + timeline)
// ============================================================
const ChapterPlayer: React.FC<{
    chapter: ChapterResponse;
    projectId: string;
    visuals?: VisualAsset[];
    onRequestPresentation?: () => void;
}> = ({ chapter, projectId, visuals = [], onRequestPresentation }) => {
    const { setPlayingSegment, playingSegmentId } = useAudiobookStore();
    const audioRef = useRef<HTMLAudioElement>(null);
    const [playing, setPlaying] = useState(false);
    const [currentIdx, setCurrentIdx] = useState(0);
    const [gap, setGap] = useState(300); // ms between segments
    const gapTimerRef = useRef<any>(null);

    const playableSegments = chapter.segments.filter((s: SegmentResponse) => s.has_audio);
    const totalDuration = playableSegments.reduce((sum: number, s: SegmentResponse) => sum + (s.duration || 0), 0);

    const playSegmentAt = useCallback((idx: number) => {
        if (idx < 0 || idx >= playableSegments.length) {
            setPlaying(false);
            setPlayingSegment(null);
            return;
        }
        setCurrentIdx(idx);
        const seg = playableSegments[idx];
        setPlayingSegment(seg.id);
        if (audioRef.current) {
            audioRef.current.src = getSegmentAudioUrl(projectId, seg.id);
            audioRef.current.play().catch(() => { });
        }
    }, [playableSegments, projectId, setPlayingSegment]);

    const handlePlayPause = useCallback(() => {
        if (playing) {
            audioRef.current?.pause();
            setPlaying(false);
            setPlayingSegment(null);
        } else {
            setPlaying(true);
            playSegmentAt(currentIdx);
        }
    }, [playing, currentIdx, playSegmentAt, setPlayingSegment]);

    const handleEnded = useCallback(() => {
        // Gap then advance
        gapTimerRef.current = setTimeout(() => {
            const nextIdx = currentIdx + 1;
            if (nextIdx < playableSegments.length) {
                playSegmentAt(nextIdx);
            } else {
                setPlaying(false);
                setPlayingSegment(null);
                setCurrentIdx(0);
            }
        }, gap);
    }, [currentIdx, playableSegments.length, gap, playSegmentAt, setPlayingSegment]);

    // Keyboard shortcuts
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
            if (e.code === 'Space' && playableSegments.length > 0) { e.preventDefault(); handlePlayPause(); }
            if (e.code === 'ArrowRight' && playing) { e.preventDefault(); playSegmentAt(Math.min(currentIdx + 1, playableSegments.length - 1)); }
            if (e.code === 'ArrowLeft' && playing) { e.preventDefault(); playSegmentAt(Math.max(currentIdx - 1, 0)); }
        };
        window.addEventListener('keydown', handler);
        return () => { window.removeEventListener('keydown', handler); clearTimeout(gapTimerRef.current); };
    }, [handlePlayPause, playing, currentIdx, playableSegments.length, playSegmentAt]);

    if (playableSegments.length === 0) return null;

    return (
        <div className="px-5 py-3 border-t border-white/[0.04]">
            <audio ref={audioRef} onEnded={handleEnded} onError={() => { setPlaying(false); setPlayingSegment(null); }} />

            {/* Visual player */}
            {visuals.length > 0 && (
                <div className="mb-3">
                    <VisualPlayer
                        currentSegment={playableSegments[currentIdx] || null}
                        visuals={visuals}
                        projectId={projectId}
                        isPlaying={playing}
                        duration={playableSegments[currentIdx]?.duration || 4}
                        onRequestPresentation={onRequestPresentation}
                    />
                </div>
            )}

            <div className="flex items-center gap-3">
                {/* Play/pause */}
                <button
                    onClick={handlePlayPause}
                    className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all flex-shrink-0
                        ${playing
                            ? 'bg-amber-500 text-white shadow-md shadow-amber-500/30'
                            : 'bg-white/[0.06] text-white/40 hover:bg-amber-500/20 hover:text-amber-400'
                        }`}
                >
                    {playing ? <Pause size={13} /> : <Play size={13} className="ml-0.5" />}
                </button>

                {/* Timeline scrubber */}
                <div className="flex-1 flex items-center h-6 rounded-lg bg-white/[0.02] overflow-hidden cursor-pointer">
                    {playableSegments.map((seg: SegmentResponse, i: number) => {
                        const width = totalDuration > 0 ? ((seg.duration || 1) / totalDuration) * 100 : 100 / playableSegments.length;
                        const isCurrent = i === currentIdx && playing;
                        return (
                            <div
                                key={seg.id}
                                onClick={() => { setPlaying(true); playSegmentAt(i); }}
                                title={`${seg.character || 'Narrator'}: ${seg.text.slice(0, 60)}...`}
                                className={`h-full transition-all duration-200 border-r border-black/20 last:border-r-0 relative group/seg
                                    ${isCurrent
                                        ? 'bg-gradient-to-r from-amber-500/40 to-orange-500/40'
                                        : i < currentIdx && playing
                                            ? 'bg-amber-500/15'
                                            : 'bg-white/[0.04] hover:bg-white/[0.08]'
                                    }`}
                                style={{ width: `${width}%` }}
                            >
                                {isCurrent && (
                                    <motion.div
                                        className="absolute inset-0 bg-amber-500/20"
                                        animate={{ opacity: [0.3, 0.6, 0.3] }}
                                        transition={{ duration: 1.5, repeat: Infinity }}
                                    />
                                )}
                            </div>
                        );
                    })}
                </div>

                {/* Time display */}
                <span className="text-[10px] text-white/25 tabular-nums flex-shrink-0 w-12 text-right">
                    {totalDuration > 0 ? `${Math.floor(totalDuration / 60)}:${String(Math.floor(totalDuration % 60)).padStart(2, '0')}` : '--:--'}
                </span>

                {/* Gap control */}
                <div className="flex items-center gap-1.5 flex-shrink-0">
                    <span className="text-[9px] text-white/15 uppercase">Gap</span>
                    <input
                        type="range"
                        min={0}
                        max={2000}
                        step={100}
                        value={gap}
                        onChange={(e) => setGap(Number(e.target.value))}
                        className="w-14 h-1 accent-amber-500 bg-white/10 rounded-full appearance-none cursor-pointer"
                        title={`${gap}ms gap between segments`}
                    />
                    <span className="text-[9px] text-white/15 tabular-nums w-7">{gap}ms</span>
                </div>
            </div>
        </div>
    );
};

// ============================================================
// Chapter Section
// ============================================================
const ChapterSection: React.FC<{
    chapter: ChapterResponse;
    projectId: string;
    voices: string[];
    onGenerateChapter?: (idx: number) => void;
    generatingSegmentId?: string | null;
}> = ({ chapter, projectId, voices, onGenerateChapter, generatingSegmentId }) => {
    const { generateChapter, generateAllVisuals, currentProject } = useAudiobookStore();
    const [expanded, setExpanded] = useState(true);
    const [showPresentation, setShowPresentation] = useState(false);
    const donePercent = Math.round(chapter.progress * 100);

    const handleGenerate = () => {
        if (onGenerateChapter) {
            onGenerateChapter(chapter.index);
        } else {
            generateChapter(chapter.index);
        }
    };

    return (
        <div className={`${cardStyle} overflow-hidden transition-colors`}>
            {/* Chapter header */}
            <button
                onClick={() => setExpanded(!expanded)}
                className={`w-full flex items-center gap-4 px-5 py-4 transition-colors text-left ${glassHover}`}
            >
                <motion.div
                    animate={{ rotate: expanded ? 90 : 0 }}
                    transition={{ duration: 0.15, ease: 'easeOut' }}
                    className="flex-shrink-0"
                >
                    <ChevronRight size={14} className="text-white/30" />
                </motion.div>

                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3">
                        <h3 className="text-sm font-semibold text-white/90 truncate">
                            {chapter.title || `Chapter ${chapter.index + 1}`}
                        </h3>
                        <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full
                            ${donePercent === 100
                                ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20'
                                : donePercent > 0
                                    ? 'bg-amber-500/10 text-amber-400/70 border border-amber-500/15'
                                    : 'bg-white/[0.04] text-white/30 border border-white/[0.06]'
                            }`}>
                            {donePercent}%
                        </span>
                    </div>
                    <div className="flex items-center gap-3 mt-1.5">
                        <span className="text-[11px] text-white/30 tabular-nums">
                            {chapter.done_segments}/{chapter.total_segments} segments
                        </span>
                        <div className="w-28">
                            <ProgressBar progress={chapter.progress} />
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
                    <button
                        onClick={handleGenerate}
                        className="px-3 py-1.5 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 text-[11px] font-medium text-amber-400/80 hover:text-amber-400 border border-amber-500/10 hover:border-amber-500/20 transition-all flex items-center gap-1.5"
                        title="Generate TTS audio for this chapter"
                    >
                        <Zap size={11} /> Generate Audio
                    </button>
                    <button
                        onClick={() => generateAllVisuals()}
                        className="px-3 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-[11px] font-medium text-emerald-400/80 hover:text-emerald-400 border border-emerald-500/10 hover:border-emerald-500/20 transition-all flex items-center gap-1.5"
                        title="Generate visuals for all segments in this chapter"
                    >
                        <Image size={11} /> Generate Visuals
                    </button>
                    <a
                        href={getChapterExportUrl(projectId, chapter.index)}
                        className="px-3 py-1.5 rounded-lg bg-white/[0.03] hover:bg-white/[0.06] text-[11px] font-medium text-white/40 hover:text-white/70 border border-white/[0.06] transition-all flex items-center gap-1.5"
                        title="Export this chapter's audio"
                    >
                        <Download size={11} /> Export
                    </a>
                </div>
            </button>

            {/* Timeline player with visuals */}
            <ChapterPlayer
                chapter={chapter}
                projectId={projectId}
                visuals={currentProject?.visuals || []}
                onRequestPresentation={() => setShowPresentation(true)}
            />

            {/* Presentation mode */}
            <AnimatePresence>
                {showPresentation && (
                    <PresentationMode
                        chapter={chapter}
                        projectId={projectId}
                        visuals={currentProject?.visuals || []}
                        onClose={() => setShowPresentation(false)}
                    />
                )}
            </AnimatePresence>

            {/* Segments */}
            <AnimatePresence>
                {expanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2, ease: 'easeInOut' }}
                        className="overflow-hidden"
                    >
                        <div className="px-5 pb-4 pt-1 space-y-1.5">
                            <SegmentVisualGrid
                                segments={chapter.segments}
                                projectId={projectId}
                                voices={voices}
                                generatingSegmentId={generatingSegmentId || undefined}
                            />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

// ============================================================
// New Project Modal
// ============================================================
const NewProjectModal: React.FC<{
    open: boolean;
    onClose: () => void;
}> = ({ open, onClose }) => {
    const { createProject, importProject, loading } = useAudiobookStore();
    const [name, setName] = useState('');
    const [text, setText] = useState('');
    const [pattern, setPattern] = useState('auto');
    const [dragOver, setDragOver] = useState(false);
    const [importFile, setImportFile] = useState<File | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const bookFileInputRef = useRef<HTMLInputElement>(null);

    const isBookFile = (filename: string) => {
        const ext = filename.split('.').pop()?.toLowerCase();
        return ext === 'epub' || ext === 'pdf';
    };

    const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            const content = ev.target?.result as string;
            setText(content);
            if (!name) setName(file.name.replace(/\.[^/.]+$/, ''));
            toast.success(`Loaded ${file.name}`);
        };
        reader.readAsText(file);
    }, [name]);

    const handleBookFileSelect = useCallback((file: File) => {
        if (isBookFile(file.name)) {
            setImportFile(file);
            if (!name) setName(file.name.replace(/\.[^/.]+$/, ''));
            toast.success(`Selected ${file.name}`);
        } else {
            toast.error('Use .epub or .pdf');
        }
    }, [name]);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) handleBookFileSelect(file);
    }, [handleBookFileSelect]);

    const handleCreate = useCallback(async () => {
        if (importFile) {
            await importProject(importFile);
            toast.success('Book imported!');
            onClose();
            setName(''); setText(''); setImportFile(null);
            return;
        }
        if (!name.trim() || !text.trim()) { toast.error('Name and text are required'); return; }
        await createProject(name.trim(), text.trim(), pattern);
        toast.success('Project created!');
        onClose();
        setName(''); setText('');
    }, [name, text, pattern, importFile, createProject, importProject, onClose]);

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 bg-black/70 backdrop-blur-md"
                onClick={onClose}
            />
            <motion.div
                initial={{ opacity: 0, scale: 0.95, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 10 }}
                transition={{ type: 'spring', damping: 25, stiffness: 400 }}
                className={`relative w-full max-w-2xl ${cardStyle} p-6 shadow-2xl shadow-black/40 max-h-[85vh] overflow-y-auto`}
            >
                {/* Close button */}
                <button onClick={onClose} className="absolute top-4 right-4 p-1.5 rounded-lg text-white/30 hover:text-white/60 hover:bg-white/5 transition-colors">
                    <X size={16} />
                </button>

                <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500/20 to-orange-500/20 flex items-center justify-center border border-amber-500/10">
                        <Plus size={18} className="text-amber-400" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-white/90">New Audiobook</h2>
                        <p className="text-xs text-white/30">Import a book or paste text to get started</p>
                    </div>
                </div>

                <div className="space-y-5">
                    {/* Drop zone */}
                    <div
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                        onClick={() => bookFileInputRef.current?.click()}
                        className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200
                            ${dragOver
                                ? 'border-amber-500/60 bg-amber-500/[0.08] scale-[1.01]'
                                : importFile
                                    ? 'border-emerald-500/30 bg-emerald-500/[0.04]'
                                    : 'border-white/[0.08] hover:border-white/[0.15] bg-white/[0.01] hover:bg-white/[0.03]'
                            }`}
                    >
                        {importFile ? (
                            <div className="flex items-center justify-center gap-4">
                                <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
                                    <BookOpen size={22} className="text-emerald-400" />
                                </div>
                                <div className="text-left">
                                    <p className="text-sm font-semibold text-emerald-400">{importFile.name}</p>
                                    <p className="text-xs text-white/35 mt-0.5">{(importFile.size / 1024).toFixed(0)} KB — Ready to import</p>
                                </div>
                                <button
                                    onClick={(e) => { e.stopPropagation(); setImportFile(null); }}
                                    className="p-1.5 rounded-lg text-white/25 hover:text-white/50 hover:bg-white/5"
                                >
                                    <X size={14} />
                                </button>
                            </div>
                        ) : (
                            <>
                                <div className="w-14 h-14 rounded-2xl bg-white/[0.04] flex items-center justify-center mx-auto mb-3 border border-white/[0.06]">
                                    <Upload size={24} className="text-white/20" />
                                </div>
                                <p className="text-sm text-white/50">
                                    Drop <span className="text-amber-400 font-semibold">.epub</span> or{' '}
                                    <span className="text-amber-400 font-semibold">.pdf</span> here
                                </p>
                                <p className="text-[11px] text-white/20 mt-1">or click to browse</p>
                            </>
                        )}
                        <input ref={bookFileInputRef} type="file" accept=".epub,.pdf" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleBookFileSelect(f); }} className="hidden" />
                    </div>

                    {!importFile && (
                        <>
                            {/* Divider */}
                            <div className="flex items-center gap-4">
                                <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/[0.08] to-transparent" />
                                <span className="text-[10px] text-white/20 uppercase tracking-[0.15em] font-medium">or paste text</span>
                                <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/[0.08] to-transparent" />
                            </div>

                            <div>
                                <label className="block text-xs font-medium text-white/40 mb-1.5">Project Name</label>
                                <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="My Audiobook" className={inputStyle} />
                            </div>

                            <div>
                                <label className="block text-xs font-medium text-white/40 mb-1.5">Chapter Detection</label>
                                <select value={pattern} onChange={(e) => setPattern(e.target.value)} className={`${selectStyle} w-full`}>
                                    <option value="auto" className="bg-[#0f0f1a]">Auto-detect</option>
                                    <option value="chapter_number" className="bg-[#0f0f1a]">Chapter 1, Chapter 2...</option>
                                    <option value="chapter_word" className="bg-[#0f0f1a]">Chapter titles</option>
                                    <option value="part" className="bg-[#0f0f1a]">Part 1, Part 2...</option>
                                    <option value="separator" className="bg-[#0f0f1a]">--- separators</option>
                                    <option value="numbered_dot" className="bg-[#0f0f1a]">1. Section, 2. Section...</option>
                                </select>
                            </div>

                            <div>
                                <div className="flex items-center justify-between mb-1.5">
                                    <label className="text-xs font-medium text-white/40">Book Text</label>
                                    <div className="flex items-center gap-3">
                                        <span className="text-[10px] text-white/20 font-mono tabular-nums">{text.length.toLocaleString()} chars</span>
                                        <button onClick={() => fileInputRef.current?.click()} className="flex items-center gap-1 text-[11px] text-amber-400/70 hover:text-amber-400 transition-colors">
                                            <Upload size={10} /> .txt
                                        </button>
                                        <input ref={fileInputRef} type="file" accept=".txt,.text,.md" onChange={handleFileUpload} className="hidden" />
                                    </div>
                                </div>
                                <textarea
                                    value={text}
                                    onChange={(e) => setText(e.target.value)}
                                    placeholder="Paste your book text here..."
                                    className={`${inputStyle} text-[13px] resize-y font-mono leading-relaxed`}
                                    rows={10}
                                />
                            </div>
                        </>
                    )}
                </div>

                <div className="flex items-center justify-end gap-3 mt-6 pt-4 border-t border-white/[0.06]">
                    <button onClick={onClose} className="px-4 py-2 rounded-xl text-sm text-white/40 hover:text-white/70 transition-colors">
                        Cancel
                    </button>
                    <button
                        onClick={handleCreate}
                        disabled={loading || (importFile ? false : (!name.trim() || !text.trim()))}
                        className={`${btnPrimary} flex items-center gap-2`}
                    >
                        {loading ? <Loader2 size={15} className="animate-spin" /> : <Sparkles size={15} />}
                        {importFile ? 'Import Book' : 'Create Project'}
                    </button>
                </div>
            </motion.div>
        </div>
    );
};

// ============================================================
// Character Voice Panel
// ============================================================
const CharacterVoicePanel: React.FC = () => {
    const { currentProject, availableVoices, updateCharacterMap, analyzeCharacters, analyzing, setCurrentProject } = useAudiobookStore();
    const [selectedChar, setSelectedChar] = useState<any>(null);
    const [extracting, setExtracting] = useState(false);
    const [extractStep, setExtractStep] = useState('');
    const [generatingPortraits, setGeneratingPortraits] = useState(false);
    const [portraitProgress, setPortraitProgress] = useState({ done: 0, total: 0, currentName: '' });
    const [generatingNarrator, setGeneratingNarrator] = useState(false);
    const [designingNarrator, setDesigningNarrator] = useState(false);
    const [narratorPrompt, setNarratorPrompt] = useState('');
    const [elapsed, setElapsed] = useState(0);
    const elapsedRef = useRef<any>(null);

    if (!currentProject) return null;

    const { detected_characters, character_voice_map, character_descriptions, narrator_voice } = currentProject;

    // Sync narrator prompt from project data
    useEffect(() => {
        setNarratorPrompt(currentProject.narrator_voice_prompt || '');
    }, [currentProject.narrator_voice_prompt]);

    const startTimer = () => {
        setElapsed(0);
        elapsedRef.current = setInterval(() => setElapsed(t => t + 1), 1000);
    };
    const stopTimer = () => { clearInterval(elapsedRef.current); elapsedRef.current = null; };

    const handleVoiceChange = (character: string, voice: string) => {
        const newMap = { ...character_voice_map, [character]: voice };
        updateCharacterMap(newMap, narrator_voice);
    };

    const handleNarratorChange = (voice: string) => {
        updateCharacterMap(character_voice_map, voice);
    };

    const handleExtractCharacters = async () => {
        setExtracting(true);
        setExtractStep('Analyzing book text for characters…');
        startTimer();
        try {
            const { extractCharacters } = await import('../../services/audiobookApi');
            const updated = await extractCharacters(currentProject.id);
            setCurrentProject(updated);
            toast.success(`Extracted ${updated.characters?.length || 0} characters`);
        } catch (e: any) {
            toast.error(e.message || 'Character extraction failed');
        } finally {
            setExtracting(false);
            setExtractStep('');
            stopTimer();
        }
    };

    const handleGeneratePortraits = async () => {
        const chars = currentProject.characters || [];
        const needPortrait = chars.filter(c => !c.portrait_path);
        setGeneratingPortraits(true);
        setPortraitProgress({ done: 0, total: needPortrait.length, currentName: needPortrait[0]?.name || '' });
        startTimer();

        // Start portrait generation (backend processes synchronously)
        const portraitPromise = (async () => {
            const { generatePortraits } = await import('../../services/audiobookApi');
            return await generatePortraits(currentProject.id);
        })();

        // Poll for progress while generating
        const pollInterval = setInterval(async () => {
            try {
                const { getProject } = await import('../../services/audiobookApi');
                const snap = await getProject(currentProject.id);
                const doneCount = (snap.characters || []).filter(c => c.portrait_path).length;
                const prevDone = (currentProject.characters || []).filter(c => c.portrait_path).length;
                const newDone = doneCount - prevDone;
                if (newDone > 0) {
                    const nextChar = needPortrait[newDone] || needPortrait[needPortrait.length - 1];
                    setPortraitProgress({ done: newDone, total: needPortrait.length, currentName: nextChar?.name || '' });
                }
            } catch { /* ignore polling errors */ }
        }, 3000);

        try {
            const updated = await portraitPromise;
            setCurrentProject(updated);
            toast.success('All portraits generated!');
        } catch (e: any) {
            toast.error(e.message || 'Portrait generation failed');
        } finally {
            clearInterval(pollInterval);
            setGeneratingPortraits(false);
            setPortraitProgress({ done: 0, total: 0, currentName: '' });
            stopTimer();
        }
    };


    const handleProjectUpdate = (updated: any) => {
        setCurrentProject(updated);
        // Update selected character if modal is open
        if (selectedChar && updated.characters) {
            const updatedChar = updated.characters.find((c: any) => c.name === selectedChar.name);
            if (updatedChar) setSelectedChar(updatedChar);
        }
    };

    const characters = currentProject.characters && currentProject.characters.length > 0
        ? currentProject.characters
        : detected_characters.map((name: string) => ({ name, description: character_descriptions?.[name] || '' }));

    return (
        <div className="space-y-4">
            {/* Action buttons */}
            <div className="space-y-2">
                <button
                    onClick={() => analyzeCharacters().then(() => toast.success('AI analysis complete!'))}
                    disabled={analyzing}
                    className="w-full px-3 py-2.5 rounded-xl text-[11px] font-semibold bg-gradient-to-r from-violet-500/15 to-purple-500/15 hover:from-violet-500/25 hover:to-purple-500/25 border border-violet-500/15 text-violet-300 hover:text-violet-200 transition-all disabled:opacity-40 flex items-center justify-center gap-2"
                    title="Detect speaking characters and assign voices automatically"
                >
                    {analyzing ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                    {analyzing ? 'Analyzing…' : '✨ Auto-Assign Voices'}
                </button>

                {/* Extract Characters */}
                {extracting ? (
                    <div className="rounded-xl border border-violet-500/15 bg-violet-500/[0.05] p-2.5 space-y-2">
                        <div className="flex items-center gap-2">
                            <Loader2 size={11} className="animate-spin text-violet-400 flex-shrink-0" />
                            <span className="text-[10px] font-medium text-violet-300 flex-1">{extractStep}</span>
                            <span className="text-[9px] text-white/20 tabular-nums font-mono">{elapsed}s</span>
                        </div>
                        <div className="w-full h-1 bg-white/[0.04] rounded-full overflow-hidden">
                            <div className="h-full bg-violet-500/50 rounded-full animate-pulse" style={{ width: '100%' }} />
                        </div>
                    </div>
                ) : (
                    <button
                        onClick={handleExtractCharacters}
                        className="w-full px-2 py-2 rounded-xl text-[10px] font-semibold bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-white/50 hover:text-white/70 transition-all flex items-center justify-center gap-1.5"
                        title="Analyze book text to identify all speaking characters and build profiles"
                    >
                        <Users size={10} /> Extract Characters
                    </button>
                )}

                {/* Generate All Portraits */}
                {generatingPortraits ? (
                    <div className="rounded-xl border border-emerald-500/15 bg-emerald-500/[0.05] p-2.5 space-y-2">
                        <div className="flex items-center gap-2">
                            <Loader2 size={11} className="animate-spin text-emerald-400 flex-shrink-0" />
                            <span className="text-[10px] font-medium text-emerald-300 flex-1 truncate">
                                Generating portrait{portraitProgress.currentName ? `: ${portraitProgress.currentName}` : '…'}
                            </span>
                            <span className="text-[9px] text-white/20 tabular-nums font-mono">{elapsed}s</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="flex-1 h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full"
                                    animate={{ width: portraitProgress.total > 0 ? `${(portraitProgress.done / portraitProgress.total) * 100}%` : '10%' }}
                                    transition={{ duration: 0.4 }}
                                />
                            </div>
                            <span className="text-[9px] text-emerald-400/60 tabular-nums">{portraitProgress.done}/{portraitProgress.total}</span>
                        </div>
                    </div>
                ) : (
                    <button
                        onClick={handleGeneratePortraits}
                        disabled={!currentProject.characters?.length}
                        className="w-full px-2 py-2 rounded-xl text-[10px] font-semibold bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-white/50 hover:text-white/70 transition-all disabled:opacity-40 flex items-center justify-center gap-1.5"
                        title="Generate visual portraits for each character using ComfyUI"
                    >
                        <Image size={10} /> Generate All Portraits
                    </button>
                )}
            </div>

            {/* Narrator */}
            <div>
                <label className="text-[10px] font-semibold text-white/30 uppercase tracking-[0.12em] mb-1.5 block flex items-center gap-1.5">
                    <Headphones size={10} /> Narrator
                </label>
                <select
                    value={narrator_voice}
                    onChange={(e) => handleNarratorChange(e.target.value)}
                    className={`${selectStyle} w-full text-xs`}
                >
                    <option value="" className="bg-[#0f0f1a]">Select voice...</option>
                    {availableVoices.map((v: string) => (
                        <option key={v} value={v} className="bg-[#0f0f1a]">{v}</option>
                    ))}
                </select>

                {/* Narrator voice prompt */}
                <div className="mt-2 space-y-1.5">
                    <textarea
                        value={narratorPrompt}
                        onChange={(e) => setNarratorPrompt(e.target.value)}
                        onBlur={async () => {
                            if (narratorPrompt !== (currentProject.narrator_voice_prompt || '')) {
                                try {
                                    const { updateNarratorVoicePrompt } = await import('../../services/audiobookApi');
                                    const updated = await updateNarratorVoicePrompt(currentProject.id, narratorPrompt);
                                    setCurrentProject(updated);
                                } catch (e: any) {
                                    toast.error(e.message || 'Failed to save narrator prompt');
                                }
                            }
                        }}
                        placeholder="Describe the ideal narrator voice... (auto-generated or edit manually)"
                        className={`${inputStyle} text-[11px] resize-y font-mono leading-relaxed min-h-[60px]`}
                        rows={3}
                    />
                    <div className="flex gap-1.5">
                        <button
                            onClick={async () => {
                                setGeneratingNarrator(true);
                                try {
                                    const { generateNarratorVoice } = await import('../../services/audiobookApi');
                                    const updated = await generateNarratorVoice(currentProject.id);
                                    setCurrentProject(updated);
                                    toast.success('Narrator voice generated!');
                                } catch (e: any) {
                                    toast.error(e.message || 'Narrator voice generation failed');
                                } finally {
                                    setGeneratingNarrator(false);
                                }
                            }}
                            disabled={generatingNarrator}
                            className="flex-1 px-2 py-1.5 rounded-lg text-[10px] font-semibold bg-gradient-to-r from-amber-500/10 to-orange-500/10 hover:from-amber-500/20 hover:to-orange-500/20 border border-amber-500/10 text-amber-400/80 hover:text-amber-400 transition-all disabled:opacity-40 flex items-center justify-center gap-1"
                        >
                            {generatingNarrator ? <Loader2 size={10} className="animate-spin" /> : <Sparkles size={10} />}
                            Generate
                        </button>
                        <button
                            onClick={async () => {
                                setDesigningNarrator(true);
                                try {
                                    const { designNarratorVoice } = await import('../../services/audiobookApi');
                                    const updated = await designNarratorVoice(currentProject.id);
                                    setCurrentProject(updated);
                                    toast.success('Narrator voice redesigned!');
                                } catch (e: any) {
                                    toast.error(e.message || 'Voice redesign failed');
                                } finally {
                                    setDesigningNarrator(false);
                                }
                            }}
                            disabled={designingNarrator || !narratorPrompt.trim()}
                            className="flex-1 px-2 py-1.5 rounded-lg text-[10px] font-semibold bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-white/50 hover:text-white/70 transition-all disabled:opacity-40 flex items-center justify-center gap-1"
                        >
                            {designingNarrator ? <Loader2 size={10} className="animate-spin" /> : <RefreshCw size={10} />}
                            Redesign
                        </button>
                    </div>
                </div>
            </div>

            {/* Characters */}
            {characters.length > 0 && (
                <div>
                    <label className="text-[10px] font-semibold text-white/30 uppercase tracking-[0.12em] mb-2 block flex items-center gap-1.5">
                        <Users size={10} /> Characters ({characters.length})
                    </label>
                    <div className="space-y-2">
                        {characters.map((char: any) => {
                            const name = char.name;
                            const portraitUrl = char.portrait_path
                                ? `/audiobook/projects/${currentProject.id}/characters/${encodeURIComponent(name)}/portrait`
                                : null;

                            return (
                                <button
                                    key={name}
                                    onClick={() => setSelectedChar(char)}
                                    className="w-full text-left bg-white/[0.02] border border-white/5 rounded-xl p-2.5 hover:bg-white/[0.05] hover:border-violet-500/15 transition-all group cursor-pointer"
                                >
                                    <div className="flex items-center gap-2.5">
                                        {portraitUrl ? (
                                            <img src={portraitUrl} alt={name} className="w-8 h-8 rounded-lg object-cover object-top border border-white/10 flex-shrink-0" />
                                        ) : (
                                            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500/20 to-pink-500/20 flex items-center justify-center border border-violet-500/10 flex-shrink-0">
                                                <span className="text-[11px] font-bold text-violet-300">{name[0]}</span>
                                            </div>
                                        )}
                                        <div className="flex-1 min-w-0">
                                            <h4 className="text-[12px] font-semibold text-white/70 truncate group-hover:text-white/90 transition-colors">{name}</h4>
                                            <span className="text-[9px] text-white/25">
                                                {character_voice_map[name] || 'Narrator'} · {char.visual_profile ? '📋' : ''} Click to view
                                            </span>
                                        </div>
                                        <ChevronRight size={12} className="text-white/15 group-hover:text-white/30 transition-colors flex-shrink-0" />
                                    </div>
                                </button>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Queue Panel */}
            <QueuePanel />

            {/* Character Profile Modal */}
            {selectedChar && (
                <CharacterProfileModal
                    character={selectedChar}
                    projectId={currentProject.id}
                    voiceMap={character_voice_map}
                    availableVoices={availableVoices}
                    onVoiceChange={handleVoiceChange}
                    onProjectUpdate={handleProjectUpdate}
                    onClose={() => setSelectedChar(null)}
                />
            )}
        </div>
    );
};

// ============================================================
// Segment-Visual Grid — inline two-track layout
// Groups consecutive segments by visual_id so one visual spans multiple rows
// ============================================================
interface VisualGroup {
    visualId: string | null;
    segments: SegmentResponse[];
    startIndex: number;
}

function groupSegmentsByVisual(segments: SegmentResponse[]): VisualGroup[] {
    const groups: VisualGroup[] = [];
    let i = 0;
    while (i < segments.length) {
        const vid = segments[i].visual_id || null;
        const group: VisualGroup = { visualId: vid, segments: [segments[i]], startIndex: i };
        i++;
        // Group consecutive segments with the same visual_id together
        while (i < segments.length && (segments[i].visual_id || null) === vid) {
            group.segments.push(segments[i]);
            i++;
        }
        groups.push(group);
    }
    return groups;
}



const InlineVisualCard: React.FC<{
    visualId: string;
    projectId: string;
    segmentCount: number;
    lastSegmentId: string;
    nextSegmentId: string | null;
    allSegments: SegmentResponse[];
    groupStartIndex: number;
    /* v2: spacious layout with full preview, bigger controls */
}> = ({ visualId, projectId, segmentCount, lastSegmentId, nextSegmentId, allSegments, groupStartIndex }) => {
    const {
        currentProject, updateVisualAsset, deleteVisualAsset,
        generateVisualAsset, generateVisualAssetPrompt, assignVisual, generating,
        visualMode, visualSettings, setVisualSettings, visualProgress, selectCandidate,
    } = useAudiobookStore();
    const [editingPrompt, setEditingPrompt] = useState(false);
    const [promptText, setPromptText] = useState('');
    const [showSettings, setShowSettings] = useState(false);
    const [showPreview, setShowPreview] = useState(false);

    // Local settings state (initialized from saved visual asset settings, then global defaults)
    const va = currentProject?.visuals?.find((v: any) => v.id === visualId);
    const [localMode, setLocalMode] = useState<VisualMode>(() =>
        (va?.visual_mode as VisualMode) || (visualMode as VisualMode)
    );
    const [fps, setFps] = useState(() => va?.gen_fps || visualSettings.fps);
    const [frames, setFrames] = useState(() => va?.gen_frames || visualSettings.frames);
    const [resPick, setResPick] = useState(() => {
        if (va?.gen_width && va?.gen_height) {
            const idx = RESOLUTION_PRESETS.findIndex(r => r.w === va.gen_width && r.h === va.gen_height);
            return idx >= 0 ? idx : 1;
        }
        return RESOLUTION_PRESETS.findIndex(r => r.w === visualSettings.width && r.h === visualSettings.height);
    });
    const [animationStyle, setAnimationStyle] = useState<string>(() =>
        va?.animation_style || 'random'
    );
    const [fillMode, setFillMode] = useState<'loop' | 'hold' | 'fade'>(() =>
        (va?.video_fill_mode as 'loop' | 'hold' | 'fade') || 'hold'
    );
    const [enableAudio, setEnableAudio] = useState(() => va?.gen_enable_audio || false);
    const [twoStage, setTwoStage] = useState(() => va?.gen_two_stage || false);
    const [numCandidates, setNumCandidates] = useState(() => va?.gen_candidates || 1);

    if (!va) return null;

    const isGenerating = generating.has(`va:${visualId}`);
    const isPromptGenerating = generating.has(`vaprompt:${visualId}`);
    const hasPortraits = !!(currentProject?.characters?.some(c => c.portrait_path));
    const charsWithPortraits = (currentProject?.characters || []).filter((c: any) => c.portrait_path);
    const [refCharacter, setRefCharacter] = useState<string>(
        va?.ref_character || charsWithPortraits[0]?.name || ''
    );

    const res = RESOLUTION_PRESETS[resPick < 0 ? 1 : resPick];
    const estTime = estimateTime(localMode, frames, res.w, res.h);
    const isVideoMode = localMode === 'video' || localMode === 'ref_video' || localMode === 'scene_video';
    const isRefMode = localMode === 'scene_image' || localMode === 'ref_video' || localMode === 'scene_video';
    const isImageMode = localMode === 'image' || localMode === 'scene_image';

    const statusColors: Record<string, string> = {
        none: 'bg-white/10 text-white/40',
        pending: 'bg-amber-500/15 text-amber-400',
        queued: 'bg-blue-500/15 text-blue-400',
        generating: 'bg-violet-500/15 text-violet-400',
        done: 'bg-emerald-500/15 text-emerald-400',
        error: 'bg-red-500/15 text-red-400',
    };

    const canExtend = nextSegmentId != null;
    const canShrink = segmentCount > 1;
    const handleExtend = () => { if (nextSegmentId) assignVisual(nextSegmentId, visualId); };
    const handleShrink = () => { assignVisual(lastSegmentId, null); };

    const handleSaveAndExecute = async (action: 'params' | 'gen' | 'prompt') => {
        // Auto-save edited prompt if needed before generating
        if (editingPrompt) {
            if (promptText !== va.scene_prompt) {
                await updateVisualAsset(va.id, { scene_prompt: promptText });
            }
            setEditingPrompt(false);
        }

        if (action === 'prompt') {
            generateVisualAssetPrompt(va.id);
        } else if (action === 'gen') {
            generateVisualAsset(va.id);
        } else if (action === 'params') {
            setVisualSettings({ fps, width: res.w, height: res.h });
            const params: VisualParams = {
                mode: localMode,
                frames, fps,
                width: res.w, height: res.h,
                ...(isImageMode ? { animation: animationStyle } : {}),
                ...(isRefMode && refCharacter ? { ref_character: refCharacter } : {}),
                ...(enableAudio ? { enable_audio: true } : {}),
                ...(twoStage ? { two_stage: true } : {}),
                ...(numCandidates > 1 ? { candidates: numCandidates } : {}),
            };
            generateVisualAsset(va.id, params);
        }
    };

    return (
        <div className="h-full flex flex-col rounded-xl overflow-hidden border border-white/[0.06] bg-white/[0.015]">
            {/* Thumbnail */}
            {va.has_visual && (() => {
                // Cache buster: combine visual_path + visual_status so the image
                // re-fetches when generation completes (status transitions to 'done')
                // even if the file path stays the same (same segment regenerated).
                const cacheBuster = encodeURIComponent(`${va.visual_path || ''}_${va.visual_status}`);
                return (
                    <div key={`thumb-${va.id}-${cacheBuster}`} className="flex-shrink-0 border-b border-white/[0.06] max-h-[220px] overflow-hidden cursor-pointer" onClick={() => setShowPreview(true)}>
                        {isBrowserVideo(va) ? (
                            <video
                                src={`${getVisualAssetFileUrl(projectId, va.id)}?t=${cacheBuster}`}
                                className="w-full h-full object-cover"
                                autoPlay muted loop playsInline
                            />
                        ) : (
                            <img
                                src={`${getVisualAssetFileUrl(projectId, va.id)}?t=${cacheBuster}`}
                                alt=""
                                className="w-full h-full object-cover"
                            />
                        )}
                    </div>
                );
            })()}

            {/* Candidate picker strip */}
            {va.candidate_count > 1 && (() => {
                const candidateUrls = Array.from({ length: va.candidate_count }, (_, i) =>
                    getVisualCandidateUrl(projectId, va.id, i)
                );
                return (
                    <div className="flex-shrink-0 border-b border-white/[0.06] bg-black/20 px-2 py-1.5">
                        <div className="flex items-center gap-1 mb-1">
                            <span className="text-[8px] font-medium text-white/30 uppercase tracking-wider">Candidates</span>
                            <span className="text-[8px] text-white/20">{va.candidate_count} variants</span>
                        </div>
                        <div className="flex gap-1 overflow-x-auto scrollbar-thin pb-0.5">
                            {candidateUrls.map((url, i) => (
                                <button
                                    key={i}
                                    onClick={() => selectCandidate(va.id, i)}
                                    className={`flex-shrink-0 relative rounded-md overflow-hidden border-2 transition-all ${va.selected_candidate === i
                                        ? 'border-emerald-400/80 shadow-[0_0_8px_rgba(52,211,153,0.3)]'
                                        : 'border-white/10 hover:border-white/30'
                                        }`}
                                    title={`Candidate ${i + 1}${va.selected_candidate === i ? ' (selected)' : ''}`}
                                >
                                    {isBrowserVideo(va) ? (
                                        <video
                                            src={`${url}?t=${Date.now()}`}
                                            className="w-16 h-12 object-cover"
                                            muted playsInline
                                            onMouseEnter={(e) => (e.target as HTMLVideoElement).play()}
                                            onMouseLeave={(e) => { const v = e.target as HTMLVideoElement; v.pause(); v.currentTime = 0; }}
                                        />
                                    ) : (
                                        <img src={`${url}?t=${Date.now()}`} alt="" className="w-16 h-12 object-cover" />
                                    )}
                                    {va.selected_candidate === i && (
                                        <div className="absolute top-0.5 right-0.5 w-3 h-3 rounded-full bg-emerald-500 flex items-center justify-center">
                                            <Check size={7} className="text-white" />
                                        </div>
                                    )}
                                    <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-1 py-0.5 text-center">
                                        <span className="text-[7px] text-white/70 font-mono">#{i + 1}</span>
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>
                );
            })()}

            {/* Real-time visual generation progress overlay */}
            {(() => {
                // Match this card's segments to the visual progress
                const segs = allSegments.filter(s => s.visual_id === visualId);
                const isLiveGenerating = visualProgress && segs.some(s => s.id === visualProgress.segmentId) && visualProgress.status !== 'idle' && visualProgress.status !== 'done';
                if (!isLiveGenerating || !visualProgress) return null;

                const statusLabels: Record<string, string> = {
                    prompt_generating: '🧠 Generating prompt…',
                    prompt_done: '📝 Prompt ready',
                    rendering: '🎨 Rendering…',
                    progress: `🎨 Rendering (${visualProgress.step}/${visualProgress.totalSteps})`,
                    error: `❌ ${visualProgress.error || 'Error'}`,
                };
                const label = statusLabels[visualProgress.status] || 'Processing…';
                const pct = visualProgress.totalSteps > 0 ? (visualProgress.step / visualProgress.totalSteps) * 100 : 0;
                const showBar = visualProgress.status === 'progress' && visualProgress.totalSteps > 0;

                return (
                    <div className="flex-shrink-0 px-3 py-2 border-b border-white/[0.06] bg-gradient-to-r from-violet-500/[0.06] to-indigo-500/[0.06]">
                        <div className="flex items-center gap-2">
                            {visualProgress.status !== 'error' && (
                                <Loader2 size={12} className="text-violet-400 animate-spin flex-shrink-0" />
                            )}
                            <span className="text-[11px] text-white/60 font-medium truncate">{label}</span>
                        </div>
                        {showBar && (
                            <div className="mt-1.5 w-full h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full rounded-full bg-gradient-to-r from-violet-500 to-indigo-500 shadow-[0_0_8px_rgba(139,92,246,0.4)]"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${pct}%` }}
                                    transition={{ duration: 0.3, ease: 'easeOut' }}
                                />
                            </div>
                        )}
                    </div>
                );
            })()}

            {/* Content */}
            <div className="flex-1 p-3 space-y-2.5 min-h-0 overflow-y-auto scrollbar-thin">
                {/* Label + status + settings toggle */}
                <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                        <span className="text-xs font-semibold text-white/60 truncate">{va.label || 'Visual'}</span>
                        <span className="text-[10px] text-white/25 flex-shrink-0">×{segmentCount}</span>
                    </div>
                    <div className="flex items-center gap-1.5 flex-shrink-0">
                        <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${statusColors[va.visual_status] || statusColors.none}`}>
                            {(() => {
                                // Show live status from WS if this card is generating
                                const segs = allSegments.filter(s => s.visual_id === visualId);
                                const isLive = visualProgress && segs.some(s => s.id === visualProgress.segmentId) && visualProgress.status !== 'idle' && visualProgress.status !== 'done';
                                if (isLive) return '⟳ generating';
                                if (isGenerating) return '⟳ gen…';
                                return va.visual_status;
                            })()}
                        </span>
                        <button
                            onClick={() => setShowSettings(!showSettings)}
                            className={`p-1 rounded-md transition-all ${showSettings
                                ? 'text-amber-400 bg-amber-500/15 border border-amber-500/20'
                                : 'text-white/30 hover:text-white/60 hover:bg-white/[0.04] border border-transparent'
                                }`}
                            title="Generation settings"
                        >
                            <Settings2 size={13} />
                        </button>
                    </div>
                </div>

                {/* Prompt */}
                {editingPrompt ? (
                    <div className="space-y-1">
                        <textarea
                            value={promptText}
                            onChange={(e) => setPromptText(e.target.value)}
                            className={`${inputStyle} text-[11px] h-16 resize-none py-2 px-3`}
                            autoFocus
                        />
                        <div className="flex gap-0.5">
                            <button
                                onClick={() => { updateVisualAsset(va.id, { scene_prompt: promptText }); setEditingPrompt(false); }}
                                className="flex-1 px-1.5 py-0.5 rounded text-[8px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/15 hover:bg-emerald-500/20 transition-all flex items-center justify-center gap-0.5"
                            >
                                <Check size={7} /> Save
                            </button>
                            <button onClick={() => setEditingPrompt(false)} className="px-1.5 py-0.5 rounded text-[8px] bg-white/[0.03] text-white/40 border border-white/[0.06] hover:bg-white/[0.06] transition-all">
                                <X size={7} />
                            </button>
                        </div>
                    </div>
                ) : (
                    <div
                        onClick={() => { setEditingPrompt(true); setPromptText(va.scene_prompt || ''); }}
                        className="text-[11px] text-white/35 bg-white/[0.02] border border-white/[0.04] rounded-lg px-3 py-2 cursor-pointer hover:bg-white/[0.04] hover:border-white/[0.08] transition-all leading-relaxed min-h-[36px]"
                    >
                        {va.scene_prompt || 'Click to add prompt…'}
                    </div>
                )}

                {/* ── Settings panel (collapsible) ── */}
                <AnimatePresence>
                    {showSettings && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.15 }}
                            className="overflow-hidden space-y-2 border-t border-b border-white/[0.04] py-2"
                        >
                            {/* Pipeline selector */}
                            <div className="space-y-1">
                                <label className="text-[8px] font-medium text-white/30 uppercase tracking-wider">Pipeline</label>
                                <div className="space-y-0.5">
                                    {VISUAL_MODES.map(m => {
                                        const disabled = m.needsRef && !hasPortraits;
                                        const isActive = localMode === m.id;
                                        return (
                                            <button
                                                key={m.id}
                                                onClick={() => !disabled && setLocalMode(m.id)}
                                                disabled={disabled}
                                                className={`w-full flex items-center gap-1.5 px-1.5 py-1 rounded text-left transition-all border ${isActive
                                                    ? 'bg-violet-500/15 text-violet-300 border-violet-500/25'
                                                    : disabled
                                                        ? 'text-white/15 border-transparent cursor-not-allowed'
                                                        : 'text-white/40 border-transparent hover:text-white/60 hover:bg-white/[0.03]'
                                                    }`}
                                            >
                                                {m.icon === 'image' ? <Image size={9} /> : <Film size={9} />}
                                                <div className="flex-1 min-w-0">
                                                    <div className="text-[9px] font-medium leading-tight truncate">{m.label}</div>
                                                </div>
                                                {m.needsRef && (
                                                    <span className={`text-[7px] px-1 py-0.5 rounded-full ${hasPortraits ? 'text-emerald-400/60' : 'text-red-400/40'}`}>
                                                        {hasPortraits ? '✓' : '⚠'}
                                                    </span>
                                                )}
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>

                            {/* Portrait selector (ref modes) */}
                            {isRefMode && charsWithPortraits.length > 0 && (
                                <div className="space-y-1">
                                    <label className="text-[8px] font-medium text-white/30 uppercase tracking-wider">Reference</label>
                                    <div className="space-y-0.5 max-h-20 overflow-y-auto">
                                        {charsWithPortraits.map((c: any) => (
                                            <button
                                                key={c.name}
                                                onClick={() => setRefCharacter(c.name)}
                                                className={`w-full flex items-center gap-1.5 px-1.5 py-1 rounded transition-all text-left ${refCharacter === c.name
                                                    ? 'bg-emerald-500/10 text-emerald-300'
                                                    : 'text-white/40 hover:text-white/60 hover:bg-white/[0.03]'
                                                    }`}
                                            >
                                                <img
                                                    src={`${getPortraitUrl(projectId, c.name)}?t=${Date.now()}`}
                                                    alt={c.name}
                                                    className="w-5 h-5 rounded object-cover border border-white/10 flex-shrink-0"
                                                />
                                                <span className="text-[9px] font-medium truncate">{c.name}</span>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Animation style (image modes) */}
                            {isImageMode && (
                                <div className="space-y-1">
                                    <label className="text-[8px] font-medium text-white/30 uppercase tracking-wider">Animation</label>
                                    <div className="grid grid-cols-3 gap-0.5">
                                        {ANIMATION_STYLES.map(s => (
                                            <button
                                                key={s.id}
                                                onClick={() => setAnimationStyle(s.id)}
                                                title={s.title}
                                                className={`py-1 rounded text-[8px] font-medium transition-all border ${animationStyle === s.id
                                                    ? 'bg-sky-500/15 text-sky-300 border-sky-500/25'
                                                    : 'text-white/30 border-transparent hover:text-white/50 hover:bg-white/[0.03]'
                                                    }`}
                                            >
                                                {s.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Video controls (video modes) */}
                            {isVideoMode && (
                                <div className="space-y-1.5">
                                    {/* Fill mode */}
                                    <div className="space-y-0.5">
                                        <label className="text-[8px] font-medium text-white/30 uppercase tracking-wider">When clip ends</label>
                                        <div className="flex rounded-lg border border-white/[0.06] overflow-hidden">
                                            {([['loop', 'Loop'], ['hold', 'Hold'], ['fade', 'Fade']] as const).map(([m, label]) => (
                                                <button
                                                    key={m}
                                                    onClick={() => setFillMode(m)}
                                                    className={`flex-1 py-1 text-[8px] font-medium transition-all ${fillMode === m
                                                        ? 'bg-violet-500/15 text-violet-300'
                                                        : 'text-white/30 hover:text-white/50'
                                                        }`}
                                                >
                                                    {label}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                    {/* Frames */}
                                    <div className="space-y-0.5">
                                        <div className="flex items-center justify-between">
                                            <label className="text-[8px] font-medium text-white/30 uppercase">Frames</label>
                                            <span className="text-[9px] text-white/40 font-mono">{frames}</span>
                                        </div>
                                        <input type="range" min={9} max={449} step={8} value={frames}
                                            onChange={e => setFrames(Number(e.target.value))}
                                            className="w-full h-1 accent-violet-500 bg-white/10 rounded-full cursor-pointer"
                                        />
                                    </div>
                                    {/* FPS */}
                                    <div className="flex items-center gap-2">
                                        <label className="text-[8px] font-medium text-white/30 uppercase w-6">FPS</label>
                                        <input type="range" min={8} max={30} step={1} value={fps}
                                            onChange={e => setFps(Number(e.target.value))}
                                            className="flex-1 h-1 accent-violet-500 bg-white/10 rounded-full cursor-pointer"
                                        />
                                        <span className="text-[9px] text-white/40 font-mono w-5 text-right">{fps}</span>
                                    </div>
                                </div>
                            )}

                            {/* Resolution */}
                            <div className="space-y-1">
                                <label className="text-[8px] font-medium text-white/30 uppercase tracking-wider">Resolution</label>
                                <div className="grid grid-cols-2 gap-0.5">
                                    {RESOLUTION_PRESETS.map((r, i) => (
                                        <button
                                            key={r.label}
                                            onClick={() => setResPick(i)}
                                            className={`py-1 rounded text-[8px] font-mono transition-all border ${resPick === i
                                                ? 'bg-amber-500/10 text-amber-300 border-amber-500/20'
                                                : 'text-white/30 border-transparent hover:text-white/50 hover:bg-white/[0.03]'
                                                }`}
                                        >
                                            {r.label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* LTX-2.3 Options (video modes only) */}
                            {isVideoMode && (
                                <div className="space-y-1">
                                    <label className="text-[8px] font-medium text-white/30 uppercase tracking-wider">Options</label>
                                    <div className="space-y-0.5">
                                        <button
                                            onClick={() => setEnableAudio(!enableAudio)}
                                            className={`w-full flex items-center gap-1.5 px-1.5 py-1 rounded text-left transition-all border ${enableAudio
                                                ? 'bg-sky-500/15 text-sky-300 border-sky-500/25'
                                                : 'text-white/30 border-transparent hover:text-white/50 hover:bg-white/[0.03]'
                                                }`}
                                        >
                                            <span className="text-[9px]">{enableAudio ? '🔊' : '🔇'}</span>
                                            <span className="text-[9px] font-medium">Audio</span>
                                        </button>
                                        <button
                                            onClick={() => setTwoStage(!twoStage)}
                                            className={`w-full flex items-center gap-1.5 px-1.5 py-1 rounded text-left transition-all border ${twoStage
                                                ? 'bg-amber-500/15 text-amber-300 border-amber-500/25'
                                                : 'text-white/30 border-transparent hover:text-white/50 hover:bg-white/[0.03]'
                                                }`}
                                        >
                                            <span className="text-[9px]">⬆</span>
                                            <span className="text-[9px] font-medium">2× Upscale</span>
                                        </button>
                                    </div>
                                </div>
                            )}

                            {/* Candidates slider (video modes only) */}
                            {isVideoMode && (
                                <div className="space-y-0.5">
                                    <div className="flex items-center justify-between">
                                        <label className="text-[8px] font-medium text-white/30 uppercase">Candidates</label>
                                        <span className="text-[9px] text-white/40 font-mono">{numCandidates}</span>
                                    </div>
                                    <input type="range" min={1} max={5} step={1} value={numCandidates}
                                        onChange={e => setNumCandidates(Number(e.target.value))}
                                        className="w-full h-1 accent-violet-500 bg-white/10 rounded-full cursor-pointer"
                                    />
                                    <p className="text-[7px] text-white/20">Generate multiple variants with random seeds, then pick the best</p>
                                </div>
                            )}

                            {/* Time estimate + Generate */}
                            <div className="flex items-center justify-between gap-1 pt-1 border-t border-white/[0.04]">
                                <div className="flex items-center gap-1 text-white/30">
                                    <Timer size={9} />
                                    <span className="text-[9px]">{estTime}</span>
                                </div>
                                <button
                                    onClick={() => handleSaveAndExecute('params')}
                                    disabled={isGenerating || va.assigned_segments === 0}
                                    className="px-2.5 py-1 rounded-lg text-[9px] font-semibold bg-gradient-to-r from-emerald-500/80 to-teal-600/80 text-white shadow-sm hover:from-emerald-500 hover:to-teal-600 transition-all active:scale-[0.98] disabled:opacity-40 flex items-center gap-1"
                                >
                                    {isGenerating ? <Loader2 size={9} className="animate-spin" /> : <Zap size={9} />}
                                    Generate
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Quick actions (when settings hidden) */}
                {!showSettings && (
                    <div className="flex gap-0.5 flex-wrap">
                        <button
                            onClick={() => handleSaveAndExecute('prompt')}
                            disabled={isPromptGenerating || va.assigned_segments === 0}
                            className="flex-1 px-1 py-1 rounded text-[8px] font-medium bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/15 text-amber-400 transition-all disabled:opacity-40 flex items-center justify-center gap-0.5"
                            title="Auto-generate scene prompt"
                        >
                            {isPromptGenerating ? <Loader2 size={8} className="animate-spin" /> : <Sparkles size={8} />}
                            Prompt
                        </button>
                        <button
                            onClick={() => handleSaveAndExecute('gen')}
                            disabled={isGenerating || va.assigned_segments === 0}
                            className="flex-1 px-1 py-1 rounded text-[8px] font-medium bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/15 text-emerald-400 transition-all disabled:opacity-40 flex items-center justify-center gap-0.5"
                            title="Generate visual"
                        >
                            {isGenerating ? <Loader2 size={8} className="animate-spin" /> : <Image size={8} />}
                            Gen
                        </button>
                        {/* Extend / Shrink segment coverage */}
                        <button
                            onClick={handleExtend}
                            disabled={!canExtend}
                            className="px-1.5 py-1 rounded text-[8px] font-medium bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/15 text-blue-400 transition-all disabled:opacity-20 flex items-center justify-center gap-0.5"
                            title={canExtend ? "Extend: assign next segment to this visual" : "Next segment already has a visual"}
                        >
                            <ChevronDown size={8} /> +
                        </button>
                        <button
                            onClick={handleShrink}
                            disabled={!canShrink}
                            className="px-1.5 py-1 rounded text-[8px] font-medium bg-orange-500/10 hover:bg-orange-500/20 border border-orange-500/15 text-orange-400 transition-all disabled:opacity-20 flex items-center justify-center gap-0.5"
                            title={canShrink ? "Shrink: unassign last segment from this visual" : "Only one segment assigned"}
                        >
                            <ChevronUp size={8} /> −
                        </button>
                        <button
                            onClick={() => { if (window.confirm('Delete visual?')) deleteVisualAsset(va.id); }}
                            className="px-1 py-1 rounded text-[8px] bg-red-500/10 hover:bg-red-500/20 border border-red-500/15 text-red-400 transition-all flex items-center justify-center"
                            title="Delete"
                        >
                            <Trash2 size={8} />
                        </button>
                    </div>
                )}

                {/* Span controls */}
                {(canExtend || canShrink) && (
                    <div className="flex gap-1.5 pt-1.5 border-t border-white/[0.04]">
                        {canExtend && (
                            <button
                                onClick={handleExtend}
                                className="flex-1 px-2 py-1 rounded-lg text-[10px] font-medium bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.06] text-white/35 hover:text-white/60 transition-all flex items-center justify-center gap-1"
                                title="Extend this visual to cover the next segment"
                            >
                                <ChevronDown size={10} /> Extend ↓
                            </button>
                        )}
                        {canShrink && (
                            <button
                                onClick={handleShrink}
                                className="flex-1 px-2 py-1 rounded-lg text-[10px] font-medium bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.06] text-white/35 hover:text-white/60 transition-all flex items-center justify-center gap-1"
                                title="Remove last segment from this visual"
                            >
                                <ChevronUp size={10} /> Shrink ↑
                            </button>
                        )}
                    </div>
                )}
            </div>

            {/* Preview Modal — portalled to body to escape overflow-hidden + framer-motion stacking context */}
            {showPreview && va.has_visual && ReactDOM.createPortal(
                <AnimatePresence>
                    <motion.div
                        key={`preview-${va.id}`}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setShowPreview(false)}
                        className="fixed inset-0 z-[9999] flex bg-black/95 backdrop-blur-md"
                    >
                        {/* Main content area */}
                        <div className="flex-1 flex items-center justify-center p-6 min-w-0" onClick={(e) => e.stopPropagation()}>
                            {isBrowserVideo(va) ? (
                                <video
                                    src={`${getVisualAssetFileUrl(projectId, va.id)}?t=${Date.now()}`}
                                    className="max-w-full max-h-[90vh] object-contain rounded-lg shadow-2xl"
                                    autoPlay controls loop
                                />
                            ) : (
                                <img
                                    src={`${getVisualAssetFileUrl(projectId, va.id)}?t=${Date.now()}`}
                                    alt=""
                                    className="max-w-full max-h-[90vh] object-contain rounded-lg shadow-2xl"
                                />
                            )}
                        </div>

                        {/* Metadata sidebar */}
                        <motion.div
                            initial={{ x: 40, opacity: 0 }}
                            animate={{ x: 0, opacity: 1 }}
                            exit={{ x: 40, opacity: 0 }}
                            transition={{ delay: 0.1 }}
                            className="w-80 flex-shrink-0 bg-white/[0.03] border-l border-white/[0.06] p-5 overflow-y-auto flex flex-col gap-4"
                            onClick={(e) => e.stopPropagation()}
                        >
                            {/* Close button */}
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-semibold text-white/80">{va.label || 'Visual Preview'}</span>
                                <button
                                    onClick={() => setShowPreview(false)}
                                    className="text-white/40 hover:text-white p-1.5 rounded-lg hover:bg-white/10 transition-all"
                                >
                                    <X size={16} />
                                </button>
                            </div>

                            {/* Status */}
                            <div className="flex items-center gap-2">
                                <span className={`text-[10px] font-semibold px-2.5 py-1 rounded-full ${statusColors[va.visual_status] || statusColors.none}`}>
                                    {va.visual_status}
                                </span>
                                <span className="text-[10px] text-white/25">
                                    {va.visual_type || 'unknown'} • {va.visual_mode || 'default'}
                                </span>
                            </div>

                            {/* Scene Prompt */}
                            <div className="space-y-1.5">
                                <div className="flex items-center gap-1.5 text-white/40">
                                    <FileText size={11} />
                                    <span className="text-[10px] font-semibold uppercase tracking-wider">Scene Prompt</span>
                                </div>
                                <div className="text-[11px] text-white/60 leading-relaxed bg-white/[0.03] border border-white/[0.06] rounded-lg px-3 py-2.5 max-h-40 overflow-y-auto scrollbar-thin">
                                    {va.scene_prompt || 'No prompt'}
                                </div>
                            </div>

                            {/* Generation Parameters */}
                            <div className="space-y-1.5">
                                <div className="flex items-center gap-1.5 text-white/40">
                                    <Sliders size={11} />
                                    <span className="text-[10px] font-semibold uppercase tracking-wider">Parameters</span>
                                </div>
                                <div className="grid grid-cols-2 gap-1.5">
                                    {va.gen_width && va.gen_height && (
                                        <div className="bg-white/[0.03] border border-white/[0.06] rounded-lg px-2.5 py-1.5">
                                            <div className="text-[8px] text-white/30 uppercase">Resolution</div>
                                            <div className="text-[11px] text-white/60 font-mono">{va.gen_width}×{va.gen_height}</div>
                                        </div>
                                    )}
                                    {va.gen_frames && (
                                        <div className="bg-white/[0.03] border border-white/[0.06] rounded-lg px-2.5 py-1.5">
                                            <div className="text-[8px] text-white/30 uppercase">Frames</div>
                                            <div className="text-[11px] text-white/60 font-mono">{va.gen_frames}</div>
                                        </div>
                                    )}
                                    {va.gen_fps && (
                                        <div className="bg-white/[0.03] border border-white/[0.06] rounded-lg px-2.5 py-1.5">
                                            <div className="text-[8px] text-white/30 uppercase">FPS</div>
                                            <div className="text-[11px] text-white/60 font-mono">{va.gen_fps}</div>
                                        </div>
                                    )}
                                    {va.animation_style && (
                                        <div className="bg-white/[0.03] border border-white/[0.06] rounded-lg px-2.5 py-1.5">
                                            <div className="text-[8px] text-white/30 uppercase">Animation</div>
                                            <div className="text-[11px] text-white/60">{va.animation_style}</div>
                                        </div>
                                    )}
                                    {va.video_fill_mode && (
                                        <div className="bg-white/[0.03] border border-white/[0.06] rounded-lg px-2.5 py-1.5">
                                            <div className="text-[8px] text-white/30 uppercase">Fill Mode</div>
                                            <div className="text-[11px] text-white/60">{va.video_fill_mode}</div>
                                        </div>
                                    )}
                                    <div className="bg-white/[0.03] border border-white/[0.06] rounded-lg px-2.5 py-1.5">
                                        <div className="text-[8px] text-white/30 uppercase">Segments</div>
                                        <div className="text-[11px] text-white/60 font-mono">{va.assigned_segments}</div>
                                    </div>
                                </div>
                            </div>

                            {/* Reference Character */}
                            {va.ref_character && (
                                <div className="space-y-1.5">
                                    <div className="flex items-center gap-1.5 text-white/40">
                                        <Users size={11} />
                                        <span className="text-[10px] font-semibold uppercase tracking-wider">Reference Character</span>
                                    </div>
                                    <div className="flex items-center gap-2.5 bg-white/[0.03] border border-white/[0.06] rounded-lg px-3 py-2">
                                        {currentProject?.characters?.find((c: any) => c.name === va.ref_character)?.portrait_path && (
                                            <img
                                                src={`${getPortraitUrl(projectId, va.ref_character)}?t=1`}
                                                alt={va.ref_character}
                                                className="w-10 h-10 rounded-lg object-cover border border-white/10"
                                            />
                                        )}
                                        <span className="text-[11px] text-white/60 font-medium">{va.ref_character}</span>
                                    </div>
                                </div>
                            )}

                            {/* Timestamp */}
                            {va.created_at && (
                                <div className="flex items-center gap-1.5 text-white/25 text-[10px]">
                                    <Clock size={10} />
                                    <span>{new Date(va.created_at).toLocaleString()}</span>
                                </div>
                            )}

                            {/* Actions */}
                            <div className="mt-auto pt-3 border-t border-white/[0.06] space-y-1.5">
                                <button
                                    onClick={() => { setShowPreview(false); setShowSettings(true); }}
                                    className="w-full px-3 py-2 rounded-lg text-[11px] font-medium bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/15 text-violet-300 transition-all flex items-center justify-center gap-1.5"
                                >
                                    <Zap size={12} /> Regenerate
                                </button>
                                <button
                                    onClick={() => { setShowPreview(false); setEditingPrompt(true); setPromptText(va.scene_prompt || ''); }}
                                    className="w-full px-3 py-2 rounded-lg text-[11px] font-medium bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/15 text-amber-300 transition-all flex items-center justify-center gap-1.5"
                                >
                                    <Edit3 size={12} /> Edit Prompt
                                </button>
                            </div>
                        </motion.div>
                    </motion.div>
                </AnimatePresence>,
                document.body
            )}
        </div>
    );
};

const SegmentVisualGrid: React.FC<{
    segments: SegmentResponse[];
    projectId: string;
    voices: string[];
    generatingSegmentId?: string | null;
}> = ({ segments, projectId, voices, generatingSegmentId }) => {
    const { createVisualAsset, assignVisual } = useAudiobookStore();
    const groups = groupSegmentsByVisual(segments);

    const handleCreateAndAssign = async (segmentId: string) => {
        await createVisualAsset();
        // After creation, the newest visual is the last in the list
        // We need to re-read from store — but since createVisualAsset updates currentProject,
        // the component will re-render. For now, user can assign from the dropdown.
    };

    return (
        <div className="space-y-0">
            {groups.map((group, gIdx) => {
                const nextGroupFirstSeg = gIdx < groups.length - 1 ? groups[gIdx + 1].segments[0] : null;

                return (
                    <div key={`g-${group.startIndex}`} className="flex gap-4 mb-1.5 w-full">
                        {/* Left track: segments */}
                        <div className="flex-1 min-w-[300px] space-y-1.5">
                            {group.segments.map((seg, i) => (
                                <div key={seg.id} className={generatingSegmentId === seg.id ? 'ring-1 ring-amber-500/30 rounded-xl' : ''}>
                                    <SegmentRow segment={seg} projectId={projectId} voices={voices} index={group.startIndex + i} />
                                </div>
                            ))}
                        </div>

                        {/* Right track: visual card (spans full height) */}
                        <div className="w-[340px] flex-shrink-0">
                            {group.visualId ? (
                                <InlineVisualCard
                                    visualId={group.visualId}
                                    projectId={projectId}
                                    segmentCount={group.segments.length}
                                    lastSegmentId={group.segments[group.segments.length - 1].id}
                                    nextSegmentId={nextGroupFirstSeg?.id || null}
                                    allSegments={segments}
                                    groupStartIndex={group.startIndex}
                                />
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center rounded-xl border border-dashed border-white/[0.08] bg-white/[0.01] min-h-[80px] gap-2.5 p-4">
                                    <div className="text-center">
                                        <Image size={20} className="mx-auto text-white/15 mb-1" />
                                        <p className="text-[11px] text-white/25">No visual assigned</p>
                                        <p className="text-[9px] text-white/15 mt-0.5">{group.segments.length} segment{group.segments.length > 1 ? 's' : ''}</p>
                                    </div>
                                    <button
                                        onClick={() => handleCreateAndAssign(group.segments[0].id)}
                                        className="px-3 py-1.5 rounded-lg text-[11px] font-medium bg-emerald-500/[0.08] hover:bg-emerald-500/15 border border-emerald-500/15 text-emerald-400/70 hover:text-emerald-400 transition-all flex items-center gap-1.5"
                                        title="Create and assign a new visual"
                                    >
                                        <Plus size={11} /> Add Visual
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

// ============================================================
// Main Audiobook Generator Page
// ============================================================
const VisualsColumn: React.FC<{ projectId: string; visuals: VisualAsset[]; characters?: { name: string }[] }> = ({ projectId, visuals, characters }) => {
    const {
        createVisualAsset, updateVisualAsset, deleteVisualAsset,
        generateVisualAsset, generateVisualAssetPrompt, generating,
    } = useAudiobookStore();
    const [editingPrompt, setEditingPrompt] = useState<string | null>(null);
    const [promptText, setPromptText] = useState('');
    const [expandedId, setExpandedId] = useState<string | null>(null);

    const statusColors: Record<string, string> = {
        none: 'bg-white/10 text-white/40',
        pending: 'bg-amber-500/15 text-amber-400',
        queued: 'bg-blue-500/15 text-blue-400',
        generating: 'bg-violet-500/15 text-violet-400',
        done: 'bg-emerald-500/15 text-emerald-400',
        error: 'bg-red-500/15 text-red-400',
    };

    return (
        <div className="w-[320px] flex-shrink-0 border-l border-white/[0.04] flex flex-col bg-gradient-to-b from-white/[0.01] to-transparent">
            {/* Header */}
            <div className="p-4 pb-3 border-b border-white/[0.04]">
                <div className="flex items-center justify-between mb-2">
                    <h2 className="text-xs font-semibold text-white/40 uppercase tracking-[0.12em] flex items-center gap-1.5">
                        <Palette size={12} className="text-emerald-400/60" />
                        Visuals
                    </h2>
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] text-white/25 tabular-nums">
                            {visuals.filter(v => v.visual_status === 'done').length}/{visuals.length}
                        </span>
                        <button
                            onClick={() => createVisualAsset()}
                            className="w-7 h-7 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/10 hover:border-emerald-500/20 flex items-center justify-center text-emerald-400/70 hover:text-emerald-400 transition-all"
                            title="Create new visual asset"
                        >
                            <Plus size={13} />
                        </button>
                    </div>
                </div>
            </div>

            {/* Visual cards list */}
            <div className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-2">
                {visuals.length === 0 && (
                    <div className="text-center py-8">
                        <div className="w-12 h-12 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mx-auto mb-3">
                            <Image size={20} className="text-white/15" />
                        </div>
                        <p className="text-[11px] text-white/20 mb-3">No visuals yet</p>
                        <button
                            onClick={() => createVisualAsset()}
                            className="px-3 py-1.5 rounded-lg text-[10px] font-medium bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/15 text-emerald-400 transition-all inline-flex items-center gap-1"
                        >
                            <Plus size={10} /> New Visual
                        </button>
                    </div>
                )}

                {visuals.map((va) => {
                    const isExpanded = expandedId === va.id;
                    const isGenerating = generating.has(`va:${va.id}`);
                    const isPromptGenerating = generating.has(`vaprompt:${va.id}`);
                    const isEditing = editingPrompt === va.id;

                    return (
                        <motion.div
                            key={va.id}
                            layout
                            className={`${cardStyle} overflow-hidden transition-all ${va.visual_status === 'done' ? 'border-emerald-500/10' :
                                va.visual_status === 'generating' || va.visual_status === 'queued' ? 'border-violet-500/10' :
                                    'border-white/[0.06]'
                                }`}
                        >
                            {/* Card header — clickable to expand */}
                            <div
                                className="px-3 py-2.5 cursor-pointer hover:bg-white/[0.02] transition-all"
                                onClick={() => setExpandedId(isExpanded ? null : va.id)}
                            >
                                <div className="flex items-center gap-2">
                                    {/* Thumbnail / Icon */}
                                    <div className="w-10 h-10 rounded-lg overflow-hidden flex-shrink-0 bg-white/[0.03] border border-white/[0.06] flex items-center justify-center">
                                        {va.has_visual ? (
                                            <img
                                                src={`${getVisualAssetFileUrl(projectId, va.id)}?t=${Date.now()}`}
                                                alt=""
                                                className="w-full h-full object-cover"
                                            />
                                        ) : (
                                            <Image size={14} className="text-white/15" />
                                        )}
                                    </div>

                                    {/* Label + metadata */}
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-1.5">
                                            <span className="text-[11px] font-medium text-white/70 truncate">
                                                {va.label || `Visual ${va.id}`}
                                            </span>
                                            {isExpanded ? <ChevronDown size={10} className="text-white/30 flex-shrink-0" /> : <ChevronRight size={10} className="text-white/30 flex-shrink-0" />}
                                        </div>
                                        <div className="flex items-center gap-1.5 mt-0.5">
                                            <span className={`text-[8px] font-medium px-1.5 py-0.5 rounded-full ${statusColors[va.visual_status] || statusColors.none}`}>
                                                {isGenerating ? '⟳ generating' : va.visual_status}
                                            </span>
                                            {va.assigned_segments > 0 && (
                                                <span className="text-[8px] text-white/25">
                                                    {va.assigned_segments} seg{va.assigned_segments > 1 ? 's' : ''}
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Expanded content */}
                            <AnimatePresence>
                                {isExpanded && (
                                    <motion.div
                                        initial={{ height: 0, opacity: 0 }}
                                        animate={{ height: 'auto', opacity: 1 }}
                                        exit={{ height: 0, opacity: 0 }}
                                        transition={{ duration: 0.2 }}
                                        className="overflow-hidden"
                                    >
                                        <div className="px-3 pb-3 space-y-2 border-t border-white/[0.04] pt-2">
                                            {/* Large thumbnail */}
                                            {va.has_visual && (
                                                <div className="rounded-lg overflow-hidden border border-white/[0.06]">
                                                    {isBrowserVideo(va) ? (
                                                        <video
                                                            src={`${getVisualAssetFileUrl(projectId, va.id)}?t=${Date.now()}`}
                                                            className="w-full h-auto"
                                                            autoPlay muted loop playsInline
                                                        />
                                                    ) : (
                                                        <img
                                                            src={`${getVisualAssetFileUrl(projectId, va.id)}?t=${Date.now()}`}
                                                            alt=""
                                                            className="w-full h-auto"
                                                        />
                                                    )}
                                                </div>
                                            )}

                                            {/* Scene prompt */}
                                            <div className="space-y-1">
                                                <label className="text-[9px] text-white/30 uppercase tracking-wider">Scene Prompt</label>
                                                {isEditing ? (
                                                    <div className="space-y-1">
                                                        <textarea
                                                            value={promptText}
                                                            onChange={(e) => setPromptText(e.target.value)}
                                                            className={`${inputStyle} text-[10px] h-16 resize-none`}
                                                            autoFocus
                                                        />
                                                        <div className="flex gap-1">
                                                            <button
                                                                onClick={() => {
                                                                    updateVisualAsset(va.id, { scene_prompt: promptText });
                                                                    setEditingPrompt(null);
                                                                }}
                                                                className="flex-1 px-2 py-1 rounded-lg text-[9px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/15 hover:bg-emerald-500/20 transition-all flex items-center justify-center gap-1"
                                                            >
                                                                <Check size={9} /> Save
                                                            </button>
                                                            <button
                                                                onClick={() => setEditingPrompt(null)}
                                                                className="px-2 py-1 rounded-lg text-[9px] font-medium bg-white/[0.03] text-white/40 border border-white/[0.06] hover:bg-white/[0.06] transition-all"
                                                            >
                                                                <X size={9} />
                                                            </button>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <div
                                                        onClick={() => {
                                                            setEditingPrompt(va.id);
                                                            setPromptText(va.scene_prompt || '');
                                                        }}
                                                        className="text-[10px] text-white/40 bg-white/[0.02] rounded-lg px-2 py-1.5 cursor-pointer hover:bg-white/[0.04] transition-all min-h-[28px] line-clamp-3"
                                                    >
                                                        {va.scene_prompt || 'Click to add prompt…'}
                                                    </div>
                                                )}
                                            </div>

                                            {/* Action buttons */}
                                            <div className="flex gap-1 flex-wrap">
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); generateVisualAssetPrompt(va.id); }}
                                                    disabled={isPromptGenerating || va.assigned_segments === 0}
                                                    className="flex-1 px-2 py-1.5 rounded-lg text-[9px] font-medium bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/15 text-amber-400 transition-all disabled:opacity-40 flex items-center justify-center gap-1"
                                                    title={va.assigned_segments === 0 ? 'Assign to a segment first' : 'Auto-generate scene prompt'}
                                                >
                                                    {isPromptGenerating ? <Loader2 size={9} className="animate-spin" /> : <Sparkles size={9} />}
                                                    Prompt
                                                </button>
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); generateVisualAsset(va.id); }}
                                                    disabled={isGenerating || va.assigned_segments === 0}
                                                    className="flex-1 px-2 py-1.5 rounded-lg text-[9px] font-medium bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/15 text-emerald-400 transition-all disabled:opacity-40 flex items-center justify-center gap-1"
                                                    title={va.assigned_segments === 0 ? 'Assign to a segment first' : 'Generate visual'}
                                                >
                                                    {isGenerating ? <Loader2 size={9} className="animate-spin" /> : <Image size={9} />}
                                                    Generate
                                                </button>
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        if (window.confirm('Delete this visual?')) deleteVisualAsset(va.id);
                                                    }}
                                                    className="px-2 py-1.5 rounded-lg text-[9px] font-medium bg-red-500/10 hover:bg-red-500/20 border border-red-500/15 text-red-400 transition-all flex items-center justify-center"
                                                    title="Delete visual"
                                                >
                                                    <Trash2 size={9} />
                                                </button>
                                            </div>

                                            {/* Reference character selector */}
                                            {characters && characters.length > 0 && (
                                                <div className="space-y-1">
                                                    <label className="text-[9px] text-white/30 uppercase tracking-wider">Ref Character</label>
                                                    <select
                                                        value={va.ref_character || ''}
                                                        onChange={(e) => updateVisualAsset(va.id, { ref_character: e.target.value || undefined })}
                                                        className={`${selectStyle} w-full text-[10px] py-1`}
                                                    >
                                                        <option value="">None</option>
                                                        {characters.map((c) => (
                                                            <option key={c.name} value={c.name}>{c.name}</option>
                                                        ))}
                                                    </select>
                                                </div>
                                            )}
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </motion.div>
                    );
                })}
            </div>
        </div>
    );
};

// ============================================================
// Main Audiobook Generator Page
// ============================================================
const AudiobookGenerator: React.FC = () => {
    const {
        projects, currentProject, availableVoices, loading, analyzing, error,
        fetchProjects, fetchVoices, loadProject, deleteProject, clearError, retryFailed,
        generateAllVisuals, exportVideo, visualMode, setVisualMode, videoExporting,
        createVisualAsset, assignVisual,
    } = useAudiobookStore();

    const [showNewModal, setShowNewModal] = useState(false);
    const [autoPlay, setAutoPlay] = useState(false);
    const autoPlayAudioRef = useRef<HTMLAudioElement>(null);

    const {
        connected,
        progress,
        startGenerateAll,
        startGenerateChapter,
        cancelGeneration,
    } = useGenerationSocket(currentProject?.id || null);

    useEffect(() => {
        fetchProjects();
        fetchVoices();
    }, []);

    useEffect(() => {
        if (error) { toast.error(error); clearError(); }
    }, [error]);

    useEffect(() => {
        if (autoPlay && progress.completedSegmentIds.length > 0 && currentProject) {
            const lastId = progress.completedSegmentIds[progress.completedSegmentIds.length - 1];
            if (autoPlayAudioRef.current) {
                autoPlayAudioRef.current.src = getSegmentAudioUrl(currentProject.id, lastId);
                autoPlayAudioRef.current.play().catch(() => { });
            }
        }
    }, [progress.completedSegmentIds, autoPlay, currentProject]);

    // Toast when video export completes
    const prevVideoExporting = React.useRef(false);
    useEffect(() => {
        if (prevVideoExporting.current && !videoExporting) {
            toast.success('Video ready — click Download MP4');
        }
        prevVideoExporting.current = videoExporting;
    }, [videoExporting]);

    const progressPercent = progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;

    return (
        <Layout>
            <audio ref={autoPlayAudioRef} />

            <div className="flex h-full">
                {/* ---- Sidebar ---- */}
                <div className="w-[280px] flex-shrink-0 border-r border-white/[0.04] flex flex-col bg-gradient-to-b from-white/[0.01] to-transparent">
                    {/* Projects header */}
                    <div className="p-5 pb-3">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-xs font-semibold text-white/40 uppercase tracking-[0.12em] flex items-center gap-1.5">
                                <BookOpen size={12} className="text-amber-400/60" />
                                Projects
                            </h2>
                            <button
                                onClick={() => setShowNewModal(true)}
                                className="w-7 h-7 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/10 hover:border-amber-500/20 flex items-center justify-center text-amber-400/70 hover:text-amber-400 transition-all"
                            >
                                <Plus size={13} />
                            </button>
                        </div>

                        {/* Project list */}
                        <div className="space-y-1 max-h-52 overflow-y-auto scrollbar-thin">
                            {projects.length === 0 && (
                                <p className="text-[11px] text-white/20 py-3 text-center">No projects yet</p>
                            )}
                            {projects.map((p: any) => (
                                <button
                                    key={p.id}
                                    className={`group w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl cursor-pointer transition-all text-left
                                        ${currentProject?.id === p.id
                                            ? 'bg-amber-500/[0.08] border border-amber-500/15 text-amber-400 shadow-sm'
                                            : 'bg-transparent border border-transparent hover:bg-white/[0.03] hover:border-white/[0.05] text-white/50'
                                        }`}
                                    onClick={() => loadProject(p.id)}
                                >
                                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${currentProject?.id === p.id ? 'bg-amber-500/15' : 'bg-white/[0.04]'}`}>
                                        <FileText size={13} className={currentProject?.id === p.id ? 'text-amber-400' : 'text-white/30'} />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <span className="text-[12px] font-medium truncate block">{p.name}</span>
                                        <div className="flex items-center gap-2 mt-0.5">
                                            <div className="w-12 h-1 bg-white/[0.04] rounded-full overflow-hidden">
                                                <div className="h-full bg-amber-500/50 rounded-full" style={{ width: `${p.progress * 100}%` }} />
                                            </div>
                                            <span className="text-[9px] text-white/20 tabular-nums">{Math.round(p.progress * 100)}%</span>
                                        </div>
                                    </div>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); deleteProject(p.id); }}
                                        className="opacity-0 group-hover:opacity-100 p-1 rounded text-red-400/40 hover:text-red-400 transition-all"
                                    >
                                        <Trash2 size={11} />
                                    </button>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Separator */}
                    <div className="mx-5 h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />

                    {/* Controls & voices — shown when project is loaded */}
                    {currentProject && (
                        <div className="p-4 flex-1 overflow-y-auto scrollbar-thin space-y-5">

                            {/* ── Audio Generation ─────────────────── */}
                            <SidebarSection
                                icon={<Music size={10} className="text-amber-400/60" />}
                                label="Audio Generation"
                                status={`${currentProject.done_segments}/${currentProject.total_segments}`}
                                statusColor={currentProject.done_segments === currentProject.total_segments ? 'text-emerald-400/60' : 'text-amber-400/60'}
                            >
                                {progress.isGenerating ? (
                                    <>
                                        <div className="rounded-xl border border-amber-500/15 bg-amber-500/[0.04] p-2.5 space-y-2">
                                            <div className="flex items-center gap-2">
                                                <div className="w-1.5 h-1.5 bg-amber-500 rounded-full animate-pulse" />
                                                <span className="text-[10px] font-semibold text-amber-400 flex-1">Generating…</span>
                                                <span className="text-[9px] text-white/30 tabular-nums">
                                                    {progress.done}/{progress.total}
                                                </span>
                                            </div>
                                            <ProgressBar progress={progress.total > 0 ? progress.done / progress.total : 0} size="sm" glow />
                                            {progress.currentSegmentPreview && (
                                                <p className="text-[9px] text-white/25 truncate italic">
                                                    "{progress.currentSegmentPreview}…"
                                                </p>
                                            )}
                                        </div>
                                        <button
                                            onClick={cancelGeneration}
                                            className="w-full px-3 py-2 rounded-xl text-[11px] font-semibold bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 text-red-400 transition-all flex items-center justify-center gap-1.5"
                                        >
                                            <Square size={11} /> Stop Generation
                                        </button>
                                    </>
                                ) : (
                                    <button
                                        onClick={startGenerateAll}
                                        disabled={!connected}
                                        className="w-full px-3 py-2 rounded-xl text-[11px] font-semibold bg-gradient-to-r from-amber-500/15 to-orange-500/15 hover:from-amber-500/25 hover:to-orange-500/25 border border-amber-500/15 text-amber-400 hover:text-amber-300 transition-all disabled:opacity-40 flex items-center justify-center gap-1.5"
                                        title="Generate TTS audio for all pending segments"
                                    >
                                        <Zap size={12} /> Generate All Audio
                                    </button>
                                )}

                                <button
                                    onClick={() => setAutoPlay(!autoPlay)}
                                    className={`w-full px-3 py-1.5 rounded-xl text-[10px] font-medium border transition-all flex items-center justify-center gap-1.5
                                        ${autoPlay
                                            ? 'bg-emerald-500/[0.08] border-emerald-500/20 text-emerald-400'
                                            : 'bg-white/[0.03] border-white/[0.06] text-white/35 hover:text-white/60'
                                        }`}
                                    title="Auto-play each segment as it finishes generating"
                                >
                                    <Volume2 size={10} /> Auto-play {autoPlay ? 'ON' : 'OFF'}
                                </button>

                                {currentProject.error_segments > 0 && (
                                    <button
                                        onClick={() => retryFailed()}
                                        className="w-full px-3 py-1.5 rounded-xl text-[10px] font-medium bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/15 hover:border-red-500/25 transition-all flex items-center justify-center gap-1.5"
                                    >
                                        <RefreshCw size={10} /> Retry {currentProject.error_segments} Failed
                                    </button>
                                )}
                            </SidebarSection>

                            {/* ── Visual Generation ─────────────────── */}
                            <SidebarSection
                                icon={<Palette size={10} className="text-emerald-400/60" />}
                                label="Visual Generation"
                                status={`${currentProject.visual_ready}/${currentProject.total_segments}`}
                                statusColor={currentProject.visual_ready === currentProject.total_segments ? 'text-emerald-400/60' : 'text-white/20'}
                            >
                                {/* Image / Video mode toggle */}
                                <div className="flex rounded-xl border border-white/[0.08] overflow-hidden">
                                    {(['image', 'video'] as const).map(m => (
                                        <button
                                            key={m}
                                            onClick={() => setVisualMode(m)}
                                            className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 text-[10px] font-medium transition-all
                                                ${visualMode === m
                                                    ? m === 'image'
                                                        ? 'bg-emerald-500/20 text-emerald-300'
                                                        : 'bg-violet-500/20 text-violet-300'
                                                    : 'text-white/30 hover:text-white/50 hover:bg-white/[0.03]'
                                                }`}
                                        >
                                            {m === 'image' ? <Image size={10} /> : <Film size={10} />}
                                            {m.charAt(0).toUpperCase() + m.slice(1)}
                                        </button>
                                    ))}
                                </div>

                                <button
                                    onClick={() => generateAllVisuals()}
                                    disabled={loading}
                                    className="w-full px-3 py-2 rounded-xl text-[11px] font-semibold bg-emerald-500/[0.08] hover:bg-emerald-500/15 border border-emerald-500/15 text-emerald-400 hover:text-emerald-300 transition-all disabled:opacity-40 flex items-center justify-center gap-1.5"
                                    title="Generate scene prompts (if missing) then visuals for all segments"
                                >
                                    {visualMode === 'video' ? <Film size={12} /> : <Image size={12} />}
                                    Generate All Visuals
                                </button>

                                <button
                                    onClick={() => exportVideo()}
                                    disabled={videoExporting}
                                    className="w-full px-3 py-1.5 rounded-xl text-[10px] font-medium bg-violet-500/[0.06] hover:bg-violet-500/12 border border-violet-500/15 text-violet-400 hover:text-violet-300 transition-all disabled:opacity-60 flex items-center justify-center gap-1.5"
                                    title="Assemble all segments into an MP4 video"
                                >
                                    {videoExporting
                                        ? <><Loader2 size={10} className="animate-spin" /> Assembling…</>
                                        : <><Film size={10} /> Export Video</>
                                    }
                                </button>
                            </SidebarSection>

                            {/* ── Export & Download ─────────────────── */}
                            <SidebarSection
                                icon={<Package size={10} className="text-white/30" />}
                                label="Export & Download"
                                defaultOpen={false}
                            >
                                <a
                                    href={getFullExportUrl(currentProject.id)}
                                    className="w-full px-3 py-1.5 rounded-xl text-[10px] font-medium bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.06] text-white/40 hover:text-white/70 transition-all flex items-center justify-center gap-1.5"
                                    title="Download concatenated audio as a single file"
                                >
                                    <Download size={10} /> Export Audio
                                </a>
                                <a
                                    href={getDownloadAllUrl(currentProject.id)}
                                    className="w-full px-3 py-1.5 rounded-xl text-[10px] font-medium bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.06] text-white/40 hover:text-white/70 transition-all flex items-center justify-center gap-1.5"
                                    title="Download all WAV audio, images, and videos as a ZIP"
                                >
                                    <Download size={10} /> Download All Assets
                                </a>
                                <a
                                    href={getVideoUrl(currentProject.id)}
                                    className="w-full px-3 py-1.5 rounded-xl text-[10px] font-medium bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.06] text-white/40 hover:text-white/70 transition-all flex items-center justify-center gap-1.5"
                                    title="Download the assembled MP4 video"
                                >
                                    <Download size={10} /> Download MP4
                                </a>
                            </SidebarSection>

                            {/* ── Characters & Voices ─────────────────── */}
                            <SidebarSection
                                icon={<Users size={10} className="text-violet-400/60" />}
                                label="Characters & Voices"
                                status={currentProject.characters?.length ? `${currentProject.characters.length}` : undefined}
                            >
                                <CharacterVoicePanel />
                            </SidebarSection>

                        </div>
                    )}
                </div>

                {/* ---- Main content ---- */}
                <div className="flex-1 overflow-y-auto scrollbar-thin">
                    {!currentProject ? (
                        /* Empty state */
                        <div className="flex flex-col items-center justify-center h-full text-center px-8">
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.1 }}
                                className="flex flex-col items-center"
                            >
                                <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-amber-500/10 to-orange-500/10 flex items-center justify-center mb-8 border border-amber-500/10 shadow-2xl shadow-amber-500/5">
                                    <BookOpen size={40} className="text-amber-400/60" />
                                </div>
                                <h2 className="text-3xl font-bold bg-gradient-to-r from-white/90 to-white/60 bg-clip-text text-transparent mb-3">
                                    Audiobook Generator
                                </h2>
                                <p className="text-sm text-white/30 max-w-sm mb-8 leading-relaxed">
                                    Import a book, assign character voices, and generate a full audiobook with per-segment control.
                                </p>
                                <button
                                    onClick={() => setShowNewModal(true)}
                                    className={`${btnPrimary} flex items-center gap-2.5 px-7 py-3 text-base`}
                                >
                                    <Plus size={18} /> New Project
                                </button>
                            </motion.div>
                        </div>
                    ) : (
                        /* Project view */
                        <div className="p-6 space-y-5 max-w-5xl mx-auto">
                            {/* Project header */}
                            <div className="flex items-start justify-between gap-4">
                                <div className="space-y-2">
                                    <h1 className="text-2xl font-bold bg-gradient-to-r from-white/95 to-white/70 bg-clip-text text-transparent">
                                        {currentProject.name}
                                    </h1>
                                    <div className="flex items-center gap-4 flex-wrap">
                                        <span className="inline-flex items-center gap-1.5 text-xs text-white/30">
                                            <Hash size={11} /> {currentProject.chapters.length} chapters
                                        </span>
                                        <span className="inline-flex items-center gap-1.5 text-xs text-white/30">
                                            <Layers size={11} /> {currentProject.total_segments} segments
                                        </span>
                                        <span className="inline-flex items-center gap-1.5 text-xs text-white/30">
                                            <Check size={11} /> {currentProject.done_segments} done
                                        </span>
                                        <div className="flex items-center gap-2">
                                            <div className="w-28">
                                                <ProgressBar progress={currentProject.progress} size="md" />
                                            </div>
                                            <span className="text-xs font-semibold text-amber-400 tabular-nums">
                                                {Math.round(currentProject.progress * 100)}%
                                            </span>
                                        </div>
                                        <span className={`inline-flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full border
                                            ${connected
                                                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                                                : 'bg-white/[0.03] text-white/20 border-white/[0.06]'
                                            }`}>
                                            {connected ? <Wifi size={9} /> : <WifiOff size={9} />}
                                            {connected ? 'Live' : 'Offline'}
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Stats bar */}
                            <div className={`${cardStyle} px-5 py-3 flex items-center gap-6 flex-wrap`}>
                                <div className="flex items-center gap-2">
                                    <div className="w-7 h-7 rounded-lg bg-amber-500/10 flex items-center justify-center">
                                        <Music size={12} className="text-amber-400" />
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-white/25 uppercase tracking-wider">Duration</p>
                                        <p className="text-sm font-semibold text-white/80 tabular-nums">
                                            {currentProject.total_duration > 0
                                                ? `${Math.floor(currentProject.total_duration / 60)}m ${Math.round(currentProject.total_duration % 60)}s`
                                                : '—'}
                                        </p>
                                    </div>
                                </div>

                                <div className="w-px h-8 bg-white/[0.06]" />

                                <div className="flex items-center gap-2">
                                    <div className="w-7 h-7 rounded-lg bg-violet-500/10 flex items-center justify-center">
                                        <FileText size={12} className="text-violet-400" />
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-white/25 uppercase tracking-wider">Characters</p>
                                        <p className="text-sm font-semibold text-white/80 tabular-nums">
                                            {currentProject.total_characters > 0 ? currentProject.total_characters.toLocaleString() : '—'}
                                        </p>
                                    </div>
                                </div>

                                <div className="w-px h-8 bg-white/[0.06]" />

                                <div className="flex items-center gap-2">
                                    <div className="w-7 h-7 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                                        <Check size={12} className="text-emerald-400" />
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-white/25 uppercase tracking-wider">Complete</p>
                                        <p className="text-sm font-semibold text-white/80 tabular-nums">
                                            {currentProject.done_segments}/{currentProject.total_segments}
                                        </p>
                                    </div>
                                </div>

                                {currentProject.error_segments > 0 && (
                                    <>
                                        <div className="w-px h-8 bg-white/[0.06]" />
                                        <div className="flex items-center gap-2">
                                            <div className="w-7 h-7 rounded-lg bg-red-500/10 flex items-center justify-center">
                                                <AlertCircle size={12} className="text-red-400" />
                                            </div>
                                            <div>
                                                <p className="text-[10px] text-white/25 uppercase tracking-wider">Errors</p>
                                                <p className="text-sm font-semibold text-red-400 tabular-nums">{currentProject.error_segments}</p>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => retryFailed()}
                                            className="ml-auto px-3 py-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-[11px] font-medium text-red-400 border border-red-500/15 hover:border-red-500/25 transition-all flex items-center gap-1.5"
                                        >
                                            <RefreshCw size={11} /> Retry All Failed
                                        </button>
                                    </>
                                )}

                                <div className="w-px h-8 bg-white/[0.06]" />

                                <div className="flex items-center gap-2">
                                    <div className="w-7 h-7 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                                        <Image size={12} className="text-emerald-400" />
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-white/25 uppercase tracking-wider">Visuals</p>
                                        <p className="text-sm font-semibold text-white/80 tabular-nums">
                                            {currentProject.visual_ready}/{currentProject.total_segments}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Live generation progress */}
                            <AnimatePresence>
                                {progress.isGenerating && (
                                    <motion.div
                                        initial={{ height: 0, opacity: 0 }}
                                        animate={{ height: 'auto', opacity: 1 }}
                                        exit={{ height: 0, opacity: 0 }}
                                        transition={{ duration: 0.3 }}
                                        className="overflow-hidden"
                                    >
                                        <div className={`${cardStyle} p-5 border-amber-500/15 bg-gradient-to-r from-amber-500/[0.04] to-orange-500/[0.02]`}>
                                            <div className="flex items-center justify-between mb-3">
                                                <div className="flex items-center gap-2.5">
                                                    <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse shadow-lg shadow-amber-500/50" />
                                                    <span className="text-sm font-semibold text-amber-400">Generating</span>
                                                </div>
                                                <span className="text-xs text-white/40 tabular-nums">
                                                    {progress.done}/{progress.total} ({progressPercent}%)
                                                    {progress.errors > 0 && (
                                                        <span className="text-red-400 ml-2">{progress.errors} failed</span>
                                                    )}
                                                </span>
                                            </div>
                                            <ProgressBar progress={progress.total > 0 ? progress.done / progress.total : 0} size="lg" glow />
                                            {progress.currentSegmentPreview && (
                                                <div className="mt-3 flex items-center gap-2">
                                                    <Loader2 size={11} className="animate-spin text-amber-400/60 flex-shrink-0" />
                                                    <p className="text-[11px] text-white/30 truncate italic">
                                                        "{progress.currentSegmentPreview}..."
                                                    </p>
                                                </div>
                                            )}
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            {/* Chapters */}
                            <div className="space-y-3">
                                {currentProject.chapters.map((ch: any) => (
                                    <ChapterSection
                                        key={ch.index}
                                        chapter={ch}
                                        projectId={currentProject.id}
                                        voices={availableVoices}
                                        onGenerateChapter={startGenerateChapter}
                                        generatingSegmentId={progress.currentSegmentId}
                                    />
                                ))}
                            </div>
                        </div>
                    )}
                </div>

            </div>

            <AnimatePresence>
                {showNewModal && (
                    <NewProjectModal open={showNewModal} onClose={() => setShowNewModal(false)} />
                )}
            </AnimatePresence>
        </Layout>
    );
};

export default AudiobookGenerator;
