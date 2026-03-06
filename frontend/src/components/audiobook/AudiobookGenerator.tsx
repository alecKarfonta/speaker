import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    BookOpen, Plus, Trash2, Play, Pause, RefreshCw, Download,
    ChevronDown, ChevronRight, Edit3, Check, X, Upload, Users,
    Volume2, Mic, Zap, AlertCircle, Loader2, FileText, Music,
    Sparkles, Settings2, Square, Radio, Wifi, WifiOff, Clock,
    Hash, Headphones, Layers, Image, Film, Palette, Search,
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import Layout from '../layout/Layout';
import { useAudiobookStore } from '../../stores/audiobookStore';
import { useGenerationSocket } from '../../hooks/useGenerationSocket';
import { getSegmentAudioUrl, getSegmentVisualUrl, getChapterExportUrl, getFullExportUrl, getVideoUrl, getPortraitUrl } from '../../services/audiobookApi';
import type { SegmentResponse, ChapterResponse, CharacterRef } from '../../services/audiobookApi';

// ============================================================
// Design tokens
// ============================================================
const glass = 'backdrop-blur-xl bg-white/[0.03] border border-white/[0.06]';
const glassHover = 'hover:bg-white/[0.06] hover:border-white/[0.1]';
const cardStyle = `${glass} rounded-2xl`;
const inputStyle = 'w-full bg-white/[0.04] border border-white/[0.08] rounded-xl px-4 py-2.5 text-white/90 placeholder:text-white/20 focus:border-amber-500/40 focus:ring-1 focus:ring-amber-500/20 focus:outline-none transition-all';
const selectStyle = 'bg-white/[0.04] border border-white/[0.08] rounded-xl px-3 py-2 text-sm text-white/80 focus:border-amber-500/40 focus:outline-none transition-all';
const btnPrimary = 'px-5 py-2.5 rounded-xl text-sm font-semibold bg-gradient-to-r from-amber-500 to-orange-600 text-white shadow-lg shadow-amber-500/20 hover:shadow-amber-500/30 hover:from-amber-400 hover:to-orange-500 transition-all active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none';
const btnGhost = `px-4 py-2 rounded-xl text-sm font-medium ${glass} ${glassHover} text-white/70 hover:text-white transition-all`;

// ============================================================
// Status badge component
// ============================================================
const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
    const config: Record<string, { bg: string; text: string; icon: React.ReactNode; label: string }> = {
        pending: { bg: 'bg-slate-500/10 border-slate-500/20', text: 'text-slate-400', icon: <Clock size={10} />, label: 'Pending' },
        generating: { bg: 'bg-sky-500/10 border-sky-500/20', text: 'text-sky-400', icon: <Loader2 size={10} className="animate-spin" />, label: 'Generating' },
        done: { bg: 'bg-emerald-500/10 border-emerald-500/20', text: 'text-emerald-400', icon: <Check size={10} />, label: 'Done' },
        error: { bg: 'bg-red-500/10 border-red-500/20', text: 'text-red-400', icon: <AlertCircle size={10} />, label: 'Error' },
    };
    const c = config[status] || config.pending;
    return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border ${c.bg} ${c.text}`}>
            {c.icon} {c.label}
        </span>
    );
};

// ============================================================
// Progress bar
// ============================================================
const ProgressBar: React.FC<{ progress: number; size?: 'sm' | 'md' | 'lg'; glow?: boolean }> = ({ progress, size = 'sm', glow = false }) => (
    <div className={`w-full bg-white/[0.04] rounded-full overflow-hidden ${size === 'lg' ? 'h-3' : size === 'md' ? 'h-2' : 'h-1.5'}`}>
        <motion.div
            className={`h-full rounded-full bg-gradient-to-r from-amber-500 via-orange-500 to-rose-500 ${glow ? 'shadow-[0_0_12px_rgba(251,191,36,0.4)]' : ''}`}
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(Math.round(progress * 100), 100)}%` }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
        />
    </div>
);

// ============================================================
// Segment Row
// ============================================================
const SegmentRow: React.FC<{
    segment: SegmentResponse;
    projectId: string;
    voices: string[];
    index: number;
}> = ({ segment, projectId, voices, index }) => {
    const { generateSegment, updateSegment, splitSegment, mergeSegment, generateVisual, generating, playingSegmentId, setPlayingSegment } = useAudiobookStore();
    const [editing, setEditing] = useState(false);
    const [editText, setEditText] = useState(segment.text);
    const [editVoice, setEditVoice] = useState(segment.voice_name || '');
    const [showFullVisual, setShowFullVisual] = useState(false);
    const audioRef = useRef<HTMLAudioElement>(null);
    const isGenerating = generating.has(segment.id);
    const isPlaying = playingSegmentId === segment.id;
    const isStale = segment.status === 'pending' && !segment.has_audio;
    const hasVisual = segment.has_visual && segment.visual_status === 'done';

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
            className={`group relative rounded-xl p-3.5 transition-all duration-200
                ${isPlaying
                    ? 'bg-amber-500/[0.08] border border-amber-500/20 shadow-[0_0_20px_rgba(251,191,36,0.06)]'
                    : isGenerating
                        ? 'bg-sky-500/[0.05] border border-sky-500/15'
                        : 'bg-white/[0.015] border border-transparent hover:bg-white/[0.04] hover:border-white/[0.06]'
                }`}
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

                {/* Visual thumbnail */}
                {hasVisual && (
                    <div className="flex-shrink-0 relative">
                        <button
                            onClick={() => setShowFullVisual(!showFullVisual)}
                            className="block rounded-lg overflow-hidden border border-white/10 hover:border-emerald-500/30 transition-all hover:shadow-lg hover:shadow-emerald-500/10"
                        >
                            <img
                                src={getSegmentVisualUrl(projectId, segment.id)}
                                alt={segment.scene_prompt || 'Generated visual'}
                                className="w-16 h-16 object-cover"
                                loading="lazy"
                            />
                        </button>
                        {showFullVisual && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="absolute top-0 left-20 z-50 rounded-xl overflow-hidden border border-white/15 shadow-2xl shadow-black/60 bg-black/90"
                            >
                                <img
                                    src={getSegmentVisualUrl(projectId, segment.id)}
                                    alt={segment.scene_prompt || 'Generated visual'}
                                    className="max-w-[400px] max-h-[400px] object-contain"
                                />
                                {segment.scene_prompt && (
                                    <div className="px-3 py-2 text-[10px] text-white/50 border-t border-white/5 max-w-[400px] leading-relaxed">
                                        {segment.scene_prompt}
                                    </div>
                                )}
                            </motion.div>
                        )}
                    </div>
                )}

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
                            <p className="text-[13px] text-white/75 leading-[1.7] line-clamp-2 group-hover:line-clamp-none transition-all">
                                {segment.text}
                            </p>
                            <div className="flex items-center gap-2.5 mt-2 flex-wrap">
                                <StatusBadge status={segment.status} />
                                {isStale && (
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-500/10 text-amber-400 border border-amber-500/15">
                                        <AlertCircle size={8} /> needs re-gen
                                    </span>
                                )}
                                {segment.voice_name && (
                                    <span className="inline-flex items-center gap-1 text-[11px] text-white/35">
                                        <Mic size={9} /> {segment.voice_name}
                                    </span>
                                )}
                                {segment.character && (
                                    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[10px] bg-violet-500/10 text-violet-400/80 border border-violet-500/10">
                                        <Users size={8} /> {segment.character}
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
                                {/* Visual status indicator */}
                                {segment.visual_status === 'done' && (
                                    <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md text-[10px] bg-emerald-500/10 text-emerald-400/80 border border-emerald-500/10">
                                        <Image size={8} /> Visual
                                    </span>
                                )}
                                {segment.visual_status === 'generating' && (
                                    <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md text-[10px] bg-sky-500/10 text-sky-400/80 border border-sky-500/10">
                                        <Loader2 size={8} className="animate-spin" /> Rendering
                                    </span>
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
                    <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-all duration-200 flex-shrink-0">
                        {[
                            { icon: <Edit3 size={13} />, title: 'Edit text', onClick: () => { setEditText(segment.text); setEditVoice(segment.voice_name || ''); setEditing(true); }, color: 'hover:text-white/80' },
                            { icon: <Sparkles size={13} />, title: 'Split segment', onClick: () => splitSegment(segment.id), color: 'hover:text-sky-400' },
                            { icon: <Layers size={13} />, title: 'Merge with next', onClick: () => mergeSegment(segment.id), color: 'hover:text-violet-400' },
                            { icon: <Image size={13} />, title: 'Generate visual', onClick: () => generateVisual(segment.id), color: 'hover:text-emerald-400' },
                            { icon: isGenerating ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />, title: 'Re-generate audio', onClick: () => generateSegment(segment.id), color: 'hover:text-amber-400', disabled: isGenerating },
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
}> = ({ chapter, projectId }) => {
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
    const { generateChapter } = useAudiobookStore();
    const [expanded, setExpanded] = useState(true);
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
                    >
                        <Zap size={11} /> Generate
                    </button>
                    <a
                        href={getChapterExportUrl(projectId, chapter.index)}
                        className="px-3 py-1.5 rounded-lg bg-white/[0.03] hover:bg-white/[0.06] text-[11px] font-medium text-white/40 hover:text-white/70 border border-white/[0.06] transition-all flex items-center gap-1.5"
                    >
                        <Download size={11} /> Export
                    </a>
                </div>
            </button>

            {/* Timeline player */}
            <ChapterPlayer chapter={chapter} projectId={projectId} />

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
                            {chapter.segments.map((seg: SegmentResponse, i: number) => (
                                <div key={seg.id} className={generatingSegmentId === seg.id ? 'ring-1 ring-amber-500/30 rounded-xl' : ''}>
                                    <SegmentRow segment={seg} projectId={projectId} voices={voices} index={i} />
                                </div>
                            ))}
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
    const { currentProject, availableVoices, updateCharacterMap, analyzeCharacters, analyzing } = useAudiobookStore();
    if (!currentProject) return null;

    const { detected_characters, character_voice_map, character_descriptions, narrator_voice } = currentProject;

    const handleVoiceChange = (character: string, voice: string) => {
        const newMap = { ...character_voice_map, [character]: voice };
        updateCharacterMap(newMap, narrator_voice);
    };

    const handleNarratorChange = (voice: string) => {
        updateCharacterMap(character_voice_map, voice);
    };

    return (
        <div className="space-y-4">
            {/* AI auto-assign */}
            <button
                onClick={() => analyzeCharacters().then(() => toast.success('AI analysis complete!'))}
                disabled={analyzing}
                className="w-full px-3 py-2.5 rounded-xl text-[11px] font-semibold bg-gradient-to-r from-violet-500/15 to-purple-500/15 hover:from-violet-500/25 hover:to-purple-500/25 border border-violet-500/15 text-violet-300 hover:text-violet-200 transition-all disabled:opacity-40 flex items-center justify-center gap-2"
            >
                {analyzing ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                {analyzing ? 'Analyzing...' : '✨ Auto-Assign Voices'}
            </button>

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
            </div>

            {/* Characters */}
            {detected_characters.length > 0 && (
                <div>
                    <label className="text-[10px] font-semibold text-white/30 uppercase tracking-[0.12em] mb-2 block flex items-center gap-1.5">
                        <Users size={10} /> Characters ({detected_characters.length})
                    </label>
                    <div className="space-y-2.5">
                        {detected_characters.map((char: string) => (
                            <div key={char} className="space-y-1">
                                <div className="flex items-center gap-2">
                                    <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-violet-500/20 to-pink-500/20 flex items-center justify-center border border-violet-500/10 flex-shrink-0">
                                        <span className="text-[10px] font-bold text-violet-300">{char[0]}</span>
                                    </div>
                                    <span className="text-xs text-white/60 w-16 truncate flex-shrink-0 font-medium" title={char}>{char}</span>
                                    <select
                                        value={character_voice_map[char] || ''}
                                        onChange={(e) => handleVoiceChange(char, e.target.value)}
                                        className={`${selectStyle} flex-1 text-[11px] py-1.5`}
                                    >
                                        <option value="" className="bg-[#0f0f1a]">Narrator</option>
                                        {availableVoices.map((v: string) => (
                                            <option key={v} value={v} className="bg-[#0f0f1a]">{v}</option>
                                        ))}
                                    </select>
                                </div>
                                {character_descriptions && character_descriptions[char] && (
                                    <p className="text-[10px] text-white/20 pl-8 leading-relaxed italic">
                                        {character_descriptions[char]}
                                    </p>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

// ============================================================
// Character Portraits Panel
// ============================================================
const CharacterPortraitPanel: React.FC = () => {
    const { currentProject, extractCharacters, generatePortraits, extractingCharacters } = useAudiobookStore();
    if (!currentProject) return null;

    const characters: CharacterRef[] = currentProject.characters || [];
    const hasCharacters = characters.length > 0;
    const hasAllPortraits = hasCharacters && characters.every((c: CharacterRef) => c.portrait_path);
    const hasAnyPortrait = hasCharacters && characters.some((c: CharacterRef) => c.portrait_path);

    return (
        <div className="space-y-3">
            <label className="text-[10px] font-semibold text-white/30 uppercase tracking-[0.12em] mb-1.5 block flex items-center gap-1.5">
                <Palette size={10} /> Characters & Portraits
            </label>

            {/* Action buttons */}
            <div className="space-y-2">
                <button
                    onClick={() => extractCharacters().then(() => toast.success('Characters extracted!'))}
                    disabled={extractingCharacters}
                    className="w-full px-3 py-2 rounded-xl text-[11px] font-semibold bg-gradient-to-r from-cyan-500/15 to-blue-500/15 hover:from-cyan-500/25 hover:to-blue-500/25 border border-cyan-500/15 text-cyan-300 hover:text-cyan-200 transition-all disabled:opacity-40 flex items-center justify-center gap-2"
                >
                    {extractingCharacters ? <Loader2 size={13} className="animate-spin" /> : <Search size={13} />}
                    {extractingCharacters ? 'Extracting...' : '🔍 Extract Characters'}
                </button>

                {hasCharacters && (
                    <button
                        onClick={() => generatePortraits().then(() => toast.success('Portraits generated!'))}
                        disabled={extractingCharacters || hasAllPortraits}
                        className="w-full px-3 py-2 rounded-xl text-[11px] font-semibold bg-gradient-to-r from-amber-500/15 to-orange-500/15 hover:from-amber-500/25 hover:to-orange-500/25 border border-amber-500/15 text-amber-300 hover:text-amber-200 transition-all disabled:opacity-40 flex items-center justify-center gap-2"
                    >
                        {extractingCharacters ? <Loader2 size={13} className="animate-spin" /> : <Palette size={13} />}
                        {hasAllPortraits ? '✅ All Portraits Ready' : '🎨 Generate Portraits'}
                    </button>
                )}
            </div>

            {/* Character list with portrait thumbnails */}
            {hasCharacters && (
                <div className="space-y-2.5">
                    {characters.map((char: CharacterRef) => (
                        <div key={char.name} className={`rounded-xl p-2.5 ${glass} transition-all`}>
                            <div className="flex items-start gap-2.5">
                                {/* Portrait thumbnail */}
                                <div className="w-12 h-12 rounded-lg overflow-hidden flex-shrink-0 border border-white/[0.08] bg-white/[0.03]">
                                    {char.portrait_path ? (
                                        <img
                                            src={getPortraitUrl(currentProject.id, char.name)}
                                            alt={char.name}
                                            className="w-full h-full object-cover"
                                        />
                                    ) : (
                                        <div className="w-full h-full flex items-center justify-center">
                                            <Users size={16} className="text-white/15" />
                                        </div>
                                    )}
                                </div>

                                {/* Character info */}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-1.5">
                                        <span className="text-xs font-semibold text-white/80 truncate">{char.name}</span>
                                        {char.portrait_comfyui && (
                                            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/15">ref</span>
                                        )}
                                    </div>
                                    {char.description && (
                                        <p className="text-[10px] text-white/25 leading-relaxed mt-0.5 line-clamp-2">
                                            {char.description}
                                        </p>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {!hasCharacters && (
                <p className="text-[10px] text-white/15 text-center py-2">
                    Extract characters from book text to generate reference portraits
                </p>
            )}

            {hasAnyPortrait && (
                <p className="text-[10px] text-white/20 text-center">
                    <span className="text-emerald-400/60">✓</span> Portraits auto-used as reference in video generation
                </p>
            )}
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
        generateAllVisuals, exportVideo,
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

                    {/* Voice mapping + Character Portraits */}
                    {currentProject && (
                        <div className="p-5 flex-1 overflow-y-auto scrollbar-thin space-y-5">
                            <div>
                                <h3 className="text-[10px] font-semibold text-white/30 uppercase tracking-[0.12em] mb-3 flex items-center gap-1.5">
                                    <Mic size={10} /> Voice Mapping
                                </h3>
                                <CharacterVoicePanel />
                            </div>

                            {/* Separator */}
                            <div className="h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />

                            {/* Character Portraits */}
                            <CharacterPortraitPanel />
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

                                <div className="flex items-center gap-2 flex-shrink-0">
                                    <button
                                        onClick={() => setAutoPlay(!autoPlay)}
                                        className={`px-3 py-2 rounded-xl text-[11px] font-medium border transition-all flex items-center gap-1.5
                                            ${autoPlay
                                                ? 'bg-emerald-500/[0.08] border-emerald-500/20 text-emerald-400'
                                                : `bg-white/[0.03] border-white/[0.06] text-white/35 hover:text-white/60`
                                            }`}
                                        title="Auto-play segments as they complete"
                                    >
                                        <Volume2 size={13} />
                                        {autoPlay ? 'ON' : 'Auto-play'}
                                    </button>

                                    {progress.isGenerating ? (
                                        <button
                                            onClick={cancelGeneration}
                                            className="px-4 py-2 rounded-xl text-sm font-semibold bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 text-red-400 transition-all flex items-center gap-2"
                                        >
                                            <Square size={13} /> Cancel
                                        </button>
                                    ) : (
                                        <button
                                            onClick={startGenerateAll}
                                            disabled={!connected}
                                            className={`${btnPrimary} flex items-center gap-2`}
                                        >
                                            <Zap size={15} /> Generate All
                                        </button>
                                    )}

                                    <a
                                        href={getFullExportUrl(currentProject.id)}
                                        className={btnGhost + ' flex items-center gap-2'}
                                    >
                                        <Download size={14} /> Export Audio
                                    </a>

                                    <button
                                        onClick={() => generateAllVisuals()}
                                        disabled={loading}
                                        className="px-3 py-2 rounded-xl text-[11px] font-medium border transition-all flex items-center gap-1.5 bg-emerald-500/[0.06] border-emerald-500/15 text-emerald-400 hover:bg-emerald-500/15 hover:border-emerald-500/25 disabled:opacity-40"
                                    >
                                        <Image size={13} /> Generate Visuals
                                    </button>

                                    <button
                                        onClick={() => exportVideo()}
                                        disabled={loading || currentProject.visual_ready < currentProject.total_segments}
                                        title={currentProject.visual_ready < currentProject.total_segments ? 'Generate all visuals first' : 'Export as video'}
                                        className="px-3 py-2 rounded-xl text-[11px] font-medium border transition-all flex items-center gap-1.5 bg-violet-500/[0.06] border-violet-500/15 text-violet-400 hover:bg-violet-500/15 hover:border-violet-500/25 disabled:opacity-40"
                                    >
                                        <Film size={13} /> Export Video
                                    </button>

                                    {currentProject.visual_ready === currentProject.total_segments && currentProject.total_segments > 0 && (
                                        <a
                                            href={getVideoUrl(currentProject.id)}
                                            className={btnGhost + ' flex items-center gap-2'}
                                        >
                                            <Download size={14} /> Download MP4
                                        </a>
                                    )}
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
