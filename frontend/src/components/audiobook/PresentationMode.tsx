/**
 * PresentationMode — Full-screen immersive audiobook playback with visuals.
 * Shows visual + minimal floating controls. Keyboard: Space=play/pause,
 * Left/Right=prev/next, Escape=exit.
 */
import React, { useEffect, useRef, useCallback, useState, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { Play, Pause, SkipBack, SkipForward, X, Volume2, Image, Film } from 'lucide-react';
import type { ChapterResponse, SegmentResponse, VisualAsset } from '../../services/audiobookApi';
import { getSegmentAudioUrl, getVisualAssetFileUrl } from '../../services/audiobookApi';
import { isBrowserVideo } from './utils';
import { useAudiobookStore } from '../../stores/audiobookStore';

// Ken Burns presets
const KB = [
    { from: { s: 1.0, x: 0, y: 0 }, to: { s: 1.15, x: -3, y: -2 } },
    { from: { s: 1.15, x: -2, y: 0 }, to: { s: 1.0, x: 2, y: -1 } },
    { from: { s: 1.0, x: 2, y: 2 }, to: { s: 1.12, x: -2, y: -2 } },
    { from: { s: 1.1, x: 0, y: -3 }, to: { s: 1.0, x: 0, y: 3 } },
    { from: { s: 1.0, x: -3, y: 0 }, to: { s: 1.08, x: 3, y: 0 } },
];

interface PresentationModeProps {
    chapter: ChapterResponse;
    projectId: string;
    visuals: VisualAsset[];
    startSegmentIdx?: number;
    onClose: () => void;
}

const PresentationMode: React.FC<PresentationModeProps> = ({
    chapter,
    projectId,
    visuals,
    startSegmentIdx = 0,
    onClose,
}) => {
    const { setPlayingSegment } = useAudiobookStore();
    const audioRef = useRef<HTMLAudioElement>(null);
    const [playing, setPlaying] = useState(false);
    const [currentIdx, setCurrentIdx] = useState(startSegmentIdx);
    const [showControls, setShowControls] = useState(true);
    const controlsTimerRef = useRef<ReturnType<typeof setTimeout>>();
    const gapTimerRef = useRef<ReturnType<typeof setTimeout>>();

    const playableSegments = useMemo(
        () => chapter.segments.filter(s => s.has_audio),
        [chapter.segments]
    );

    const currentSeg = playableSegments[currentIdx] || null;

    // Find visual for current segment
    const currentVisual = useMemo(() => {
        if (!currentSeg?.visual_id) return null;
        return visuals.find(v => v.id === currentSeg.visual_id) || null;
    }, [currentSeg?.visual_id, visuals]);

    const isVideo = currentVisual ? isBrowserVideo(currentVisual) : false;
    const hasVisual = currentVisual?.has_visual;
    const visualUrl = currentVisual ? `${getVisualAssetFileUrl(projectId, currentVisual.id)}?t=1` : '';

    // Ken Burns preset
    const kb = useMemo(() => {
        if (!currentSeg) return KB[0];
        const hash = currentSeg.id.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
        return KB[hash % KB.length];
    }, [currentSeg?.id]);

    const duration = currentSeg?.duration || 4;

    // Play a segment
    const playAt = useCallback((idx: number) => {
        if (idx < 0 || idx >= playableSegments.length) {
            setPlaying(false);
            setPlayingSegment(null);
            setCurrentIdx(0);
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
            playAt(currentIdx);
        }
    }, [playing, currentIdx, playAt, setPlayingSegment]);

    const handleEnded = useCallback(() => {
        gapTimerRef.current = setTimeout(() => {
            const nextIdx = currentIdx + 1;
            if (nextIdx < playableSegments.length) {
                playAt(nextIdx);
            } else {
                setPlaying(false);
                setPlayingSegment(null);
                setCurrentIdx(0);
            }
        }, 400);
    }, [currentIdx, playableSegments.length, playAt, setPlayingSegment]);

    // Keyboard shortcuts
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.code === 'Escape') { e.preventDefault(); onClose(); }
            if (e.code === 'Space') { e.preventDefault(); handlePlayPause(); }
            if (e.code === 'ArrowRight') { e.preventDefault(); playAt(Math.min(currentIdx + 1, playableSegments.length - 1)); if (!playing) { setPlaying(true); } }
            if (e.code === 'ArrowLeft') { e.preventDefault(); playAt(Math.max(currentIdx - 1, 0)); if (!playing) { setPlaying(true); } }
        };
        window.addEventListener('keydown', handler);
        return () => { window.removeEventListener('keydown', handler); clearTimeout(gapTimerRef.current); };
    }, [handlePlayPause, playing, currentIdx, playableSegments.length, playAt, onClose]);

    // Auto-hide controls
    useEffect(() => {
        const handleMove = () => {
            setShowControls(true);
            clearTimeout(controlsTimerRef.current);
            controlsTimerRef.current = setTimeout(() => setShowControls(false), 3000);
        };
        window.addEventListener('mousemove', handleMove);
        handleMove();
        return () => { window.removeEventListener('mousemove', handleMove); clearTimeout(controlsTimerRef.current); };
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            setPlayingSegment(null);
            clearTimeout(gapTimerRef.current);
        };
    }, [setPlayingSegment]);

    const content = (
        <motion.div
            className="fixed inset-0 z-[9999] bg-black flex flex-col"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
        >
            <audio ref={audioRef} onEnded={handleEnded} onError={() => setPlaying(false)} />

            {/* Visual area */}
            <div className="flex-1 relative overflow-hidden flex items-center justify-center">
                <AnimatePresence mode="wait">
                    {hasVisual && currentSeg ? (
                        <motion.div
                            key={`pres-${currentSeg.id}`}
                            className="absolute inset-0"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.8, ease: 'easeInOut' }}
                        >
                            {isVideo ? (
                                <video
                                    key={visualUrl}
                                    src={visualUrl}
                                    className="w-full h-full object-contain"
                                    autoPlay muted loop playsInline
                                />
                            ) : (
                                <motion.img
                                    src={visualUrl}
                                    alt=""
                                    className="w-full h-full object-contain"
                                    initial={{
                                        scale: kb.from.s,
                                        x: `${kb.from.x}%`,
                                        y: `${kb.from.y}%`,
                                    }}
                                    animate={{
                                        scale: kb.to.s,
                                        x: `${kb.to.x}%`,
                                        y: `${kb.to.y}%`,
                                    }}
                                    transition={{
                                        duration: Math.max(duration, 3),
                                        ease: 'linear',
                                    }}
                                />
                            )}
                        </motion.div>
                    ) : (
                        <motion.div
                            key="no-vis"
                            className="flex flex-col items-center gap-3"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 0.3 }}
                            exit={{ opacity: 0 }}
                        >
                            <Image size={48} className="text-white/20" />
                            <p className="text-sm text-white/15">No visual</p>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Segment text overlay */}
                {currentSeg && (
                    <motion.div
                        key={`text-${currentSeg.id}`}
                        className="absolute bottom-24 left-0 right-0 px-12"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.4 }}
                    >
                        <div className="max-w-3xl mx-auto">
                            <p className="text-center text-white/80 text-lg leading-relaxed font-light drop-shadow-[0_2px_12px_rgba(0,0,0,0.8)]">
                                {currentSeg.text}
                            </p>
                            <p className="text-center text-white/30 text-xs mt-2 drop-shadow-[0_1px_6px_rgba(0,0,0,0.8)]">
                                {currentSeg.character || 'Narrator'}
                            </p>
                        </div>
                    </motion.div>
                )}
            </div>

            {/* Controls bar */}
            <motion.div
                className="absolute bottom-0 left-0 right-0 z-10"
                initial={{ opacity: 1 }}
                animate={{ opacity: showControls ? 1 : 0 }}
                transition={{ duration: 0.3 }}
            >
                {/* Timeline */}
                <div className="px-6 mb-2">
                    <div className="flex items-center h-1.5 rounded-full bg-white/10 overflow-hidden">
                        {playableSegments.map((seg, i) => {
                            const total = playableSegments.reduce((s, x) => s + (x.duration || 1), 0);
                            const w = ((seg.duration || 1) / total) * 100;
                            return (
                                <div
                                    key={seg.id}
                                    onClick={() => { setPlaying(true); playAt(i); }}
                                    className={`h-full cursor-pointer transition-colors ${i === currentIdx ? 'bg-amber-500' :
                                            i < currentIdx ? 'bg-amber-500/40' : 'bg-white/10'
                                        }`}
                                    style={{ width: `${w}%` }}
                                />
                            );
                        })}
                    </div>
                </div>

                {/* Buttons */}
                <div className="px-6 pb-6 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => { setPlaying(true); playAt(Math.max(currentIdx - 1, 0)); }}
                            className="p-2 rounded-full text-white/50 hover:text-white hover:bg-white/10 transition-all"
                        >
                            <SkipBack size={18} />
                        </button>
                        <button
                            onClick={handlePlayPause}
                            className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${playing
                                    ? 'bg-amber-500 text-white shadow-lg shadow-amber-500/30'
                                    : 'bg-white/15 text-white/80 hover:bg-white/25'
                                }`}
                        >
                            {playing ? <Pause size={20} /> : <Play size={20} className="ml-0.5" />}
                        </button>
                        <button
                            onClick={() => { setPlaying(true); playAt(Math.min(currentIdx + 1, playableSegments.length - 1)); }}
                            className="p-2 rounded-full text-white/50 hover:text-white hover:bg-white/10 transition-all"
                        >
                            <SkipForward size={18} />
                        </button>
                    </div>

                    <div className="flex items-center gap-4">
                        <span className="text-xs text-white/40 tabular-nums">
                            {currentIdx + 1} / {playableSegments.length}
                        </span>
                        <span className="text-xs text-white/25">
                            {chapter.title}
                        </span>
                        <button
                            onClick={onClose}
                            className="p-2 rounded-full text-white/40 hover:text-white hover:bg-white/10 transition-all"
                            title="Exit (Esc)"
                        >
                            <X size={18} />
                        </button>
                    </div>
                </div>
            </motion.div>
        </motion.div>
    );

    return createPortal(content, document.body);
};

export default PresentationMode;
