/**
 * VisualPlayer — Displays the visual asset for the currently playing segment.
 * Handles Ken Burns animations for images and video playback for videos.
 * Crossfades between segments using AnimatePresence.
 */
import React, { useState, useEffect, useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Image, Film, Maximize2 } from 'lucide-react';
import type { SegmentResponse, VisualAsset } from '../../services/audiobookApi';
import { getVisualAssetFileUrl } from '../../services/audiobookApi';
import { isBrowserVideo } from './utils';

// Ken Burns animation presets — randomly assigned per segment
const KEN_BURNS_PRESETS = [
    { from: { scale: 1.0, x: 0, y: 0 }, to: { scale: 1.15, x: -3, y: -2 } },     // slow zoom + drift left
    { from: { scale: 1.15, x: -2, y: 0 }, to: { scale: 1.0, x: 2, y: -1 } },      // zoom out + drift right
    { from: { scale: 1.0, x: 2, y: 2 }, to: { scale: 1.12, x: -2, y: -2 } },     // pan diagonally
    { from: { scale: 1.1, x: 0, y: -3 }, to: { scale: 1.0, x: 0, y: 3 } },       // tilt down
    { from: { scale: 1.0, x: -3, y: 0 }, to: { scale: 1.08, x: 3, y: 0 } },      // slow pan right
];

interface VisualPlayerProps {
    currentSegment: SegmentResponse | null;
    visuals: VisualAsset[];
    projectId: string;
    isPlaying: boolean;
    duration?: number;  // current segment duration in seconds
    onRequestPresentation?: () => void;
}

export const VisualPlayer: React.FC<VisualPlayerProps> = ({
    currentSegment,
    visuals,
    projectId,
    isPlaying,
    duration = 4,
    onRequestPresentation,
}) => {
    // Find the visual asset for the current segment
    const currentVisual = useMemo(() => {
        if (!currentSegment?.visual_id) return null;
        return visuals.find(v => v.id === currentSegment.visual_id) || null;
    }, [currentSegment?.visual_id, visuals]);

    // Deterministic Ken Burns preset based on segment id hash
    const kenBurnsPreset = useMemo(() => {
        if (!currentSegment) return KEN_BURNS_PRESETS[0];
        const hash = currentSegment.id.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
        return KEN_BURNS_PRESETS[hash % KEN_BURNS_PRESETS.length];
    }, [currentSegment?.id]);

    const hasVisual = currentVisual?.has_visual;
    const isVideo = currentVisual ? isBrowserVideo(currentVisual) : false;
    const visualUrl = currentVisual ? `${getVisualAssetFileUrl(projectId, currentVisual.id)}?t=${Date.now()}` : '';

    return (
        <div className="relative w-full aspect-video bg-gradient-to-br from-[#0a0a12] via-[#0d0d1a] to-[#0a0a12] rounded-xl overflow-hidden border border-white/[0.06]">
            <AnimatePresence mode="wait">
                {hasVisual && currentSegment ? (
                    <motion.div
                        key={`visual-${currentSegment.id}`}
                        className="absolute inset-0"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.6, ease: 'easeInOut' }}
                    >
                        {isVideo ? (
                            <video
                                key={visualUrl}
                                src={visualUrl}
                                className="w-full h-full object-cover"
                                autoPlay
                                muted
                                loop
                                playsInline
                            />
                        ) : (
                            <motion.img
                                src={visualUrl}
                                alt=""
                                className="w-full h-full object-cover"
                                initial={{
                                    scale: kenBurnsPreset.from.scale,
                                    x: `${kenBurnsPreset.from.x}%`,
                                    y: `${kenBurnsPreset.from.y}%`,
                                }}
                                animate={{
                                    scale: kenBurnsPreset.to.scale,
                                    x: `${kenBurnsPreset.to.x}%`,
                                    y: `${kenBurnsPreset.to.y}%`,
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
                        key="no-visual"
                        className="absolute inset-0 flex flex-col items-center justify-center gap-2"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <Image size={32} className="text-white/10" />
                        <p className="text-[11px] text-white/20">
                            {currentSegment ? 'No visual for this segment' : 'Select a segment to preview'}
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Presentation mode button */}
            {onRequestPresentation && (
                <button
                    onClick={onRequestPresentation}
                    className="absolute top-3 right-3 p-2 rounded-lg bg-black/40 backdrop-blur-sm text-white/50 hover:text-white hover:bg-black/60 transition-all z-10 border border-white/[0.08]"
                    title="Presentation mode"
                >
                    <Maximize2 size={14} />
                </button>
            )}

            {/* Segment info overlay */}
            {currentSegment && (
                <div className="absolute bottom-0 left-0 right-0 px-4 py-3 bg-gradient-to-t from-black/80 via-black/40 to-transparent z-10">
                    <div className="flex items-center gap-2">
                        {isVideo ? (
                            <Film size={11} className="text-violet-400 flex-shrink-0" />
                        ) : hasVisual ? (
                            <Image size={11} className="text-emerald-400 flex-shrink-0" />
                        ) : null}
                        <span className="text-[10px] text-white/40 truncate">
                            {currentSegment.character || 'Narrator'}
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default VisualPlayer;
