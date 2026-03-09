/**
 * Shared utilities, constants, and helper functions for audiobook components.
 */

// ── Animation Styles ────────────────────────────────────────────────────
export const ANIMATION_STYLES = [
    { id: 'random', label: '🔀 Random', title: 'Randomly pick zoom in or out each time' },
    { id: 'zoom_in', label: '🔍+ Zoom In', title: 'Slow Ken Burns zoom toward center' },
    { id: 'zoom_out', label: '🔍− Zoom Out', title: 'Slow Ken Burns zoom back from center' },
    { id: 'pan_left', label: '◀ Pan', title: 'Camera drifts left to right' },
    { id: 'pan_right', label: '▶ Pan', title: 'Camera drifts right to left' },
    { id: 'pan_up', label: '⬆ Pan Up', title: 'Camera drifts upward' },
    { id: 'static', label: '☐ Static', title: 'No camera movement' },
];

// ── Resolution Presets ──────────────────────────────────────────────────
export const RESOLUTION_PRESETS = [
    { label: '512×512', w: 512, h: 512 },
    { label: '768×512', w: 768, h: 512 },
    { label: '1024×576', w: 1024, h: 576 },
    { label: '1920×1080', w: 1920, h: 1080 },
];

// ── Visual Modes ────────────────────────────────────────────────────────
export type VisualMode = 'image' | 'video' | 'scene_image' | 'ref_video' | 'scene_video';

export const VISUAL_MODES: { id: VisualMode; label: string; desc: string; icon: 'image' | 'video'; needsRef: boolean }[] = [
    { id: 'image', label: 'Image', desc: 'Text → image', icon: 'image', needsRef: false },
    { id: 'video', label: 'Video', desc: 'Text → video (LTX-2)', icon: 'video', needsRef: false },
    { id: 'scene_image', label: 'FaceID Image', desc: 'Portrait → character-consistent scene', icon: 'image', needsRef: true },
    { id: 'ref_video', label: 'Ref Video', desc: 'Image ref → guided video (LTX I2V)', icon: 'video', needsRef: true },
    { id: 'scene_video', label: 'FaceID Video', desc: 'Portrait → scene → animate (2-stage)', icon: 'video', needsRef: true },
];

// ── Time Estimation ─────────────────────────────────────────────────────
// Benchmark: ~49s for 25 frames at 768×512 on RTX 5090 ≈ 2s/frame
// Image: ~15s, Scene image (FaceID): ~25s, Scene video: ~25s + video time
export const estimateTime = (mode: string, frames: number, w: number, h: number): string => {
    if (mode === 'image') return '~15s';
    if (mode === 'scene_image') return '~25s';
    if (mode === 'scene_video') {
        const pixelFactor = (w * h) / (768 * 512);
        const secs = Math.round(25 + frames * 2.0 * pixelFactor);
        return secs < 60 ? `~${secs}s` : `~${Math.round(secs / 60)}m ${secs % 60}s`;
    }
    // video and ref_video
    const pixelFactor = (w * h) / (768 * 512);
    const secs = Math.round(frames * 2.0 * pixelFactor);
    return secs < 60 ? `~${secs}s` : `~${Math.round(secs / 60)}m ${secs % 60}s`;
};

// ── Character Color ─────────────────────────────────────────────────────
// Generate a stable color for a character name
export const getCharacterColor = (name: string | null): { hue: number; border: string; bg: string; text: string; badge: string; badgeBorder: string } => {
    if (!name) {
        // Narrator = amber
        return {
            hue: 38,
            border: 'border-l-amber-500/40',
            bg: 'bg-amber-500/[0.04]',
            text: 'text-amber-400/80',
            badge: 'bg-amber-500/10',
            badgeBorder: 'border-amber-500/15',
        };
    }
    // Hash character name to a hue (avoid yellows/ambers reserved for narrator)
    let hash = 0;
    for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
    const hue = ((Math.abs(hash) % 280) + 60) % 360; // skip 30-60 (amber range)
    return {
        hue,
        border: '',  // will use inline style
        bg: '',
        text: '',
        badge: '',
        badgeBorder: '',
    };
};

// ── Browser Video Detection ─────────────────────────────────────────────
/** Check visual_path extension to determine if browser should use <video> tag.
 * Only .mp4, .webm, .mov are true browser video formats.
 * .webp, .png, .jpg are always images, even if visual_type says 'video'. */
export const isBrowserVideo = (va: any): boolean => {
    if (!va?.visual_path) return false;
    const ext = va.visual_path.split('.').pop()?.toLowerCase() || '';
    return ['mp4', 'webm', 'mov'].includes(ext);
};
