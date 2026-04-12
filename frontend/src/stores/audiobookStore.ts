/**
 * Zustand store for audiobook generator state.
 */
import { create } from 'zustand';
import * as api from '../services/audiobookApi';
import type { ProjectDetail, ProjectSummary, VisualParams, QueueStatus } from '../services/audiobookApi';
import type { VisualProgress } from '../hooks/useGenerationSocket';

interface AudiobookState {
    // Data
    projects: ProjectSummary[];
    currentProject: ProjectDetail | null;
    availableVoices: string[];

    // UI state
    loading: boolean;
    analyzing: boolean; // AI analysis in progress
    extractingCharacters: boolean; // character extraction / portrait generation in progress
    generating: Set<string>; // segment IDs currently generating
    playingSegmentId: string | null;
    visualMode: 'image' | 'video' | 'scene_image' | 'ref_video' | 'scene_video';
    visualSettings: { frames: number; fps: number; width: number; height: number };
    error: string | null;
    queueStatus: QueueStatus | null;
    videoExporting: boolean;
    visualProgress: VisualProgress | null;

    // Actions
    fetchProjects: () => Promise<void>;
    fetchVoices: () => Promise<void>;
    loadProject: (projectId: string) => Promise<void>;
    createProject: (name: string, text: string, chapterPattern?: string) => Promise<void>;
    deleteProject: (projectId: string) => Promise<void>;
    updateCharacterMap: (map: Record<string, string>, narratorVoice?: string) => Promise<void>;
    updateSegment: (segmentId: string, update: { text?: string; voice_name?: string; scene_prompt?: string }) => Promise<void>;
    generateSegment: (segmentId: string) => Promise<void>;
    generateChapter: (chapterIdx: number) => Promise<void>;
    generateAll: () => Promise<void>;
    analyzeCharacters: () => Promise<void>;
    extractCharacters: () => Promise<void>;
    generatePortraits: () => Promise<void>;
    importProject: (file: File) => Promise<void>;
    splitSegment: (segmentId: string, splitAt?: number) => Promise<void>;
    mergeSegment: (segmentId: string) => Promise<void>;
    retryFailed: () => Promise<void>;
    generateScenePrompt: (segmentId: string) => Promise<void>;
    generateVisual: (segmentId: string, params?: VisualParams) => Promise<void>;
    generateAllVisuals: (mode?: string) => Promise<void>;
    exportVideo: () => Promise<void>;
    // Visual asset management
    createVisualAsset: (label?: string, scenePrompt?: string) => Promise<ProjectDetail | null>;
    updateVisualAsset: (visualId: string, update: { label?: string; scene_prompt?: string; animation_style?: string; video_fill_mode?: string; ref_character?: string }) => Promise<void>;
    deleteVisualAsset: (visualId: string) => Promise<void>;
    generateVisualAsset: (visualId: string, params?: VisualParams) => Promise<void>;
    generateVisualAssetPrompt: (visualId: string) => Promise<void>;
    selectCandidate: (visualId: string, index: number) => Promise<void>;
    assignVisual: (segmentId: string, visualId: string | null) => Promise<void>;
    setPlayingSegment: (segmentId: string | null) => void;
    setCurrentProject: (project: ProjectDetail) => void;
    setVisualMode: (mode: 'image' | 'video' | 'scene_image' | 'ref_video' | 'scene_video') => void;
    setVisualSettings: (settings: Partial<{ frames: number; fps: number; width: number; height: number }>) => void;
    clearError: () => void;
    fetchQueueStatus: () => Promise<void>;
    // Called by WebSocket handler when a queued TTS job finishes
    handleSegmentComplete: (segmentId: string) => Promise<void>;
    handleSegmentError: (segmentId: string) => Promise<void>;
    // Called by WebSocket handler when a visual generation job finishes
    handleVisualComplete: (segmentId: string) => Promise<void>;
    handleVisualError: (segmentId: string) => Promise<void>;
}

export const useAudiobookStore = create<AudiobookState>((set, get) => ({
    projects: [],
    currentProject: null,
    availableVoices: [],
    loading: false,
    analyzing: false,
    extractingCharacters: false,
    generating: new Set(),
    playingSegmentId: null,
    visualMode: 'image',
    visualSettings: { frames: 25, fps: 10, width: 768, height: 512 },
    error: null,
    queueStatus: null,
    videoExporting: false,
    visualProgress: null,

    fetchProjects: async () => {
        try {
            const projects = await api.listProjects();
            set({ projects });
        } catch (e: any) {
            set({ error: e.message });
        }
    },

    fetchVoices: async () => {
        try {
            const res = await fetch('/voices');
            const data = await res.json();
            set({ availableVoices: Array.isArray(data) ? data : (data.voices || []) });
        } catch (e: any) {
            set({ error: e.message });
        }
    },

    loadProject: async (projectId) => {
        set({ loading: true, error: null });
        try {
            const project = await api.getProject(projectId);
            set({ currentProject: project, loading: false });
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    createProject: async (name, text, chapterPattern = 'auto') => {
        set({ loading: true, error: null });
        try {
            const project = await api.createProject(name, text, chapterPattern);
            set({ currentProject: project, loading: false });
            get().fetchProjects();
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    deleteProject: async (projectId) => {
        try {
            await api.deleteProject(projectId);
            const { currentProject } = get();
            if (currentProject?.id === projectId) {
                set({ currentProject: null });
            }
            get().fetchProjects();
        } catch (e: any) {
            set({ error: e.message });
        }
    },

    updateCharacterMap: async (map, narratorVoice) => {
        const { currentProject } = get();
        if (!currentProject) return;
        try {
            const updated = await api.updateCharacterMap(currentProject.id, map, narratorVoice);
            set({ currentProject: updated });
        } catch (e: any) {
            set({ error: e.message });
        }
    },

    updateSegment: async (segmentId, update) => {
        const { currentProject } = get();
        if (!currentProject) return;
        try {
            const updated = await api.updateSegment(currentProject.id, segmentId, update);
            set({ currentProject: updated });
        } catch (e: any) {
            set({ error: e.message });
        }
    },

    generateSegment: async (segmentId) => {
        const { currentProject } = get();
        if (!currentProject) return;

        // Optimistically mark as generating in the UI
        set((s) => ({ generating: new Set([...s.generating, segmentId]) }));

        try {
            // This call simply enqueues the job and returns immediately.
            // The actual completion is signalled via WebSocket tts_segment_done/tts_segment_error,
            // which calls handleSegmentComplete / handleSegmentError to clear the generating flag.
            await api.generateSegment(currentProject.id, segmentId);
        } catch (e: any) {
            // Enqueue itself failed (e.g. 404, network error) — clean up immediately
            set((s) => {
                const next = new Set(s.generating);
                next.delete(segmentId);
                return { generating: next, error: e.message };
            });
        }
    },

    handleSegmentComplete: async (segmentId) => {
        const { currentProject } = get();
        // Remove from generating set
        set((s) => {
            const next = new Set(s.generating);
            next.delete(segmentId);
            return { generating: next };
        });
        // Reload project to get updated has_audio, status, duration
        if (currentProject) {
            try {
                const updated = await api.getProject(currentProject.id);
                set({ currentProject: updated });
            } catch { }
        }
    },

    handleSegmentError: async (segmentId) => {
        const { currentProject } = get();
        // Remove from generating set
        set((s) => {
            const next = new Set(s.generating);
            next.delete(segmentId);
            return { generating: next };
        });
        // Reload project to get error state
        if (currentProject) {
            try {
                const updated = await api.getProject(currentProject.id);
                set({ currentProject: updated });
            } catch { }
        }
    },

    handleVisualComplete: async (segmentId) => {
        const { currentProject } = get();
        // Resolve the visual_id linked to this segment to clear va:${id} too
        let visualId: string | null = null;
        if (currentProject) {
            for (const ch of currentProject.chapters) {
                const seg = ch.segments.find((s: any) => s.id === segmentId);
                if (seg) { visualId = (seg as any).visual_id || null; break; }
            }
        }
        set((s) => {
            const next = new Set(s.generating);
            next.delete(segmentId);
            if (visualId) next.delete(`va:${visualId}`);
            return { generating: next };
        });
        if (currentProject) {
            try {
                const updated = await api.getProject(currentProject.id);
                set({ currentProject: updated });
            } catch { }
        }
    },

    handleVisualError: async (segmentId) => {
        const { currentProject } = get();
        let visualId: string | null = null;
        if (currentProject) {
            for (const ch of currentProject.chapters) {
                const seg = ch.segments.find((s: any) => s.id === segmentId);
                if (seg) { visualId = (seg as any).visual_id || null; break; }
            }
        }
        set((s) => {
            const next = new Set(s.generating);
            next.delete(segmentId);
            if (visualId) next.delete(`va:${visualId}`);
            return { generating: next };
        });
        if (currentProject) {
            try {
                const updated = await api.getProject(currentProject.id);
                set({ currentProject: updated });
            } catch { }
        }
    },

    generateChapter: async (chapterIdx) => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ loading: true });
        try {
            await api.generateChapter(currentProject.id, chapterIdx);
            const updated = await api.getProject(currentProject.id);
            set({ currentProject: updated, loading: false });
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    generateAll: async () => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ loading: true });
        try {
            await api.generateAll(currentProject.id);
            const updated = await api.getProject(currentProject.id);
            set({ currentProject: updated, loading: false });
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    analyzeCharacters: async () => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ analyzing: true, error: null });
        try {
            const updated = await api.analyzeCharacters(currentProject.id);
            set({ currentProject: updated, analyzing: false });
        } catch (e: any) {
            set({ error: e.message, analyzing: false });
        }
    },

    extractCharacters: async () => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ extractingCharacters: true, error: null });
        try {
            const updated = await api.extractCharacters(currentProject.id);
            set({ currentProject: updated, extractingCharacters: false });
        } catch (e: any) {
            set({ error: e.message, extractingCharacters: false });
        }
    },

    generatePortraits: async () => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ extractingCharacters: true, error: null });
        try {
            const updated = await api.generatePortraits(currentProject.id);
            set({ currentProject: updated, extractingCharacters: false });
        } catch (e: any) {
            set({ error: e.message, extractingCharacters: false });
        }
    },

    importProject: async (file: File) => {
        set({ loading: true, error: null });
        try {
            const project = await api.importProject(file);
            set({ currentProject: project, loading: false });
            // Refresh project list
            const projects = await api.listProjects();
            set({ projects });
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    splitSegment: async (segmentId: string, splitAt?: number) => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ loading: true, error: null });
        try {
            const updated = await api.splitSegment(currentProject.id, segmentId, splitAt);
            set({ currentProject: updated, loading: false });
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    mergeSegment: async (segmentId: string) => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ loading: true, error: null });
        try {
            const updated = await api.mergeSegment(currentProject.id, segmentId);
            set({ currentProject: updated, loading: false });
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    retryFailed: async () => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ loading: true, error: null });
        try {
            const updated = await api.retryFailed(currentProject.id);
            set({ currentProject: updated, loading: false });
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    generateScenePrompt: async (segmentId) => {
        const { currentProject, generating } = get();
        if (!currentProject) return;
        set((s) => ({ generating: new Set([...s.generating, `prompt:${segmentId}`]) }));
        try {
            const updated = await api.generateScenePrompt(currentProject.id, segmentId);
            set((s) => {
                const next = new Set(s.generating);
                next.delete(`prompt:${segmentId}`);
                return { currentProject: updated, generating: next };
            });
        } catch (e: any) {
            set((s) => {
                const next = new Set(s.generating);
                next.delete(`prompt:${segmentId}`);
                return { error: e.message, generating: next };
            });
        }
    },

    generateVisual: async (segmentId, params) => {
        const { currentProject, visualMode, visualSettings } = get();
        if (!currentProject) return;
        // Mark as generating — cleared by WebSocket visual_done/visual_error via handleVisualComplete/Error
        set((s) => ({ loading: false, error: null, generating: new Set([...s.generating, segmentId]) }));
        try {
            const merged: VisualParams = {
                mode: params?.mode ?? visualMode,
                frames: params?.frames ?? visualSettings.frames,
                fps: params?.fps ?? visualSettings.fps,
                width: params?.width ?? visualSettings.width,
                height: params?.height ?? visualSettings.height,
                ...params,
            };
            await api.generateVisual(currentProject.id, segmentId, merged);

            // Safety net: if the WS event never arrives (e.g. connection dropped),
            // fall back to a single poll after 45s to avoid stuck spinners.
            setTimeout(async () => {
                const { generating } = get();
                if (!generating.has(segmentId)) return; // already resolved via WS
                try {
                    const updated = await api.getProject(currentProject.id);
                    set((s) => {
                        const next = new Set(s.generating);
                        next.delete(segmentId);
                        return { currentProject: updated, generating: next };
                    });
                } catch {
                    set((s) => {
                        const next = new Set(s.generating);
                        next.delete(segmentId);
                        return { generating: next };
                    });
                }
            }, 45_000);
        } catch (e: any) {
            set((s) => {
                const next = new Set(s.generating);
                next.delete(segmentId);
                return { error: e.message, generating: next };
            });
        }
    },


    generateAllVisuals: async (mode) => {
        const { currentProject, visualMode } = get();
        if (!currentProject) return;
        set({ loading: true, error: null });
        try {
            const updated = await api.generateAllVisuals(currentProject.id, mode || visualMode);
            set({ currentProject: updated, loading: false });
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    exportVideo: async () => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ videoExporting: true, error: null });
        try {
            await api.exportVideo(currentProject.id);
            // Poll /export-status every 3s until done or error
            const poll = async () => {
                const result = await api.getExportStatus(currentProject.id);
                if (result.status === 'done' || result.video_ready) {
                    set({ videoExporting: false });
                    return;
                } else if (result.status === 'error') {
                    set({ videoExporting: false, error: result.error || 'Video assembly failed' });
                    return;
                }
                setTimeout(poll, 3000);
            };
            setTimeout(poll, 3000);
        } catch (e: any) {
            set({ error: e.message, videoExporting: false });
        }
    },

    setPlayingSegment: (segmentId) => set({ playingSegmentId: segmentId }),
    setCurrentProject: (project) => set({ currentProject: project }),
    setVisualMode: (mode) => set({ visualMode: mode }),
    setVisualSettings: (settings) => set((s) => ({ visualSettings: { ...s.visualSettings, ...settings } })),
    clearError: () => set({ error: null }),

    // --- Visual Asset Management ---
    createVisualAsset: async (label, scenePrompt) => {
        const { currentProject } = get();
        if (!currentProject) return null;
        try {
            const updated = await api.createVisualAsset(currentProject.id, label, scenePrompt);
            set({ currentProject: updated });
            return updated;
        } catch (e: any) {
            set({ error: e.message });
            return null;
        }
    },

    updateVisualAsset: async (visualId, update) => {
        const { currentProject } = get();
        if (!currentProject) return;
        try {
            const updated = await api.updateVisualAsset(currentProject.id, visualId, update);
            set({ currentProject: updated });
        } catch (e: any) {
            set({ error: e.message });
        }
    },

    deleteVisualAsset: async (visualId) => {
        const { currentProject } = get();
        if (!currentProject) return;
        try {
            const updated = await api.deleteVisualAsset(currentProject.id, visualId);
            set({ currentProject: updated });
        } catch (e: any) {
            set({ error: e.message });
        }
    },

    generateVisualAsset: async (visualId, params) => {
        const { currentProject, visualMode, visualSettings } = get();
        if (!currentProject) return;
        // Mark as generating — cleared by WebSocket visual_done/visual_error via handleVisualComplete/Error
        set((s) => ({ generating: new Set([...s.generating, `va:${visualId}`]) }));
        try {
            const merged: VisualParams = {
                mode: params?.mode ?? visualMode,
                frames: params?.frames ?? visualSettings.frames,
                fps: params?.fps ?? visualSettings.fps,
                width: params?.width ?? visualSettings.width,
                height: params?.height ?? visualSettings.height,
                ...params,
            };
            await api.generateVisualAsset(currentProject.id, visualId, merged);

            // Safety net: if the WS event never arrives, fall back after 45s
            setTimeout(async () => {
                const { generating } = get();
                if (!generating.has(`va:${visualId}`)) return; // already resolved via WS
                try {
                    const updated = await api.getProject(currentProject.id);
                    set((s) => {
                        const next = new Set(s.generating);
                        next.delete(`va:${visualId}`);
                        return { currentProject: updated, generating: next };
                    });
                } catch {
                    set((s) => {
                        const next = new Set(s.generating);
                        next.delete(`va:${visualId}`);
                        return { generating: next };
                    });
                }
            }, 45_000);
        } catch (e: any) {
            set((s) => {
                const next = new Set(s.generating);
                next.delete(`va:${visualId}`);
                return { error: e.message, generating: next };
            });
        }
    },

    generateVisualAssetPrompt: async (visualId) => {
        const { currentProject } = get();
        if (!currentProject) return;
        set((s) => ({ generating: new Set([...s.generating, `vaprompt:${visualId}`]) }));
        try {
            const updated = await api.generateVisualAssetPrompt(currentProject.id, visualId);
            set((s) => {
                const next = new Set(s.generating);
                next.delete(`vaprompt:${visualId}`);
                return { currentProject: updated, generating: next };
            });
        } catch (e: any) {
            set((s) => {
                const next = new Set(s.generating);
                next.delete(`vaprompt:${visualId}`);
                return { error: e.message, generating: next };
            });
        }
    },

    selectCandidate: async (visualId, index) => {
        const { currentProject } = get();
        if (!currentProject) return;
        try {
            const updated = await api.selectCandidate(currentProject.id, visualId, index);
            set({ currentProject: updated });
        } catch (e: any) {
            set({ error: e.message });
        }
    },

    assignVisual: async (segmentId, visualId) => {
        const { currentProject } = get();
        if (!currentProject) return;
        try {
            const updated = await api.assignVisual(currentProject.id, segmentId, visualId);
            set({ currentProject: updated });
        } catch (e: any) {
            set({ error: e.message });
        }
    },

    fetchQueueStatus: async () => {
        try {
            const status = await api.getQueueStatus();
            const prev = get().queueStatus;
            // Only update state if data actually changed to avoid unnecessary re-renders
            if (JSON.stringify(prev) !== JSON.stringify(status)) {
                set({ queueStatus: status });
            }
        } catch {
            // silently ignore — queue endpoint may not be available during startup
        }
    },
}));
