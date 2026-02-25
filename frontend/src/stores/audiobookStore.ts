/**
 * Zustand store for audiobook generator state.
 */
import { create } from 'zustand';
import * as api from '../services/audiobookApi';
import type { ProjectDetail, ProjectSummary } from '../services/audiobookApi';

interface AudiobookState {
    // Data
    projects: ProjectSummary[];
    currentProject: ProjectDetail | null;
    availableVoices: string[];

    // UI state
    loading: boolean;
    analyzing: boolean; // AI analysis in progress
    generating: Set<string>; // segment IDs currently generating
    playingSegmentId: string | null;
    error: string | null;

    // Actions
    fetchProjects: () => Promise<void>;
    fetchVoices: () => Promise<void>;
    loadProject: (projectId: string) => Promise<void>;
    createProject: (name: string, text: string, chapterPattern?: string) => Promise<void>;
    deleteProject: (projectId: string) => Promise<void>;
    updateCharacterMap: (map: Record<string, string>, narratorVoice?: string) => Promise<void>;
    updateSegment: (segmentId: string, update: { text?: string; voice_name?: string }) => Promise<void>;
    generateSegment: (segmentId: string) => Promise<void>;
    generateChapter: (chapterIdx: number) => Promise<void>;
    generateAll: () => Promise<void>;
    analyzeCharacters: () => Promise<void>;
    importProject: (file: File) => Promise<void>;
    splitSegment: (segmentId: string, splitAt?: number) => Promise<void>;
    mergeSegment: (segmentId: string) => Promise<void>;
    retryFailed: () => Promise<void>;
    generateVisual: (segmentId: string, mode?: string) => Promise<void>;
    generateAllVisuals: (mode?: string) => Promise<void>;
    exportVideo: () => Promise<void>;
    setPlayingSegment: (segmentId: string | null) => void;
    clearError: () => void;
}

export const useAudiobookStore = create<AudiobookState>((set, get) => ({
    projects: [],
    currentProject: null,
    availableVoices: [],
    loading: false,
    analyzing: false,
    generating: new Set(),
    playingSegmentId: null,
    error: null,

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
            set({ availableVoices: data.voices || [] });
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
        const { currentProject, generating } = get();
        if (!currentProject) return;

        const newGenerating = new Set(generating);
        newGenerating.add(segmentId);
        set({ generating: newGenerating });

        try {
            await api.generateSegment(currentProject.id, segmentId);
            // Reload project to get updated status
            const updated = await api.getProject(currentProject.id);
            const doneGenerating = new Set(get().generating);
            doneGenerating.delete(segmentId);
            set({ currentProject: updated, generating: doneGenerating });
        } catch (e: any) {
            const doneGenerating = new Set(get().generating);
            doneGenerating.delete(segmentId);
            set({ generating: doneGenerating, error: e.message });
            // Still reload to get error status
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

    generateVisual: async (segmentId, mode) => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ loading: true, error: null });
        try {
            const updated = await api.generateVisual(currentProject.id, segmentId, mode);
            set({ currentProject: updated, loading: false });
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    generateAllVisuals: async (mode) => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ loading: true, error: null });
        try {
            const updated = await api.generateAllVisuals(currentProject.id, mode);
            set({ currentProject: updated, loading: false });
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    exportVideo: async () => {
        const { currentProject } = get();
        if (!currentProject) return;
        set({ loading: true, error: null });
        try {
            await api.exportVideo(currentProject.id);
            set({ loading: false });
        } catch (e: any) {
            set({ error: e.message, loading: false });
        }
    },

    setPlayingSegment: (segmentId) => set({ playingSegmentId: segmentId }),
    clearError: () => set({ error: null }),
}));
