/**
 * Custom React hook for WebSocket-based audiobook generation.
 * Connects to /audiobook/ws/{projectId} and streams real-time progress.
 * Also receives visual generation events from the backend event bus.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { useAudiobookStore } from '../stores/audiobookStore';

// Message types from server
interface ProgressInfo {
    done: number;
    errors: number;
    total: number;
}

interface WsMessage {
    type: string;
    segment_id?: string;
    duration?: number;
    error?: string;
    message?: string;
    progress?: ProgressInfo;
    total?: number;
    generated?: number;
    errors?: number;
    skipped?: number;
    remaining?: number;
    text_preview?: string;
    voice?: string;
    index?: number;
    project_id?: string;
    project_name?: string;
    // Visual progress fields
    mode?: string;
    prompt?: string;
    step?: number;
    total_steps?: number;
    visual_type?: string;
    visual_path?: string;
}

export interface GenerationProgress {
    isGenerating: boolean;
    currentSegmentId: string | null;
    currentSegmentPreview: string;
    done: number;
    errors: number;
    total: number;
    completedSegmentIds: string[];
}

const initialProgress: GenerationProgress = {
    isGenerating: false,
    currentSegmentId: null,
    currentSegmentPreview: '',
    done: 0,
    errors: 0,
    total: 0,
    completedSegmentIds: [],
};

// Visual generation progress
export interface VisualProgress {
    segmentId: string | null;
    status: 'idle' | 'prompt_generating' | 'prompt_done' | 'rendering' | 'progress' | 'done' | 'error';
    mode: string;
    step: number;
    totalSteps: number;
    error: string | null;
    prompt: string | null;
}

const initialVisualProgress: VisualProgress = {
    segmentId: null,
    status: 'idle',
    mode: '',
    step: 0,
    totalSteps: 0,
    error: null,
    prompt: null,
};

export function useGenerationSocket(projectId: string | null) {
    const wsRef = useRef<WebSocket | null>(null);
    const [connected, setConnected] = useState(false);
    const [progress, setProgress] = useState<GenerationProgress>(initialProgress);
    const [lastMessage, setLastMessage] = useState<WsMessage | null>(null);
    const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();
    const loadProject = useAudiobookStore((s) => s.loadProject);

    // Build WebSocket URL from current location
    const getWsUrl = useCallback(() => {
        if (!projectId) return null;
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${proto}//${window.location.host}/audiobook/ws/${projectId}`;
    }, [projectId]);

    // Helper to update visual progress in the store
    const setVisualProgress = useCallback((vp: VisualProgress) => {
        useAudiobookStore.setState({ visualProgress: vp });
    }, []);

    const getVisualProgress = useCallback((): VisualProgress => {
        return useAudiobookStore.getState().visualProgress || initialVisualProgress;
    }, []);

    // Connect to WebSocket
    const connect = useCallback(() => {
        const url = getWsUrl();
        if (!url) return;

        // Close existing connection
        if (wsRef.current) {
            wsRef.current.close();
        }

        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
            setConnected(true);
        };

        ws.onmessage = (event) => {
            try {
                const msg: WsMessage = JSON.parse(event.data);
                setLastMessage(msg);

                switch (msg.type) {
                    case 'connected':
                        break;

                    case 'generation_started':
                        setProgress({
                            isGenerating: true,
                            currentSegmentId: null,
                            currentSegmentPreview: '',
                            done: 0,
                            errors: 0,
                            total: msg.total || 0,
                            completedSegmentIds: [],
                        });
                        break;

                    case 'segment_start':
                        setProgress((prev) => ({
                            ...prev,
                            currentSegmentId: msg.segment_id || null,
                            currentSegmentPreview: msg.text_preview || '',
                        }));
                        break;

                    case 'segment_done':
                        setProgress((prev) => ({
                            ...prev,
                            currentSegmentId: null,
                            currentSegmentPreview: '',
                            done: msg.progress?.done ?? prev.done + 1,
                            errors: msg.progress?.errors ?? prev.errors,
                            completedSegmentIds: msg.segment_id
                                ? [...prev.completedSegmentIds, msg.segment_id]
                                : prev.completedSegmentIds,
                        }));
                        break;

                    case 'segment_error':
                        setProgress((prev) => ({
                            ...prev,
                            currentSegmentId: null,
                            currentSegmentPreview: '',
                            errors: msg.progress?.errors ?? prev.errors + 1,
                        }));
                        break;

                    case 'complete':
                        setProgress((prev) => ({
                            ...prev,
                            isGenerating: false,
                            currentSegmentId: null,
                            currentSegmentPreview: '',
                            done: msg.generated ?? prev.done,
                            errors: msg.errors ?? prev.errors,
                        }));
                        // Reload project to get final state
                        if (projectId) loadProject(projectId);
                        break;

                    case 'cancelled':
                        setProgress((prev) => ({
                            ...prev,
                            isGenerating: false,
                            currentSegmentId: null,
                            currentSegmentPreview: '',
                        }));
                        // Reload to get partial progress
                        if (projectId) loadProject(projectId);
                        break;

                    case 'error':
                        // Don't reset generation state for non-fatal errors
                        break;

                    // --- Visual generation events (relayed from backend event bus) ---

                    case 'visual_start':
                        setVisualProgress({
                            segmentId: msg.segment_id || null,
                            status: 'rendering',
                            mode: msg.mode || '',
                            step: 0,
                            totalSteps: 0,
                            error: null,
                            prompt: null,
                        });
                        break;

                    case 'visual_prompt_generating': {
                        const prev = getVisualProgress();
                        setVisualProgress({
                            ...prev,
                            segmentId: msg.segment_id || prev.segmentId,
                            status: 'prompt_generating',
                        });
                        break;
                    }

                    case 'visual_prompt_done': {
                        const prev = getVisualProgress();
                        setVisualProgress({
                            ...prev,
                            segmentId: msg.segment_id || prev.segmentId,
                            status: 'prompt_done',
                            prompt: msg.prompt || null,
                        });
                        break;
                    }

                    case 'visual_rendering': {
                        const prev = getVisualProgress();
                        setVisualProgress({
                            ...prev,
                            segmentId: msg.segment_id || prev.segmentId,
                            status: 'rendering',
                            mode: msg.mode || prev.mode,
                            step: 0,
                            totalSteps: 0,
                        });
                        break;
                    }

                    case 'visual_progress': {
                        const prev = getVisualProgress();
                        setVisualProgress({
                            ...prev,
                            segmentId: msg.segment_id || prev.segmentId,
                            status: 'progress',
                            step: msg.step || 0,
                            totalSteps: msg.total_steps || 0,
                        });
                        break;
                    }

                    case 'visual_done':
                        setVisualProgress({
                            segmentId: msg.segment_id || null,
                            status: 'done',
                            mode: msg.visual_type || '',
                            step: 0,
                            totalSteps: 0,
                            error: null,
                            prompt: null,
                        });
                        // Reload project to pick up the new visual
                        if (projectId) loadProject(projectId);
                        break;

                    case 'visual_error': {
                        const prev = getVisualProgress();
                        setVisualProgress({
                            ...prev,
                            segmentId: msg.segment_id || prev.segmentId,
                            status: 'error',
                            error: msg.error || 'Unknown error',
                        });
                        // Reload to get error state
                        if (projectId) loadProject(projectId);
                        break;
                    }
                }
            } catch (e) {
                // Ignore parse errors
            }
        };

        ws.onclose = () => {
            setConnected(false);
            // Auto-reconnect after 3 seconds if we still have a project
            if (projectId) {
                reconnectTimeoutRef.current = setTimeout(() => {
                    connect();
                }, 3000);
            }
        };

        ws.onerror = () => {
            // onclose will handle reconnect
        };
    }, [getWsUrl, projectId, loadProject, setVisualProgress, getVisualProgress]);

    // Send a message to the WebSocket
    const send = useCallback((msg: Record<string, unknown>) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(msg));
        }
    }, []);

    // Actions
    const startGenerateAll = useCallback(() => {
        send({ action: 'generate_all' });
    }, [send]);

    const startGenerateChapter = useCallback((chapterIdx: number) => {
        send({ action: 'generate_chapter', chapter_idx: chapterIdx });
    }, [send]);

    const cancelGeneration = useCallback(() => {
        send({ action: 'cancel' });
    }, [send]);

    // Connect when projectId changes
    useEffect(() => {
        if (projectId) {
            connect();
        }
        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            setConnected(false);
            setProgress(initialProgress);
        };
    }, [projectId, connect]);

    return {
        connected,
        progress,
        lastMessage,
        startGenerateAll,
        startGenerateChapter,
        cancelGeneration,
    };
}
