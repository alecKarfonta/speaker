/**
 * Audiobook API service — typed fetch wrappers for all /audiobook/* endpoints.
 */

const API_BASE = '';

// --- Types ---

export type SegmentStatus = 'pending' | 'generating' | 'done' | 'error';

export interface CharacterRef {
    name: string;
    description: string;
    portrait_path: string | null;
    portrait_comfyui: string | null;
}

export interface SegmentResponse {
    id: string;
    text: string;
    voice_name: string | null;
    character: string | null;
    emotion: string | null;
    status: SegmentStatus;
    duration: number | null;
    error_message: string | null;
    has_audio: boolean;
    scene_prompt: string | null;
    has_visual: boolean;
    visual_type: string | null;
    visual_mode: string | null;
    visual_status: string;
}

export interface ChapterResponse {
    index: number;
    title: string;
    segments: SegmentResponse[];
    total_segments: number;
    done_segments: number;
    progress: number;
}

export interface ProjectDetail {
    id: string;
    name: string;
    created_at: string;
    updated_at: string;
    chapters: ChapterResponse[];
    character_voice_map: Record<string, string>;
    narrator_voice: string;
    detected_characters: string[];
    character_descriptions: Record<string, string>;
    total_segments: number;
    done_segments: number;
    progress: number;
    // Stats
    total_duration: number;
    error_segments: number;
    total_characters: number;
    visual_ready: number;
    visual_style: string;
    characters: CharacterRef[];
}

export interface ProjectSummary {
    id: string;
    name: string;
    created_at: string;
    updated_at: string;
    total_chapters: number;
    total_segments: number;
    done_segments: number;
    progress: number;
    detected_characters: string[];
}

export interface GenerationResult {
    generated: number;
    errors: number;
    skipped: number;
}

// --- API Functions ---

async function handleResponse<T>(res: Response): Promise<T> {
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || err.error?.message || res.statusText);
    }
    return res.json();
}

export async function listProjects(): Promise<ProjectSummary[]> {
    const res = await fetch(`${API_BASE}/audiobook/projects`);
    return handleResponse<ProjectSummary[]>(res);
}

export async function createProject(
    name: string,
    text: string,
    chapterPattern: string = 'auto',
    narratorVoice?: string
): Promise<ProjectDetail> {
    const res = await fetch(`${API_BASE}/audiobook/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name,
            text,
            chapter_pattern: chapterPattern,
            narrator_voice: narratorVoice,
        }),
    });
    return handleResponse<ProjectDetail>(res);
}

export async function getProject(projectId: string): Promise<ProjectDetail> {
    const res = await fetch(`${API_BASE}/audiobook/projects/${projectId}`);
    return handleResponse<ProjectDetail>(res);
}

export async function deleteProject(projectId: string): Promise<void> {
    const res = await fetch(`${API_BASE}/audiobook/projects/${projectId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete project');
}

export async function reparseProject(
    projectId: string,
    chapterPattern: string
): Promise<ProjectDetail> {
    const res = await fetch(`${API_BASE}/audiobook/projects/${projectId}/parse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chapter_pattern: chapterPattern }),
    });
    return handleResponse<ProjectDetail>(res);
}

export async function getCharacters(projectId: string) {
    const res = await fetch(`${API_BASE}/audiobook/projects/${projectId}/characters`);
    return handleResponse<{
        detected_characters: string[];
        character_voice_map: Record<string, string>;
        narrator_voice: string;
    }>(res);
}

export async function updateCharacterMap(
    projectId: string,
    characterVoiceMap: Record<string, string>,
    narratorVoice?: string
): Promise<ProjectDetail> {
    const res = await fetch(`${API_BASE}/audiobook/projects/${projectId}/characters`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            character_voice_map: characterVoiceMap,
            narrator_voice: narratorVoice,
        }),
    });
    return handleResponse<ProjectDetail>(res);
}

export async function updateSegment(
    projectId: string,
    segmentId: string,
    update: { text?: string; voice_name?: string }
): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/segments/${segmentId}`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(update),
        }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function generateSegment(
    projectId: string,
    segmentId: string
): Promise<{ status: string; duration: number; segment_id: string }> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/segments/${segmentId}/generate`,
        { method: 'POST' }
    );
    return handleResponse(res);
}

export function getSegmentAudioUrl(projectId: string, segmentId: string): string {
    return `${API_BASE}/audiobook/projects/${projectId}/segments/${segmentId}/audio`;
}

export function getSegmentVisualUrl(projectId: string, segmentId: string): string {
    return `${API_BASE}/audiobook/projects/${projectId}/segments/${segmentId}/visual`;
}

export async function splitSegment(projectId: string, segmentId: string, splitAt?: number): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/segments/${segmentId}/split`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(splitAt !== undefined ? { split_at: splitAt } : {}),
        }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function mergeSegment(projectId: string, segmentId: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/segments/${segmentId}/merge`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function retryFailed(projectId: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/retry-failed`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function generateChapter(
    projectId: string,
    chapterIdx: number
): Promise<GenerationResult> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/chapters/${chapterIdx}/generate`,
        { method: 'POST' }
    );
    return handleResponse<GenerationResult>(res);
}

export function getChapterExportUrl(projectId: string, chapterIdx: number): string {
    return `${API_BASE}/audiobook/projects/${projectId}/chapters/${chapterIdx}/export`;
}

export async function generateAll(projectId: string): Promise<GenerationResult> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/generate-all`,
        { method: 'POST' }
    );
    return handleResponse<GenerationResult>(res);
}

export function getFullExportUrl(projectId: string): string {
    return `${API_BASE}/audiobook/projects/${projectId}/export`;
}

// --- AI Analysis ---

export async function analyzeCharacters(projectId: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/analyze`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function checkAiStatus(): Promise<{ available: boolean }> {
    const res = await fetch(`${API_BASE}/audiobook/ai-status`);
    return handleResponse<{ available: boolean }>(res);
}

// --- File Import ---

export async function importProject(file: File): Promise<ProjectDetail> {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/audiobook/projects/import`, {
        method: 'POST',
        body: form,
    });
    return handleResponse<ProjectDetail>(res);
}

// --- Character Portrait Extraction ---

export async function extractCharacters(projectId: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/extract-characters`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function generatePortraits(projectId: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/generate-portraits`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export function getPortraitUrl(projectId: string, characterName: string): string {
    return `${API_BASE}/audiobook/projects/${projectId}/characters/${encodeURIComponent(characterName)}/portrait`;
}

// --- Visual Generation ---

export async function generateVisual(projectId: string, segmentId: string, mode?: string): Promise<ProjectDetail> {
    const params = mode ? `?mode=${mode}` : '';
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/segments/${segmentId}/generate-visual${params}`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function generateAllVisuals(projectId: string, mode?: string): Promise<ProjectDetail> {
    const params = mode ? `?mode=${mode}` : '';
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/generate-visuals${params}`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function generateVisualStyle(projectId: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/generate-style`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function updateVisualStyle(projectId: string, visualStyle: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/visual-style`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ visual_style: visualStyle }),
        }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function exportVideo(projectId: string): Promise<{ status: string; video_path: string; message: string }> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/export-video`,
        { method: 'POST' }
    );
    return handleResponse<{ status: string; video_path: string; message: string }>(res);
}

export function getVideoUrl(projectId: string): string {
    return `${API_BASE}/audiobook/projects/${projectId}/video`;
}

export async function checkComfyuiHealth(): Promise<{ healthy: boolean; url: string }> {
    const res = await fetch(`${API_BASE}/audiobook/comfyui/health`);
    return handleResponse<{ healthy: boolean; url: string }>(res);
}
