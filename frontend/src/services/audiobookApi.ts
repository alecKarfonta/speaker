/**
 * Audiobook API service — typed fetch wrappers for all /audiobook/* endpoints.
 */

const API_BASE = '';

// --- Types ---

export type SegmentStatus = 'pending' | 'generating' | 'done' | 'error';

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
    visual_status: string;
    animation_style: string | null;
    video_fill_mode: string;
    visual_id: string | null;
}

export interface ChapterResponse {
    index: number;
    title: string;
    segments: SegmentResponse[];
    total_segments: number;
    done_segments: number;
    progress: number;
}

export interface VisualAsset {
    id: string;
    label: string;
    scene_prompt: string | null;
    has_visual: boolean;
    visual_type: string | null;
    visual_mode: string | null;
    visual_status: string;
    animation_style: string | null;
    video_fill_mode: string;
    ref_character: string | null;
    gen_frames: number | null;
    gen_fps: number | null;
    gen_width: number | null;
    gen_height: number | null;
    gen_enable_audio: boolean;
    gen_two_stage: boolean;
    created_at: string | null;
    assigned_segments: number;
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
    visual_style?: string;
    characters?: CharacterRef[];
    narrator_voice_prompt?: string;
    visuals: VisualAsset[];
}

export interface CharacterRef {
    name: string;
    description: string;
    portrait_prompt?: string;
    voice_prompt?: string;
    visual_profile?: {
        face?: string;
        skin?: string;
        build?: string;
        clothing?: string;
    };
    portrait_path?: string;
    portrait_comfyui?: string;
    portrait_variants?: string[];
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
            narrator_voice: narratorVoice || '',
        }),
    });
    return handleResponse<ProjectDetail>(res);
}

export async function getProject(projectId: string): Promise<ProjectDetail> {
    const res = await fetch(`${API_BASE}/audiobook/projects/${projectId}`);
    return handleResponse<ProjectDetail>(res);
}

export async function deleteProject(projectId: string): Promise<void> {
    await fetch(`${API_BASE}/audiobook/projects/${projectId}`, { method: 'DELETE' });
}

export async function reparseProject(
    projectId: string,
    chapterPattern: string
): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/reparse`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chapter_pattern: chapterPattern }),
        }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function getCharacters(projectId: string) {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/characters`
    );
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
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/characters`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                character_voice_map: characterVoiceMap,
                narrator_voice: narratorVoice || '',
            }),
        }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function updateSegment(
    projectId: string,
    segmentId: string,
    update: { text?: string; voice_name?: string; scene_prompt?: string }
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
    return handleResponse<{ status: string; duration: number; segment_id: string }>(res);
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

export function getDownloadAllUrl(projectId: string): string {
    return `${API_BASE}/audiobook/projects/${projectId}/download-all`;
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

// --- Visual Generation ---

export interface VisualParams {
    mode?: string;
    frames?: number;
    fps?: number;
    width?: number;
    height?: number;
    animation?: string;
    ref_character?: string;
    enable_audio?: boolean;
    two_stage?: boolean;
}

export async function generateVisual(projectId: string, segmentId: string, params?: VisualParams): Promise<ProjectDetail> {
    const p = new URLSearchParams();
    if (params?.mode) p.set('mode', params.mode);
    if (params?.frames != null) p.set('frames', String(params.frames));
    if (params?.fps != null) p.set('fps', String(params.fps));
    if (params?.width != null) p.set('width', String(params.width));
    if (params?.height != null) p.set('height', String(params.height));
    if (params?.animation) p.set('animation', params.animation);
    if (params?.ref_character) p.set('ref_character', params.ref_character);
    const qs = p.toString() ? `?${p.toString()}` : '';
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/segments/${segmentId}/generate-visual${qs}`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function generateScenePrompt(projectId: string, segmentId: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/segments/${segmentId}/generate-scene-prompt`,
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

// --- Character Profile ---

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

export async function generatePortraitVariant(projectId: string, characterName: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/characters/${encodeURIComponent(characterName)}/generate-portrait-variant`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function selectPortrait(projectId: string, characterName: string, variantIndex: number): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/characters/${encodeURIComponent(characterName)}/select-portrait?variant_index=${variantIndex}`,
        { method: 'PUT' }
    );
    return handleResponse<ProjectDetail>(res);
}

export function getPortraitUrl(projectId: string, characterName: string, variant: number = -1): string {
    return `${API_BASE}/audiobook/projects/${projectId}/characters/${encodeURIComponent(characterName)}/portrait${variant >= 0 ? `?variant=${variant}` : ''}`;
}

export function getVoicePreviewUrl(projectId: string, characterName: string): string {
    return `${API_BASE}/audiobook/projects/${projectId}/characters/${encodeURIComponent(characterName)}/preview-voice`;
}

export async function updateCharacter(
    projectId: string,
    characterName: string,
    fields: { description?: string; portrait_prompt?: string; voice_prompt?: string }
): Promise<ProjectDetail> {
    const params = new URLSearchParams();
    if (fields.description !== undefined) params.set('description', fields.description);
    if (fields.portrait_prompt !== undefined) params.set('portrait_prompt', fields.portrait_prompt);
    if (fields.voice_prompt !== undefined) params.set('voice_prompt', fields.voice_prompt);
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/characters/${encodeURIComponent(characterName)}?${params.toString()}`,
        { method: 'PUT' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function designVoice(projectId: string, characterName: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/characters/${encodeURIComponent(characterName)}/design-voice`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

// --- Narrator Voice ---

export async function generateNarratorVoice(projectId: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/generate-narrator-voice`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function updateNarratorVoicePrompt(projectId: string, narratorVoicePrompt: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/narrator-voice-prompt`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ narrator_voice_prompt: narratorVoicePrompt }),
        }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function designNarratorVoice(projectId: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/design-narrator-voice`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

// --- Queue Status ---

export interface QueueStatus {
    tts: { queued: number; active: { segment_id: string; project_id: string } | null };
    prompt: { queued: number; active: { segment_id: string; project_id: string } | null };
    visual: { queued: number; active: { segment_id: string; project_id: string } | null };
    recent: Array<{ type: string; segment_id: string; status: string; error?: string }>;
}

export async function getQueueStatus(): Promise<QueueStatus> {
    const res = await fetch(`${API_BASE}/audiobook/queue-status`);
    return handleResponse<QueueStatus>(res);
}

export interface ExportStatus {
    status: 'idle' | 'generating' | 'done' | 'error';
    video_ready: boolean;
    error: string | null;
}

export async function getExportStatus(projectId: string): Promise<ExportStatus> {
    const res = await fetch(`${API_BASE}/audiobook/projects/${projectId}/export-status`);
    return handleResponse<ExportStatus>(res);
}

export async function setVideoFillMode(
    projectId: string,
    segmentId: string,
    mode: 'loop' | 'hold' | 'fade'
): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/segments/${segmentId}/video-fill-mode?mode=${mode}`,
        { method: 'PATCH' }
    );
    return handleResponse<ProjectDetail>(res);
}

// --- Visual Asset CRUD ---

export async function createVisualAsset(
    projectId: string,
    label?: string,
    scenePrompt?: string
): Promise<ProjectDetail> {
    const res = await fetch(`${API_BASE}/audiobook/projects/${projectId}/visuals`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: label || '', scene_prompt: scenePrompt || null }),
    });
    return handleResponse<ProjectDetail>(res);
}

export async function updateVisualAsset(
    projectId: string,
    visualId: string,
    update: { label?: string; scene_prompt?: string; animation_style?: string; video_fill_mode?: string; ref_character?: string }
): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/visuals/${visualId}`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(update),
        }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function deleteVisualAsset(projectId: string, visualId: string): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/visuals/${visualId}`,
        { method: 'DELETE' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function generateVisualAsset(
    projectId: string,
    visualId: string,
    params?: VisualParams
): Promise<ProjectDetail> {
    const p = new URLSearchParams();
    if (params?.mode) p.set('mode', params.mode);
    if (params?.frames != null) p.set('frames', String(params.frames));
    if (params?.fps != null) p.set('fps', String(params.fps));
    if (params?.width != null) p.set('width', String(params.width));
    if (params?.height != null) p.set('height', String(params.height));
    if (params?.animation) p.set('animation', params.animation);
    if (params?.ref_character) p.set('ref_character', params.ref_character);
    if (params?.enable_audio) p.set('enable_audio', 'true');
    if (params?.two_stage) p.set('two_stage', 'true');
    const qs = p.toString() ? `?${p.toString()}` : '';
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/visuals/${visualId}/generate${qs}`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function generateVisualAssetPrompt(
    projectId: string,
    visualId: string
): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/visuals/${visualId}/generate-prompt`,
        { method: 'POST' }
    );
    return handleResponse<ProjectDetail>(res);
}

export async function assignVisual(
    projectId: string,
    segmentId: string,
    visualId: string | null
): Promise<ProjectDetail> {
    const res = await fetch(
        `${API_BASE}/audiobook/projects/${projectId}/segments/${segmentId}/assign-visual`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ visual_id: visualId }),
        }
    );
    return handleResponse<ProjectDetail>(res);
}

export function getVisualAssetFileUrl(projectId: string, visualId: string): string {
    return `${API_BASE}/audiobook/projects/${projectId}/visuals/${visualId}/file`;
}
