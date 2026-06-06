export interface SfxRequest {
  prompt: string;
  seconds?: number;
  num_inference_steps?: number;
  cfg_scale?: number;
  sigma_shift?: number;
  seed?: number;
}

export interface SfxResult {
  blob: Blob;
  duration: number;
  genTime: number;
  sampleRate: number;
}

const API_BASE = '';

async function readWavBlob(response: Response): Promise<Blob> {
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength < 12) {
    throw new Error('Empty audio response from server');
  }
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== 'RIFF') {
    const snippet = new TextDecoder().decode(
      new Uint8Array(buffer, 0, Math.min(256, buffer.byteLength))
    );
    throw new Error(
      `Server did not return WAV audio (got ${JSON.stringify(magic)}): ${snippet.slice(0, 100)}`
    );
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

export function ensureAudioWavBlob(blob: Blob): Blob {
  if (blob.type === 'audio/wav' || blob.type === 'audio/x-wav') {
    return blob;
  }
  return new Blob([blob], { type: 'audio/wav' });
}

export async function generateSfx(req: SfxRequest): Promise<SfxResult> {
  const response = await fetch(`${API_BASE}/sfx`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

  if (!response.ok) {
    let detail = `SFX generation failed (${response.status})`;
    try {
      const err = await response.json();
      detail = err.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }

  const blob = await readWavBlob(response);
  return {
    blob,
    duration: parseFloat(response.headers.get('X-Audio-Duration') || '0'),
    genTime: parseFloat(response.headers.get('X-Generation-Time') || '0'),
    sampleRate: parseInt(response.headers.get('X-Sample-Rate') || '48000', 10),
  };
}

export async function checkSfxHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/sfx`, { method: 'OPTIONS' });
    return res.ok || res.status === 405;
  } catch {
    return false;
  }
}
