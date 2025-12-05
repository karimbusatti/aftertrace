const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_URL = API_BASE.endsWith("/api") ? API_BASE : `${API_BASE}/api`;

export interface ProcessMetadata {
  // Basic stats
  frames_processed: number;
  total_points_spawned: number;
  average_points_per_frame: number;
  beats_detected: number;
  duration_seconds: number;
  processing_time_seconds: number;
  output_path: string;
  preset_used: string;
  
  // Surveillance stats
  max_continuous_tracking_frames: number;
  longest_track_seconds: number;
  trackability_score: number;
  people_detected: number;

  // Composition stats (v2)
  composition_mode?: boolean;
  segments_applied?: {
    effect: string;
    start_frame: number;
    end_frame: number;
  }[];
}

export interface CompositionSegment {
  effect_id: string;
  start: number;
  end: number;
}

export interface ProcessResponse {
  success: boolean;
  job_id: string;
  filename: string;
  preset: string;
  overlay_mode: boolean;
  metadata: ProcessMetadata;
  download_url: string;
  original_download_url: string | null;  // For alternating playback
}

export interface Preset {
  id: string;
  name: string;
  description: string;
}

export interface PresetsResponse {
  presets: Preset[];
}

export interface ApiError {
  detail: string;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`);
    const data = await res.json();
    return data.status === "ok";
  } catch {
    return false;
  }
}

export async function getPresets(): Promise<Preset[]> {
  try {
    const res = await fetch(`${API_URL}/presets`);
    const data: PresetsResponse = await res.json();
    return data.presets;
  } catch {
    return [];
  }
}

export async function processVideo(
  file: File,
  preset: string,
  overlayMode: boolean = false,
  composition: CompositionSegment[] | null = null
): Promise<ProcessResponse> {
  const formData = new FormData();
  formData.append("file", file);
  
  if (composition && composition.length > 0) {
    // Send composition as JSON string if using sequence mode
    formData.append("composition", JSON.stringify(composition));
    // Preset is ignored in backend but good to send fallback
    formData.append("preset", "composition");
  } else {
    formData.append("preset", preset);
  }
  
  formData.append("overlay_mode", overlayMode.toString());

  const res = await fetch(`${API_URL}/process`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const error: ApiError = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `Request failed: ${res.status}`);
  }

  return res.json();
}

/**
 * Helper to split a sequence of preset IDs into equal time segments.
 * e.g., ['a', 'b'] -> [{effect_id: 'a', start: 0, end: 0.5}, {effect_id: 'b', start: 0.5, end: 1}]
 */
export function buildComposition(sequence: string[]): CompositionSegment[] {
  if (!sequence.length) return [];
  
  const count = sequence.length;
  const durationPerSegment = 1.0 / count;
  
  return sequence.map((effectId, index) => ({
    effect_id: effectId,
    start: index * durationPerSegment,
    end: (index + 1) * durationPerSegment,
  }));
}

export function getDownloadUrl(jobId: string): string {
  return `${API_URL}/download/${jobId}`;
}

export function getOriginalUrl(jobId: string): string {
  return `${API_URL}/download/${jobId}/original`;
}
