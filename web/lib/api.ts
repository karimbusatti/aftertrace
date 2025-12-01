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
  overlayMode: boolean = false
): Promise<ProcessResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("preset", preset);
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

export function getDownloadUrl(jobId: string): string {
  return `${API_URL}/download/${jobId}`;
}

export function getOriginalUrl(jobId: string): string {
  return `${API_URL}/download/${jobId}/original`;
}
