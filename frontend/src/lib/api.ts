export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type JobStatus = "pending" | "processing" | "ready" | "failed";

export interface VideoStatus {
  video_id: string;
  filename: string;
  status: JobStatus;
  num_chunks: number;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface SegmentResult {
  video_id: string;
  start: number;
  end: number;
  text: string;
  score: number;
}

export interface QueryResponse {
  query: string;
  segments: SegmentResult[];
  answer: string | null;
  cached: boolean;
  latency_ms: number;
}

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}) as { detail?: string });
    throw new Error(body.detail ?? `Request failed with ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function listVideos(): Promise<VideoStatus[]> {
  const res = await fetch(`${API_URL}/videos`, { cache: "no-store" });
  const data = await unwrap<{ videos: VideoStatus[] }>(res);
  return data.videos;
}

export async function getVideoStatus(videoId: string): Promise<VideoStatus> {
  const res = await fetch(`${API_URL}/videos/${videoId}/status`, { cache: "no-store" });
  return unwrap<VideoStatus>(res);
}

export async function uploadVideo(
  file: File,
  language: string,
): Promise<{ video_id: string; status: JobStatus }> {
  const form = new FormData();
  form.append("file", file);
  form.append("language", language);
  const res = await fetch(`${API_URL}/videos`, { method: "POST", body: form });
  return unwrap(res);
}

export interface QueryParams {
  query: string;
  video_id?: string;
  top_k?: number;
  rerank?: boolean;
  synthesize_answer?: boolean;
}

export async function runQuery(params: QueryParams): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return unwrap<QueryResponse>(res);
}

export function videoFileUrl(videoId: string): string {
  return `${API_URL}/videos/${videoId}/file`;
}
