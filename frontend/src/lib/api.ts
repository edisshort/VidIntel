/**
 * VidIntel AI — API Client
 * Typed wrappers around all backend endpoints.
 */

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface VideoIngestRequest {
  url: string;
  extract_frames?: boolean;
  frame_interval?: number;
  run_ocr?: boolean;
  collection_name?: string;
}

export interface VideoIngestResponse {
  video_id: string;
  title: string;
  duration_seconds: number;
  transcript_chunks: number;
  frames_extracted: number;
  ocr_text_chunks: number;
  collection_name: string;
  status: string;
  message: string;
}

export interface BatchIngestRequest {
  urls: string[];
  collection_name: string;
  extract_frames?: boolean;
  frame_interval?: number;
  run_ocr?: boolean;
}

export interface TimestampResult {
  video_id: string;
  video_title: string;
  timestamp_seconds: number;
  timestamp_label: string;
  text_snippet: string;
  score: number;
  source: "transcript" | "ocr";
  video_url?: string;
}

export interface QueryRequest {
  question: string;
  collection_name: string;
  mode?: "hybrid" | "semantic" | "keyword" | "visual";
  agent_mode?: "full" | "retrieval" | "consensus" | "research";
  top_k?: number;
  include_visual?: boolean;
}

export interface QueryResponse {
  answer: string;
  timestamps: TimestampResult[];
  agent_used: string;
  retrieval_mode: string;
  sources_used: number;
}

export interface CreatorOpinion {
  video_id: string;
  video_title: string;
  creator: string;
  opinion: string;
  timestamp_seconds: number | null;
  timestamp_label: string | null;
  sentiment: "positive" | "negative" | "neutral" | "mixed";
  url?: string;
}

export interface ConsensusResponse {
  question: string;
  consensus_summary: string;
  agreements: string[];
  disagreements: string[];
  creator_opinions: CreatorOpinion[];
  confidence_score: number;
}

export interface VisualResult {
  video_id: string;
  video_title: string;
  timestamp_seconds: number;
  timestamp_label: string;
  frame_path: string;
  ocr_text: string;
  score: number;
}

export interface VisualSearchResponse {
  query: string;
  results: VisualResult[];
}

export interface VideoMeta {
  video_id: string;
  title: string;
  channel: string;
  duration_seconds: number;
  url: string;
  thumbnail_url: string | null;
  collection_name: string;
  ingested_at: string;
}

export interface CollectionInfo {
  name: string;
  video_count: number;
  total_chunks: number;
  videos: VideoMeta[];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

// ─── API Methods ──────────────────────────────────────────────────────────────

export const api = {
  // Ingestion
  ingestVideo: (req: VideoIngestRequest) =>
    post<VideoIngestResponse>("/api/ingest", req),

  ingestBatch: (req: BatchIngestRequest) =>
    post<VideoIngestResponse[]>("/api/ingest/batch", req),

  // Library
  getLibrary: () => get<CollectionInfo[]>("/api/library"),

  // Query
  query: (req: QueryRequest) => post<QueryResponse>("/api/query", req),

  // Consensus
  consensus: (req: { question: string; collection_name: string; top_k?: number }) =>
    post<ConsensusResponse>("/api/consensus", req),

  // Visual search
  visualSearch: (req: { query: string; collection_name: string; top_k?: number }) =>
    post<VisualSearchResponse>("/api/visual/search", req),

  // Frame image URL
  frameImageUrl: (framePath: string) =>
    `${BASE_URL}/api/visual/frame?path=${encodeURIComponent(framePath)}`,

  // Health
  health: () => get<{ status: string; version: string; components: Record<string, string> }>("/health"),

  // YouTube timestamp link
  youtubeTimestampUrl: (videoUrl: string, seconds: number) => {
    try {
      const url = new URL(videoUrl);
      url.searchParams.set("t", String(Math.floor(seconds)));
      return url.toString();
    } catch {
      return videoUrl;
    }
  },
};
