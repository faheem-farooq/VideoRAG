"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  type QueryResponse,
  type SegmentResult,
  type VideoStatus,
  listVideos,
  runQuery,
  uploadVideo,
  videoFileUrl,
} from "@/lib/api";
import { formatTime } from "@/lib/format";

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "es", label: "Spanish" },
  { code: "hi", label: "Hindi" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
];

export default function Home() {
  const [videos, setVideos] = useState<VideoStatus[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string>("");
  const [uploadLanguage, setUploadLanguage] = useState("en");
  const [uploading, setUploading] = useState(false);
  const [uploadStatusText, setUploadStatusText] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [queryText, setQueryText] = useState("");
  const [topK, setTopK] = useState(5);
  const [rerank, setRerank] = useState(false);
  const [restrictToVideo, setRestrictToVideo] = useState(true);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);

  const refreshVideos = useCallback(async () => {
    try {
      const list = await listVideos();
      setVideos(list);
      return list;
    } catch {
      return [];
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    listVideos()
      .then((list) => {
        if (!cancelled) setVideos(list);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const pollUntilDone = useCallback(
    async (videoId: string) => {
      for (let attempt = 0; attempt < 120; attempt++) {
        await new Promise((r) => setTimeout(r, 1500));
        const list = await refreshVideos();
        const job = list.find((v) => v.video_id === videoId);
        if (!job) continue;
        if (job.status === "ready") {
          setUploadStatusText(`Ready — ${job.num_chunks} chunks indexed.`);
          setSelectedVideoId(videoId);
          return;
        }
        if (job.status === "failed") {
          setUploadError(job.error ?? "Ingestion failed.");
          return;
        }
        setUploadStatusText(`Status: ${job.status}…`);
      }
      setUploadError("Ingestion is taking unusually long — check the API logs.");
    },
    [refreshVideos],
  );

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    setUploadStatusText("Uploading…");
    try {
      const { video_id } = await uploadVideo(file, uploadLanguage);
      setUploadStatusText("Processing (transcribing + indexing)…");
      await pollUntilDone(video_id);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function handleQuery(e: React.FormEvent) {
    e.preventDefault();
    if (!queryText.trim()) return;
    setQueryLoading(true);
    setQueryError(null);
    try {
      const response = await runQuery({
        query: queryText,
        video_id: restrictToVideo && activeVideoId ? activeVideoId : undefined,
        top_k: topK,
        rerank,
        synthesize_answer: true,
      });
      setResult(response);
    } catch (err) {
      setQueryError(err instanceof Error ? err.message : "Query failed.");
    } finally {
      setQueryLoading(false);
    }
  }

  function seekTo(seconds: number) {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = seconds;
    el.play().catch(() => {});
  }

  function jumpToSegment(seg: SegmentResult) {
    if (seg.video_id !== activeVideoId) {
      setSelectedVideoId(seg.video_id);
      // Give the <video> element a moment to swap sources before seeking.
      setTimeout(() => seekTo(seg.start), 150);
      return;
    }
    seekTo(seg.start);
  }

  const readyVideos = videos.filter((v) => v.status === "ready");
  // Derived rather than effect-driven: falls back to the first ready video
  // until the user makes an explicit selection.
  const activeVideoId = selectedVideoId || readyVideos[0]?.video_id || "";

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-8 px-6 py-10">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">VideoRAG</h1>
        <p className="max-w-2xl text-sm text-neutral-400">
          Ask a question in any language and get back the exact timestamped segment of a
          video that answers it. Retrieval is cross-lingual at the embedding level — the
          LLM is only used to synthesize a final answer, not to translate every query.
        </p>
      </header>

      <section className="grid gap-6 md:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-3">
          <video
            ref={videoRef}
            key={activeVideoId}
            controls
            preload="auto"
            className="aspect-video w-full rounded-lg border border-neutral-800 bg-black"
            src={activeVideoId ? videoFileUrl(activeVideoId) : undefined}
          >
            <track kind="captions" />
          </video>
          {activeVideoId && (
            <p className="text-xs text-neutral-500">
              Playing:{" "}
              {videos.find((v) => v.video_id === activeVideoId)?.filename ?? activeVideoId}
            </p>
          )}
        </div>

        <div className="space-y-4 rounded-lg border border-neutral-800 p-4">
          <div>
            <label
              className="mb-1 block text-xs font-medium text-neutral-400"
              htmlFor="video-select"
            >
              Indexed videos
            </label>
            <select
              id="video-select"
              className="w-full rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm"
              value={activeVideoId}
              onChange={(e) => setSelectedVideoId(e.target.value)}
            >
              <option value="">Search across all videos</option>
              {readyVideos.map((v) => (
                <option key={v.video_id} value={v.video_id}>
                  {v.filename}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-2 border-t border-neutral-800 pt-3">
            <label className="block text-xs font-medium text-neutral-400" htmlFor="upload-input">
              Upload a new video
            </label>
            <select
              className="w-full rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm"
              value={uploadLanguage}
              onChange={(e) => setUploadLanguage(e.target.value)}
              disabled={uploading}
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  Spoken language: {l.label}
                </option>
              ))}
            </select>
            <input
              id="upload-input"
              type="file"
              accept="video/*"
              disabled={uploading}
              onChange={handleUpload}
              className="w-full text-xs text-neutral-400 file:mr-2 file:rounded file:border-0 file:bg-neutral-800 file:px-2 file:py-1 file:text-neutral-200"
            />
            {uploadStatusText && <p className="text-xs text-emerald-400">{uploadStatusText}</p>}
            {uploadError && <p className="text-xs text-red-400">{uploadError}</p>}
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <form onSubmit={handleQuery} className="flex flex-col gap-3 sm:flex-row">
          <input
            type="text"
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            placeholder="Ask a question in any language…"
            className="flex-1 rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm"
          />
          <button
            type="submit"
            disabled={queryLoading || !queryText.trim()}
            className="rounded bg-emerald-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {queryLoading ? "Searching…" : "Search"}
          </button>
        </form>

        <div className="flex flex-wrap items-center gap-4 text-xs text-neutral-400">
          <label className="flex items-center gap-1.5">
            top_k
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-14 rounded border border-neutral-700 bg-neutral-900 px-1.5 py-0.5"
            />
          </label>
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
            rerank
          </label>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={restrictToVideo}
              onChange={(e) => setRestrictToVideo(e.target.checked)}
            />
            restrict to selected video
          </label>
        </div>

        {queryError && <p className="text-sm text-red-400">{queryError}</p>}

        {result && (
          <div className="space-y-4">
            {result.answer && (
              <div className="rounded-lg border border-emerald-900 bg-emerald-950/40 p-4 text-sm">
                {result.answer}
              </div>
            )}
            <p className="text-xs text-neutral-500">
              {result.segments.length} segment(s) · {result.latency_ms.toFixed(0)}ms
              {result.cached ? " · cached" : ""}
            </p>
            <ul className="space-y-2">
              {result.segments.map((seg, i) => (
                <li key={`${seg.video_id}-${seg.start}-${i}`}>
                  <button
                    type="button"
                    onClick={() => jumpToSegment(seg)}
                    className="w-full rounded-lg border border-neutral-800 p-3 text-left text-sm hover:border-emerald-700 hover:bg-neutral-900"
                  >
                    <div className="mb-1 flex items-center justify-between text-xs text-neutral-500">
                      <span>
                        {formatTime(seg.start)}–{formatTime(seg.end)} · {seg.video_id}
                      </span>
                      <span>score {seg.score.toFixed(3)}</span>
                    </div>
                    <p className="text-neutral-200">{seg.text}</p>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </div>
  );
}
