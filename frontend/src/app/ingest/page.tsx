"use client";

import { useState } from "react";
import { api, VideoIngestResponse } from "@/lib/api";
import { Video, Plus, Trash2, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return h > 0 ? `${h}h ${m}m` : `${m}m ${s}s`;
}

export default function IngestPage() {
  const [mode, setMode] = useState<"single" | "batch">("single");
  const [url, setUrl] = useState("");
  const [urls, setUrls] = useState<string[]>([""]);
  const [collectionName, setCollectionName] = useState("");
  const [extractFrames, setExtractFrames] = useState(true);
  const [runOcr, setRunOcr] = useState(true);
  const [frameInterval, setFrameInterval] = useState(30);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<VideoIngestResponse[]>([]);
  const [error, setError] = useState<string | null>(null);

  const addUrl = () => setUrls([...urls, ""]);
  const removeUrl = (i: number) => setUrls(urls.filter((_, idx) => idx !== i));
  const updateUrl = (i: number, val: string) => {
    const next = [...urls];
    next[i] = val;
    setUrls(next);
  };

  const handleSingle = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.ingestVideo({
        url: url.trim(),
        extract_frames: extractFrames,
        run_ocr: runOcr,
        frame_interval: frameInterval,
        collection_name: collectionName || undefined,
      });
      setResults([result]);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBatch = async () => {
    const validUrls = urls.filter((u) => u.trim());
    if (!validUrls.length || !collectionName.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const results = await api.ingestBatch({
        urls: validUrls,
        collection_name: collectionName.trim(),
        extract_frames: extractFrames,
        run_ocr: runOcr,
        frame_interval: frameInterval,
      });
      setResults(results);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">Ingest Videos</h1>
        <p className="text-gray-400 text-sm">
          Add YouTube videos to extract transcripts, frames, and OCR data for AI analysis.
        </p>
      </div>

      {/* Mode tabs */}
      <div className="flex gap-2 mb-6 bg-gray-900 border border-gray-800 rounded-xl p-1 w-fit">
        {(["single", "batch"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors capitalize ${
              mode === m
                ? "bg-indigo-600 text-white"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            {m === "single" ? "Single Video" : "Batch / Collection"}
          </button>
        ))}
      </div>

      {/* Form */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-4">
        {mode === "single" ? (
          <div>
            <label className="block text-sm text-gray-400 mb-1.5">YouTube URL</label>
            <input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://youtube.com/watch?v=..."
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white placeholder-gray-600 focus:outline-none focus:border-indigo-500 text-sm"
            />
          </div>
        ) : (
          <div>
            <label className="block text-sm text-gray-400 mb-1.5">
              YouTube URLs (one per row)
            </label>
            <div className="space-y-2">
              {urls.map((u, i) => (
                <div key={i} className="flex gap-2">
                  <input
                    value={u}
                    onChange={(e) => updateUrl(i, e.target.value)}
                    placeholder={`Video URL ${i + 1}`}
                    className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white placeholder-gray-600 focus:outline-none focus:border-indigo-500 text-sm"
                  />
                  {urls.length > 1 && (
                    <button onClick={() => removeUrl(i)} className="text-gray-600 hover:text-red-400">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
            <button
              onClick={addUrl}
              className="mt-2 flex items-center gap-1.5 text-indigo-400 text-xs hover:text-indigo-300"
            >
              <Plus className="w-3.5 h-3.5" /> Add another URL
            </button>
          </div>
        )}

        {/* Collection name */}
        <div>
          <label className="block text-sm text-gray-400 mb-1.5">
            Collection Name {mode === "single" ? "(optional)" : "(required)"}
          </label>
          <input
            value={collectionName}
            onChange={(e) => setCollectionName(e.target.value)}
            placeholder={mode === "single" ? "Auto-generated if empty" : "e.g. macbook_reviews"}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white placeholder-gray-600 focus:outline-none focus:border-indigo-500 text-sm"
          />
        </div>

        {/* Options */}
        <div className="border-t border-gray-800 pt-4">
          <p className="text-sm text-gray-400 mb-3">Processing Options</p>
          <div className="flex flex-wrap gap-4">
            <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                checked={extractFrames}
                onChange={(e) => setExtractFrames(e.target.checked)}
                className="accent-indigo-600"
              />
              Extract frames
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                checked={runOcr}
                onChange={(e) => setRunOcr(e.target.checked)}
                className="accent-indigo-600"
              />
              Run OCR
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-300">
              Frame every
              <input
                type="number"
                value={frameInterval}
                onChange={(e) => setFrameInterval(Number(e.target.value))}
                min={5}
                max={120}
                className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-center text-sm"
              />
              seconds
            </label>
          </div>
        </div>

        {/* Submit */}
        <button
          onClick={mode === "single" ? handleSingle : handleBatch}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white py-2.5 rounded-lg font-medium transition-colors"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Processing... (this may take a few minutes)
            </>
          ) : (
            <>
              <Video className="w-4 h-4" />
              {mode === "single" ? "Ingest Video" : `Ingest ${urls.filter((u) => u.trim()).length} Videos`}
            </>
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 bg-red-950 border border-red-800 rounded-xl p-4 flex gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="mt-6 space-y-3">
          <h2 className="font-semibold text-gray-300">Ingestion Results</h2>
          {results.map((r) => (
            <div
              key={r.video_id}
              className={`bg-gray-900 border rounded-xl p-4 ${
                r.status === "success" ? "border-emerald-800" : "border-red-800"
              }`}
            >
              <div className="flex items-start gap-3">
                {r.status === "success" ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                )}
                <div className="flex-1">
                  <p className="font-medium text-white text-sm">{r.title}</p>
                  <p className="text-gray-500 text-xs mt-0.5">{r.message}</p>
                  {r.status === "success" && (
                    <div className="flex flex-wrap gap-3 mt-2">
                      {[
                        { label: "Duration", value: formatDuration(r.duration_seconds) },
                        { label: "Transcript chunks", value: r.transcript_chunks },
                        { label: "Frames", value: r.frames_extracted },
                        { label: "OCR chunks", value: r.ocr_text_chunks },
                        { label: "Collection", value: r.collection_name },
                      ].map(({ label, value }) => (
                        <span key={label} className="text-xs text-gray-400">
                          <span className="text-gray-600">{label}:</span>{" "}
                          <span className="text-gray-300">{value}</span>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
