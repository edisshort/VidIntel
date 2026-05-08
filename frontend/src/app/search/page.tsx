"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { api, CollectionInfo, QueryResponse, ConsensusResponse, VisualSearchResponse } from "@/lib/api";
import {
  Send,
  Search,
  Eye,
  Users,
  Clock,
  BrainCircuit,
  ExternalLink,
  ChevronDown,
  Loader2,
} from "lucide-react";

const MODES = [
  { id: "full", label: "Auto (AI Research)", icon: BrainCircuit, color: "indigo" },
  { id: "retrieval", label: "Timestamp Search", icon: Clock, color: "emerald" },
  { id: "consensus", label: "Consensus", icon: Users, color: "violet" },
  { id: "visual", label: "Visual / OCR", icon: Eye, color: "amber" },
] as const;

type ModeId = (typeof MODES)[number]["id"];

function TimestampBadge({ label, url, seconds }: { label: string; url: string; seconds: number }) {
  const ytUrl = api.youtubeTimestampUrl(url, seconds);
  return (
    <a
      href={ytUrl}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 bg-gray-800 hover:bg-gray-700 text-indigo-300 text-xs px-2 py-1 rounded transition-colors"
    >
      <Clock className="w-3 h-3" />
      {label}
      <ExternalLink className="w-2.5 h-2.5" />
    </a>
  );
}

function SearchContent() {
  const params = useSearchParams();
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<ModeId>((params.get("mode") as ModeId) || "full");

  const switchMode = (newMode: ModeId) => {
    setMode(newMode);
    setQuestion("");
    setResult(null);
    setError(null);
    inputRef.current?.focus();
  };
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [collection, setCollection] = useState(params.get("collection") || "");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | ConsensusResponse | VisualSearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.getLibrary().then((cols) => {
      setCollections(cols);
      if (!collection && cols.length > 0) setCollection(cols[0].name);
    });
  }, []);

  const handleSearch = async () => {
    if (!question.trim() || !collection) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      if (mode === "consensus") {
        const r = await api.consensus({ question, collection_name: collection, top_k: 8 });
        setResult(r);
      } else if (mode === "visual") {
        const r = await api.visualSearch({ query: question, collection_name: collection, top_k: 6 });
        setResult(r);
      } else {
        const r = await api.query({
          question,
          collection_name: collection,
          agent_mode: mode === "retrieval" ? "retrieval" : "full",
          top_k: 5,
          include_visual: true,
        });
        setResult(r);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const renderAnswer = () => {
    if (!result) return null;

    // Consensus
    if ("agreements" in result) {
      const r = result as ConsensusResponse;
      return (
        <div className="space-y-4">
          <div className="bg-indigo-950/50 border border-indigo-800 rounded-xl p-4">
            <p className="text-sm font-medium text-indigo-300 mb-2">Consensus Summary</p>
            <p className="text-gray-200">{r.consensus_summary}</p>
            <div className="mt-2 flex items-center gap-2">
              <div className="h-1.5 flex-1 bg-gray-800 rounded-full">
                <div
                  className="h-full bg-indigo-500 rounded-full"
                  style={{ width: `${r.confidence_score * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-400">{Math.round(r.confidence_score * 100)}% confidence</span>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-emerald-950/30 border border-emerald-900 rounded-xl p-4">
              <p className="text-xs font-medium text-emerald-400 mb-2">✓ Agreements</p>
              <ul className="space-y-1">
                {r.agreements.map((a, i) => <li key={i} className="text-gray-300 text-sm">• {a}</li>)}
              </ul>
            </div>
            <div className="bg-red-950/30 border border-red-900 rounded-xl p-4">
              <p className="text-xs font-medium text-red-400 mb-2">✗ Disagreements</p>
              <ul className="space-y-1">
                {r.disagreements.map((d, i) => <li key={i} className="text-gray-300 text-sm">• {d}</li>)}
              </ul>
            </div>
          </div>
          {r.creator_opinions.map((op) => (
            <div key={op.video_id} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-1">
                <p className="font-medium text-white text-sm">{op.creator}</p>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  op.sentiment === "positive" ? "bg-emerald-900 text-emerald-300" :
                  op.sentiment === "negative" ? "bg-red-900 text-red-300" :
                  "bg-gray-800 text-gray-400"
                }`}>{op.sentiment}</span>
              </div>
              <p className="text-gray-400 text-xs mb-2">{op.video_title}</p>
              <p className="text-gray-200 text-sm">{op.opinion}</p>
              {op.timestamp_label && (
                <div className="mt-2">
                  <TimestampBadge label={op.timestamp_label} url={op.url || ""} seconds={op.timestamp_seconds || 0} />
                </div>
              )}
            </div>
          ))}
        </div>
      );
    }

    // Visual
    if ("results" in result && "query" in result) {
      const r = result as VisualSearchResponse;
      return (
        <div className="space-y-3">
          {r.results.length === 0 && <p className="text-gray-400">No visual matches found.</p>}
          {r.results.map((res, i) => (
            <div key={i} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="flex items-center gap-3 mb-2">
                <span className="bg-amber-900/50 text-amber-300 text-xs px-2 py-0.5 rounded">
                  <Eye className="w-3 h-3 inline mr-1" />{res.timestamp_label}
                </span>
                <span className="text-gray-500 text-xs">{res.video_title}</span>
              </div>
              <p className="text-gray-300 text-sm font-mono bg-gray-950 p-3 rounded-lg">
                {res.ocr_text}
              </p>
              {res.frame_path && (
                <img
                  src={api.frameImageUrl(res.frame_path)}
                  alt={`Frame at ${res.timestamp_label}`}
                  className="mt-3 rounded-lg max-h-48 object-contain bg-black"
                  onError={(e) => (e.currentTarget.style.display = "none")}
                />
              )}
            </div>
          ))}
        </div>
      );
    }

    // Standard QueryResponse
    const r = result as QueryResponse;
    return (
      <div className="space-y-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
              {r.agent_used}
            </span>
            <span className="text-xs text-gray-600">{r.sources_used} sources used</span>
          </div>
          <div
            className="answer-content text-gray-300 text-sm leading-relaxed"
            dangerouslySetInnerHTML={{
              __html: r.answer
                .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                .replace(/\n/g, "<br/>"),
            }}
          />
        </div>
        {r.timestamps.length > 0 && (
          <div>
            <p className="text-xs text-gray-500 mb-2">Referenced moments</p>
            <div className="space-y-2">
              {r.timestamps.slice(0, 5).map((ts, i) => (
                <div key={i} className="flex items-start gap-3 bg-gray-900/60 rounded-lg p-3">
                  <TimestampBadge label={ts.timestamp_label} url={ts.video_url || ""} seconds={ts.timestamp_seconds} />
                  <div className="flex-1 min-w-0">
                    <p className="text-gray-500 text-xs mb-0.5">{ts.video_title}</p>
                    <p className="text-gray-300 text-xs truncate">{ts.text_snippet}</p>
                  </div>
                  <span className={`text-xs px-1.5 py-0.5 rounded ${
                    ts.source === "ocr" ? "bg-amber-900/50 text-amber-400" : "bg-gray-800 text-gray-500"
                  }`}>{ts.source}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">Ask & Search</h1>
        <p className="text-gray-400 text-sm">
          Query your ingested video knowledge using AI agents and Hybrid RAG.
        </p>
      </div>

      {/* Mode selector */}
      <div className="flex gap-2 mb-4 flex-wrap">
        {MODES.map((m) => {
          const Icon = m.icon;
          return (
            <button
              key={m.id}
              onClick={() => switchMode(m.id)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                mode === m.id
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-900 text-gray-400 border border-gray-800 hover:text-gray-200"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {m.label}
            </button>
          );
        })}
      </div>

      {/* Query bar */}
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-3 mb-6">
        <div className="flex gap-3 mb-3">
          <input
            ref={inputRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSearch()}
            placeholder={
              mode === "visual"
                ? "Find where folder structure is shown..."
                : mode === "consensus"
                ? "What do reviewers agree on about the battery life?"
                : mode === "retrieval"
                ? "Where does he explain dependency injection?"
                : "Ask anything about your videos..."
            }
            className="flex-1 bg-transparent text-white placeholder-gray-600 focus:outline-none text-sm"
          />
          <button
            onClick={handleSearch}
            disabled={loading || !question.trim() || !collection}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white p-2 rounded-lg transition-colors"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
        <div className="flex items-center gap-2">
          <Search className="w-3.5 h-3.5 text-gray-600" />
          <select
            value={collection}
            onChange={(e) => setCollection(e.target.value)}
            className="bg-transparent text-gray-400 text-xs focus:outline-none"
          >
            {collections.map((c) => (
              <option key={c.name} value={c.name} className="bg-gray-900">
                {c.name} ({c.video_count} videos)
              </option>
            ))}
            {collections.length === 0 && (
              <option value="" className="bg-gray-900">No collections — ingest a video first</option>
            )}
          </select>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-950 border border-red-800 rounded-xl p-4 mb-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="text-center">
            <Loader2 className="w-8 h-8 animate-spin text-indigo-400 mx-auto mb-3" />
            <p className="text-gray-400 text-sm">AI agents processing your query...</p>
          </div>
        </div>
      )}

      {/* Result */}
      {!loading && renderAnswer()}
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense fallback={<div className="p-8 text-gray-400">Loading...</div>}>
      <SearchContent />
    </Suspense>
  );
}
