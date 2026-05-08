"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { api, CollectionInfo } from "@/lib/api";
import {
  BrainCircuit,
  Video,
  Search,
  Eye,
  Users,
  Clock,
  ArrowRight,
  Zap,
} from "lucide-react";

export default function HomePage() {
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getLibrary().then(setCollections).catch(console.error).finally(() => setLoading(false));
  }, []);

  const totalVideos = collections.reduce((s, c) => s + c.video_count, 0);
  const totalChunks = collections.reduce((s, c) => s + c.total_chunks, 0);

  const features = [
    {
      icon: <Users className="w-6 h-6 text-indigo-400" />,
      title: "Cross-Video Consensus",
      desc: "Compare creator opinions, detect agreements and contradictions across multiple videos.",
      href: "/search?mode=consensus",
    },
    {
      icon: <Clock className="w-6 h-6 text-emerald-400" />,
      title: "Exact Timestamp Retrieval",
      desc: "Find the precise moment in a tutorial where any topic is explained.",
      href: "/search?mode=retrieval",
    },
    {
      icon: <Eye className="w-6 h-6 text-violet-400" />,
      title: "Visual OCR Search",
      desc: "Search what appears on screen — code, diagrams, slides — even if never spoken.",
      href: "/search?mode=visual",
    },
    {
      icon: <BrainCircuit className="w-6 h-6 text-amber-400" />,
      title: "AI Research Agent",
      desc: "Full multi-agent synthesis. Ask any question, get a grounded intelligence report.",
      href: "/search",
    },
  ];

  return (
    <div className="min-h-screen p-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-12">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center">
            <BrainCircuit className="w-6 h-6" />
          </div>
          <h1 className="text-3xl font-bold text-white">VidIntel AI</h1>
        </div>
        <p className="text-gray-400 text-lg max-w-2xl">
          Multimodal AI intelligence engine for long-form video knowledge.
          Cross-video reasoning · Consensus analysis · Visual OCR search · Exact timestamps.
        </p>
      </div>

      {/* Stats */}
      {!loading && (
        <div className="grid grid-cols-3 gap-4 mb-10">
          {[
            { label: "Collections", value: collections.length },
            { label: "Videos indexed", value: totalVideos },
            { label: "Text chunks", value: totalChunks.toLocaleString() },
          ].map((stat) => (
            <div key={stat.label} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <p className="text-3xl font-bold text-white">{stat.value}</p>
              <p className="text-gray-500 text-sm mt-1">{stat.label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Quick actions */}
      <div className="flex gap-3 mb-10">
        <Link
          href="/ingest"
          className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2.5 rounded-lg font-medium transition-colors"
        >
          <Video className="w-4 h-4" />
          Ingest Videos
        </Link>
        <Link
          href="/search"
          className="flex items-center gap-2 bg-gray-800 hover:bg-gray-700 text-white px-4 py-2.5 rounded-lg font-medium transition-colors"
        >
          <Search className="w-4 h-4" />
          Search & Ask
        </Link>
      </div>

      {/* Features grid */}
      <h2 className="text-lg font-semibold text-gray-300 mb-4">Capabilities</h2>
      <div className="grid grid-cols-2 gap-4 mb-10">
        {features.map((f) => (
          <Link
            key={f.title}
            href={f.href}
            className="group bg-gray-900 border border-gray-800 hover:border-gray-600 rounded-xl p-5 transition-all"
          >
            <div className="flex items-start justify-between">
              <div className="mb-3">{f.icon}</div>
              <ArrowRight className="w-4 h-4 text-gray-600 group-hover:text-gray-300 transition-colors" />
            </div>
            <h3 className="font-semibold text-white mb-1">{f.title}</h3>
            <p className="text-gray-400 text-sm">{f.desc}</p>
          </Link>
        ))}
      </div>

      {/* Recent collections */}
      {collections.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-300">Recent Collections</h2>
            <Link href="/library" className="text-indigo-400 text-sm hover:underline">
              View all →
            </Link>
          </div>
          <div className="space-y-3">
            {collections.slice(0, 3).map((col) => (
              <div key={col.name} className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-center justify-between">
                <div>
                  <p className="font-medium text-white">{col.name}</p>
                  <p className="text-gray-500 text-sm">{col.video_count} videos · {col.total_chunks.toLocaleString()} chunks</p>
                </div>
                <Link
                  href={`/search?collection=${encodeURIComponent(col.name)}`}
                  className="flex items-center gap-1.5 text-indigo-400 text-sm hover:text-indigo-300"
                >
                  <Zap className="w-3.5 h-3.5" />
                  Search
                </Link>
              </div>
            ))}
          </div>
        </div>
      )}

      {totalVideos === 0 && !loading && (
        <div className="text-center py-16 border border-dashed border-gray-800 rounded-2xl">
          <Video className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-white font-semibold mb-2">No videos ingested yet</h3>
          <p className="text-gray-500 mb-6 text-sm">
            Start by ingesting a YouTube video or playlist.
          </p>
          <Link
            href="/ingest"
            className="bg-indigo-600 hover:bg-indigo-500 text-white px-5 py-2.5 rounded-lg font-medium transition-colors"
          >
            Ingest your first video
          </Link>
        </div>
      )}
    </div>
  );
}
