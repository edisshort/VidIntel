"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { api, CollectionInfo } from "@/lib/api";
import { Library, Video, FileText, Eye, Search, ChevronDown, ChevronUp } from "lucide-react";

function formatDuration(s: number) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

export default function LibraryPage() {
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    api.getLibrary().then(setCollections).catch(console.error).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center">
        <div className="text-gray-400">Loading library...</div>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Library</h1>
          <p className="text-gray-400 text-sm">
            {collections.length} collections ·{" "}
            {collections.reduce((s, c) => s + c.video_count, 0)} videos indexed
          </p>
        </div>
        <Link
          href="/ingest"
          className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
        >
          <Video className="w-4 h-4" />
          Add Videos
        </Link>
      </div>

      {collections.length === 0 ? (
        <div className="text-center py-20 border border-dashed border-gray-800 rounded-2xl">
          <Library className="w-10 h-10 text-gray-700 mx-auto mb-3" />
          <p className="text-gray-500">No videos ingested yet.</p>
          <Link href="/ingest" className="text-indigo-400 text-sm hover:underline mt-1 block">
            Ingest your first video →
          </Link>
        </div>
      ) : (
        <div className="space-y-4">
          {collections.map((col) => (
            <div key={col.name} className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
              {/* Collection header */}
              <div
                className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-800/50 transition-colors"
                onClick={() => setExpanded(expanded === col.name ? null : col.name)}
              >
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 bg-indigo-900/50 rounded-lg flex items-center justify-center">
                    <Library className="w-4.5 h-4.5 text-indigo-400" />
                  </div>
                  <div>
                    <p className="font-semibold text-white">{col.name}</p>
                    <p className="text-gray-500 text-xs">
                      {col.video_count} videos · {col.total_chunks.toLocaleString()} chunks
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Link
                    href={`/search?collection=${encodeURIComponent(col.name)}`}
                    className="flex items-center gap-1.5 text-indigo-400 text-xs hover:text-indigo-300 px-3 py-1.5 bg-indigo-950/50 rounded-lg"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Search className="w-3 h-3" />
                    Search
                  </Link>
                  {expanded === col.name ? (
                    <ChevronUp className="w-4 h-4 text-gray-500" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-gray-500" />
                  )}
                </div>
              </div>

              {/* Video list */}
              {expanded === col.name && (
                <div className="border-t border-gray-800">
                  {col.videos.map((v) => (
                    <div
                      key={v.video_id}
                      className="flex items-center gap-3 p-4 border-b border-gray-800/50 last:border-0 hover:bg-gray-800/30 transition-colors"
                    >
                      {v.thumbnail_url ? (
                        <img
                          src={v.thumbnail_url}
                          alt={v.title}
                          className="w-20 h-12 object-cover rounded-lg bg-gray-800 flex-shrink-0"
                        />
                      ) : (
                        <div className="w-20 h-12 bg-gray-800 rounded-lg flex items-center justify-center flex-shrink-0">
                          <Video className="w-5 h-5 text-gray-600" />
                        </div>
                      )}
                      <div className="flex-1 min-w-0">
                        <p className="text-white text-sm font-medium truncate">{v.title}</p>
                        <p className="text-gray-500 text-xs">{v.channel} · {formatDuration(v.duration_seconds)}</p>
                      </div>
                      <a
                        href={v.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-gray-600 hover:text-gray-300 flex-shrink-0"
                      >
                        <Eye className="w-4 h-4" />
                      </a>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
