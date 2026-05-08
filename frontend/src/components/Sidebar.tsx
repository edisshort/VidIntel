"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BrainCircuit,
  LayoutDashboard,
  Video,
  Search,
  Eye,
  Library,
  Users,
} from "lucide-react";

const nav = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/ingest", label: "Ingest", icon: Video },
  { href: "/search", label: "Ask & Search", icon: Search },
  { href: "/search?mode=consensus", label: "Consensus", icon: Users },
  { href: "/search?mode=visual", label: "Visual Search", icon: Eye },
  { href: "/library", label: "Library", icon: Library },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-56 bg-gray-950 border-r border-gray-800 flex flex-col h-screen sticky top-0">
      {/* Logo */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center flex-shrink-0">
            <BrainCircuit className="w-4.5 h-4.5 text-white" />
          </div>
          <div>
            <p className="font-bold text-white text-sm">VidIntel</p>
            <p className="text-[10px] text-gray-500">AI Knowledge Engine</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-3 space-y-1">
        {nav.map(({ href, label, icon: Icon }) => {
          const active =
            href === "/" ? pathname === "/" : pathname.startsWith(href.split("?")[0]);
          return (
            <Link
              key={label}
              href={href}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                active
                  ? "bg-indigo-600/20 text-indigo-300 font-medium"
                  : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
              }`}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800">
        <p className="text-[10px] text-gray-600">
          LLMs · Agents · Hybrid RAG · Visual RAG
        </p>
      </div>
    </aside>
  );
}
