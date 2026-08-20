import { useState } from "react";
import { BookOpen, ChevronDown, ChevronUp } from "lucide-react";
import clsx from "clsx";

import type { RetrievedChunk } from "../types";
import { formatPercent } from "../utils/format";

interface RetrievedChunksProps {
  chunks: RetrievedChunk[];
}

export default function RetrievedChunks({ chunks }: RetrievedChunksProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  if (chunks.length === 0) return null;

  return (
    <div className="glass-panel p-6 space-y-4 animate-fade-in-up delay-200" id="retrieved-chunks-panel">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-teal-400" />
          <h2 className="text-sm font-semibold" style={{ color: "hsl(var(--text-primary))" }}>
            Retrieved Medical Guidelines
          </h2>
        </div>
        <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-teal-500/15 text-teal-400 border border-teal-500/20">
          {chunks.length} chunks
        </span>
      </div>

      {/* Chunk cards */}
      <div className="space-y-3">
        {chunks.map((chunk, index) => {
          const isExpanded = expandedIndex === index;
          const similarity = Math.round(chunk.similarity * 100);
          const displayText = isExpanded ? chunk.text : chunk.text.slice(0, 220);

          return (
            <div
              key={index}
              className="glass-panel-sm p-4 space-y-3 transition-all duration-200 hover:border-[hsl(var(--border-glow)/0.25)]"
              style={{ animationDelay: `${index * 80}ms` }}
            >
              {/* Source & similarity */}
              <div className="flex items-center justify-between gap-3">
                <span
                  className="text-[11px] font-semibold uppercase tracking-wide truncate"
                  style={{ color: "hsl(var(--text-muted))" }}
                >
                  {chunk.source.replace(/\.[^.]+$/, "").replace(/[_-]/g, " ")}
                </span>
                <span
                  className={clsx(
                    "text-[11px] font-bold tabular-nums flex-shrink-0",
                    similarity >= 80
                      ? "text-emerald-400"
                      : similarity >= 60
                        ? "text-teal-400"
                        : "text-amber-400",
                  )}
                >
                  {formatPercent(chunk.similarity)} match
                </span>
              </div>

              {/* Similarity bar */}
              <div className="sim-bar">
                <div className="sim-bar-fill" style={{ width: `${similarity}%` }} />
              </div>

              {/* Text */}
              <p
                className="text-xs leading-relaxed"
                style={{ color: "hsl(var(--text-secondary))" }}
              >
                {displayText}
                {!isExpanded && chunk.text.length > 220 && "…"}
              </p>

              {/* Expand toggle */}
              {chunk.text.length > 220 && (
                <button
                  onClick={() => setExpandedIndex(isExpanded ? null : index)}
                  className="flex items-center gap-1 text-[11px] font-medium text-teal-400 hover:text-teal-300 transition-colors"
                >
                  {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  {isExpanded ? "Collapse" : "Read more"}
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
