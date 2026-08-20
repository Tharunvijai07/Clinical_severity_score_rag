import { useState } from "react";
import { FileText, ChevronDown, ChevronUp } from "lucide-react";
import clsx from "clsx";

interface ExtractedTextProps {
  text: string;
}

export default function ExtractedText({ text }: ExtractedTextProps) {
  const [expanded, setExpanded] = useState(false);

  const wordCount = text.split(/\s+/).filter(Boolean).length;
  const charCount = text.length;
  const preview = expanded ? text : text.slice(0, 600);

  return (
    <div className="glass-panel p-6 space-y-4 animate-fade-in-up delay-100" id="extracted-text-panel">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-teal-400" />
          <h2 className="text-sm font-semibold" style={{ color: "hsl(var(--text-primary))" }}>
            Extracted Clinical Text
          </h2>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-medium" style={{ color: "hsl(var(--text-muted))" }}>
          <span>{wordCount.toLocaleString()} words</span>
          <span>•</span>
          <span>{charCount.toLocaleString()} chars</span>
        </div>
      </div>

      {/* Text content */}
      <div className="glass-panel-sm p-4 relative">
        <pre
          className="text-xs leading-relaxed whitespace-pre-wrap break-words font-[inherit]"
          style={{ color: "hsl(var(--text-secondary))" }}
        >
          {preview}
          {!expanded && text.length > 600 && "…"}
        </pre>

        {/* Fade overlay when collapsed */}
        {!expanded && text.length > 600 && (
          <div
            className="absolute bottom-0 left-0 right-0 h-16 pointer-events-none rounded-b-[10px]"
            style={{
              background: "linear-gradient(transparent, hsl(var(--bg-panel) / 0.9))",
            }}
          />
        )}
      </div>

      {/* Expand toggle */}
      {text.length > 600 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className={clsx(
            "flex items-center gap-1 text-xs font-medium transition-colors",
            "text-teal-400 hover:text-teal-300",
          )}
          id="toggle-extracted-text"
        >
          {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          {expanded ? "Show less" : "Show full text"}
        </button>
      )}
    </div>
  );
}
