import { useEffect, useRef, useMemo } from "react";
import {
  AlertTriangle,
  Download,
  Shield,
  Stethoscope,
  ClipboardList,
  Quote,
} from "lucide-react";
import clsx from "clsx";

import type { SeverityResult, RetrievedChunk } from "../types";
import { severityColor, formatPercent, normalizeConfidence } from "../utils/format";

interface SeverityDashboardProps {
  result: SeverityResult;
  chunks: RetrievedChunk[];
  extractedText: string;
  rawResponse: string;
}

/* ── SVG Gauge ────────────────────────────────────────────────────── */
function SeverityGauge({ score, level }: { score: number; level: string }) {
  const gaugeRef = useRef<SVGCircleElement>(null);
  const color = severityColor(level as SeverityResult["severity_level"]);
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(score / 10, 1);
  const dashOffset = circumference * (1 - progress);

  useEffect(() => {
    const el = gaugeRef.current;
    if (!el) return;
    el.style.setProperty("--circumference", `${circumference}`);
    el.style.setProperty("--dash-offset", `${dashOffset}`);
    el.style.strokeDasharray = `${circumference}`;
    el.style.strokeDashoffset = `${circumference}`;

    // Trigger animation
    requestAnimationFrame(() => {
      el.style.animation = "gauge-draw 1.2s cubic-bezier(0.25,1,0.5,1) forwards";
    });
  }, [circumference, dashOffset]);

  return (
    <div className="relative flex items-center justify-center" id="severity-gauge">
      <svg width="180" height="180" viewBox="0 0 180 180" className="-rotate-90">
        {/* Background track */}
        <circle
          cx="90"
          cy="90"
          r={radius}
          fill="none"
          stroke="hsl(var(--border))"
          strokeWidth="10"
          strokeLinecap="round"
          opacity="0.3"
        />
        {/* Animated arc */}
        <circle
          ref={gaugeRef}
          cx="90"
          cy="90"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          className="severity-ring"
          style={{ color }}
        />
      </svg>

      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-extrabold tabular-nums" style={{ color }}>
          {score}
        </span>
        <span className="text-xs font-medium mt-1" style={{ color: "hsl(var(--text-muted))" }}>
          out of 10
        </span>
      </div>
    </div>
  );
}

/* ── Level Badge ──────────────────────────────────────────────────── */
function LevelBadge({ level }: { level: string }) {
  const color = severityColor(level as SeverityResult["severity_level"]);
  return (
    <div
      className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold border"
      style={{
        background: `${color}15`,
        borderColor: `${color}40`,
        color,
      }}
      id="severity-level-badge"
    >
      <Shield className="w-4 h-4" />
      {level}
    </div>
  );
}

/* ── Main Dashboard ────────────────────────────────────────────────── */
export default function SeverityDashboard({
  result,
  chunks,
  extractedText,
  rawResponse,
}: SeverityDashboardProps) {
  const confidence = normalizeConfidence(result.confidence);

  const exportData = useMemo(
    () =>
      JSON.stringify(
        {
          severity_score: result.severity_score,
          severity_level: result.severity_level,
          confidence: result.confidence,
          key_findings: result.key_findings,
          evidence: result.evidence,
          summary: result.summary,
          retrieved_chunks: chunks.map((c) => ({
            source: c.source,
            similarity: c.similarity,
            text: c.text,
          })),
          extracted_text: extractedText,
          raw_model_response: rawResponse,
        },
        null,
        2,
      ),
    [result, chunks, extractedText, rawResponse],
  );

  const handleDownload = () => {
    const blob = new Blob([exportData], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `medseverity-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-5 animate-fade-in-up delay-300" id="severity-dashboard">
      {/* ── Score + Level row ───────────────────────────────────────── */}
      <div className="glass-panel p-6">
        <div className="flex items-center gap-2 mb-5">
          <Stethoscope className="w-4 h-4 text-teal-400" />
          <h2 className="text-sm font-semibold" style={{ color: "hsl(var(--text-primary))" }}>
            Severity Assessment
          </h2>
        </div>

        <div className="flex flex-col sm:flex-row items-center gap-6 sm:gap-10">
          {/* Gauge */}
          <SeverityGauge score={result.severity_score} level={result.severity_level} />

          {/* Level + Confidence */}
          <div className="flex flex-col items-center sm:items-start gap-4 flex-1 min-w-0">
            <LevelBadge level={result.severity_level} />

            {/* Confidence bar */}
            <div className="w-full max-w-xs space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium" style={{ color: "hsl(var(--text-secondary))" }}>
                  Confidence
                </span>
                <span className="text-xs font-bold tabular-nums text-teal-400">
                  {formatPercent(confidence)}
                </span>
              </div>
              <div className="w-full h-2 rounded-full" style={{ background: "hsl(var(--bg-surface))" }}>
                <div
                  className="progress-fill h-full"
                  style={{ width: `${Math.round(confidence * 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Key Findings ────────────────────────────────────────────── */}
      {result.key_findings.length > 0 && (
        <div className="glass-panel p-6 space-y-3">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-semibold" style={{ color: "hsl(var(--text-primary))" }}>
              Key Findings
            </h3>
          </div>

          <ul className="space-y-2">
            {result.key_findings.map((finding, i) => (
              <li
                key={i}
                className="flex items-start gap-3 glass-panel-sm p-3 animate-slide-in"
                style={{ animationDelay: `${i * 60}ms` }}
              >
                <span
                  className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold mt-0.5"
                  style={{
                    background: "hsl(var(--accent) / 0.15)",
                    color: "hsl(var(--accent-glow))",
                    border: "1px solid hsl(var(--accent) / 0.25)",
                  }}
                >
                  {i + 1}
                </span>
                <p className="text-xs leading-relaxed" style={{ color: "hsl(var(--text-secondary))" }}>
                  {finding}
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Evidence ────────────────────────────────────────────────── */}
      {result.evidence.length > 0 && (
        <div className="glass-panel p-6 space-y-3">
          <div className="flex items-center gap-2">
            <Quote className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold" style={{ color: "hsl(var(--text-primary))" }}>
              Medical Evidence
            </h3>
          </div>

          <div className="space-y-2">
            {result.evidence.map((item, i) => (
              <div
                key={i}
                className="glass-panel-sm p-3 border-l-2 animate-slide-in"
                style={{
                  borderLeftColor: "hsl(var(--accent))",
                  animationDelay: `${i * 60}ms`,
                }}
              >
                <p className="text-xs leading-relaxed" style={{ color: "hsl(var(--text-secondary))" }}>
                  {item}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Clinical Summary ────────────────────────────────────────── */}
      <div className="glass-panel p-6 space-y-3">
        <div className="flex items-center gap-2">
          <ClipboardList className="w-4 h-4 text-teal-400" />
          <h3 className="text-sm font-semibold" style={{ color: "hsl(var(--text-primary))" }}>
            Clinical Summary
          </h3>
        </div>
        <p className="text-sm leading-relaxed" style={{ color: "hsl(var(--text-secondary))" }}>
          {result.summary}
        </p>
      </div>

      {/* ── Export ───────────────────────────────────────────────────── */}
      <button
        onClick={handleDownload}
        className={clsx(
          "flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold w-full",
          "border border-teal-500/30 text-teal-400",
          "hover:bg-teal-500/10 hover:border-teal-500/50 transition-all duration-200 active:scale-[0.98]",
        )}
        id="download-json-button"
      >
        <Download className="w-4 h-4" />
        Download Assessment as JSON
      </button>
    </div>
  );
}
