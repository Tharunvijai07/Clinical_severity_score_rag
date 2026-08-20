import type { SeverityLevel } from "../types";

export function formatPercent(value: number): string {
  const normalized = value > 1 ? value / 100 : value;
  return `${Math.round(Math.max(0, Math.min(1, normalized)) * 100)}%`;
}

export function normalizeConfidence(value: number): number {
  return value > 1 ? value / 100 : value;
}

export function severityColor(level: SeverityLevel): string {
  return {
    Low: "#4f8f6a",
    Moderate: "#b89b46",
    High: "#c97942",
  }[level];
}

export function severityTone(level: SeverityLevel): string {
  return {
    Low: "border-severity-low bg-green-50 text-green-900",
    Moderate: "border-severity-moderate bg-amber-50 text-amber-950",
    High: "border-severity-high bg-orange-50 text-orange-950",
  }[level];
}

export function truncateText(text: string, limit = 360): string {
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit).trim()}...`;
}
