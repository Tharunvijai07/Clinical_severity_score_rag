import type { AnalyzeResponse, BuildStatus, KnowledgeBaseStatus } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

async function parseError(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { detail?: unknown; message?: unknown };
    const detail = data.detail ?? data.message;
    if (typeof detail === "string") {
      return detail;
    }
  } catch {
    // Fall through to status text.
  }
  return response.statusText || "Request failed";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return (await response.json()) as T;
}

export function getKnowledgeBaseStatus(): Promise<KnowledgeBaseStatus> {
  return request<KnowledgeBaseStatus>("/api/knowledge-base/status");
}

export function getBuildStatus(): Promise<BuildStatus> {
  return request<BuildStatus>("/api/knowledge-base/build/status");
}

export function buildKnowledgeBase(force: boolean): Promise<BuildStatus> {
  return request<BuildStatus>(`/api/knowledge-base/build?force=${force}`, {
    method: "POST",
  });
}

export function analyzeReport(file: File): Promise<AnalyzeResponse> {
  const body = new FormData();
  body.append("file", file);

  return request<AnalyzeResponse>("/api/analyze", {
    method: "POST",
    body,
  });
}

export { API_BASE_URL };
