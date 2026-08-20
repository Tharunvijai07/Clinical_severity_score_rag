export type SeverityLevel = "Low" | "Moderate" | "High";

export interface RetrievedChunk {
  text: string;
  source: string;
  distance: number;
  similarity: number;
}

export interface SeverityResult {
  severity_score: number;
  severity_level: SeverityLevel;
  confidence: number;
  key_findings: string[];
  evidence: string[];
  summary: string;
}

export interface AnalyzeResponse {
  extracted_text: string;
  retrieved_chunks: RetrievedChunk[];
  severity_result: SeverityResult;
  raw_model_response: string;
}

export interface KnowledgeBaseStatus {
  ready: boolean;
  chunk_count: number;
  collection_name: string;
}

export interface BuildStatus extends KnowledgeBaseStatus {
  running: boolean;
  progress: number;
  message: string;
  error: string | null;
}
