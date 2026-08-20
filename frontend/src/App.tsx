import { useState, useCallback } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";

import { analyzeReport, getKnowledgeBaseStatus } from "./api/client";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import StepIndicator from "./components/StepIndicator";
import UploadZone from "./components/UploadZone";
import ExtractedText from "./components/ExtractedText";
import RetrievedChunks from "./components/RetrievedChunks";
import SeverityDashboard from "./components/SeverityDashboard";
import type { AnalyzeResponse } from "./types";

const STEPS = [
  { label: "Upload", description: "PDF lab report" },
  { label: "Extract", description: "Clinical text" },
  { label: "Retrieve", description: "Medical guidelines" },
  { label: "Assess", description: "Severity result" },
];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  // KB readiness
  const { data: kbStatus } = useQuery({
    queryKey: ["kb-status"],
    queryFn: getKnowledgeBaseStatus,
    refetchInterval: 30_000,
  });

  const kbReady = kbStatus?.ready ?? false;

  // Analysis mutation
  const analyzeMutation = useMutation({
    mutationFn: analyzeReport,
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleAnalyze = useCallback(() => {
    if (!file) return;
    setResult(null);
    analyzeMutation.mutate(file);
  }, [file, analyzeMutation]);

  const handleFileSelect = useCallback((f: File | null) => {
    setFile(f);
    if (!f) setResult(null);
  }, []);

  // Derive current step
  const currentStep = result
    ? 3
    : analyzeMutation.isPending
      ? 2
      : file
        ? 1
        : 0;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "hsl(var(--bg-deep))" }}>
      {/* Header */}
      <div className="sticky top-0 z-30 px-4 pt-4">
        <Header />
      </div>

      {/* Body */}
      <div className="flex-1 flex flex-col lg:flex-row gap-5 p-4 pt-5 max-w-[1400px] mx-auto w-full">
        {/* Sidebar */}
        <div className="lg:w-72 xl:w-80 flex-shrink-0">
          <div className="lg:sticky lg:top-24">
            <Sidebar />
          </div>
        </div>

        {/* Main content */}
        <main className="flex-1 min-w-0 space-y-5" id="main-content">
          {/* Step indicator */}
          <div className="glass-panel px-5 py-3">
            <StepIndicator steps={STEPS} currentStep={currentStep} />
          </div>

          {/* Upload zone */}
          <UploadZone
            file={file}
            onFileSelect={handleFileSelect}
            onAnalyze={handleAnalyze}
            isAnalyzing={analyzeMutation.isPending}
            kbReady={kbReady}
          />

          {/* Loading state */}
          {analyzeMutation.isPending && (
            <div className="glass-panel p-10 flex flex-col items-center justify-center gap-4 animate-fade-in-up">
              <div className="relative">
                <div className="w-16 h-16 rounded-2xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center">
                  <Loader2 className="w-8 h-8 text-teal-400 animate-spin" />
                </div>
              </div>
              <div className="text-center space-y-1">
                <p className="text-sm font-semibold" style={{ color: "hsl(var(--text-primary))" }}>
                  Analyzing Clinical Report
                </p>
                <p className="text-xs" style={{ color: "hsl(var(--text-muted))" }}>
                  Extracting text → Retrieving guidelines → Generating assessment…
                </p>
              </div>
              {/* Shimmer placeholders */}
              <div className="w-full space-y-3 mt-2">
                <div className="shimmer-loading h-4 w-3/4 mx-auto" />
                <div className="shimmer-loading h-4 w-1/2 mx-auto" />
                <div className="shimmer-loading h-4 w-2/3 mx-auto" />
              </div>
            </div>
          )}

          {/* Error */}
          {analyzeMutation.isError && (
            <div
              className="glass-panel p-5 border-red-500/30 animate-fade-in-up"
              role="alert"
              id="analysis-error"
            >
              <p className="text-sm font-semibold text-red-400">Analysis Failed</p>
              <p className="text-xs mt-1 text-red-400/70">
                {analyzeMutation.error instanceof Error
                  ? analyzeMutation.error.message
                  : "An unexpected error occurred. Check the backend logs."}
              </p>
            </div>
          )}

          {/* Results */}
          {result && (
            <>
              <ExtractedText text={result.extracted_text} />
              <RetrievedChunks chunks={result.retrieved_chunks} />
              <SeverityDashboard
                result={result.severity_result}
                chunks={result.retrieved_chunks}
                extractedText={result.extracted_text}
                rawResponse={result.raw_model_response}
              />
            </>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="text-center py-6 text-[11px]" style={{ color: "hsl(var(--text-muted))" }}>
        MedSeverity AI • RAG-Powered Clinical Decision Support
      </footer>
    </div>
  );
}
