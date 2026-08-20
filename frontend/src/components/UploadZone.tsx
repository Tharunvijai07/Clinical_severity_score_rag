import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, X, Loader2, Search } from "lucide-react";
import clsx from "clsx";

interface UploadZoneProps {
  file: File | null;
  onFileSelect: (file: File | null) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  kbReady: boolean;
}

export default function UploadZone({ file, onFileSelect, onAnalyze, isAnalyzing, kbReady }: UploadZoneProps) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) {
        onFileSelect(accepted[0]);
      }
    },
    [onFileSelect],
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
    disabled: isAnalyzing,
  });

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="glass-panel p-6 space-y-5 animate-fade-in-up" id="upload-zone-panel">
      <div className="flex items-center gap-2">
        <Upload className="w-4 h-4 text-teal-400" />
        <h2 className="text-sm font-semibold" style={{ color: "hsl(var(--text-primary))" }}>
          Upload Lab Report
        </h2>
      </div>

      {/* Drop zone */}
      {!file ? (
        <div
          {...getRootProps()}
          className={clsx(
            "drop-zone flex flex-col items-center justify-center gap-3 p-10 text-center",
            isDragActive && !isDragReject && "active",
            isDragReject && "reject",
          )}
          id="drop-area"
        >
          <input {...getInputProps()} id="file-input" />
          <div className="w-14 h-14 rounded-2xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center">
            <FileText className={clsx("w-7 h-7 text-teal-400 transition-transform", isDragActive && "scale-110")} />
          </div>
          <div>
            <p className="text-sm font-medium" style={{ color: "hsl(var(--text-primary))" }}>
              {isDragActive
                ? isDragReject
                  ? "Only PDF files are accepted"
                  : "Drop your PDF here"
                : "Drag & drop a PDF lab report"}
            </p>
            <p className="text-xs mt-1" style={{ color: "hsl(var(--text-muted))" }}>
              or click to browse • PDF files only
            </p>
          </div>
        </div>
      ) : (
        /* File preview */
        <div className="glass-panel-sm p-4 flex items-center justify-between animate-slide-in" id="file-preview">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-10 h-10 rounded-lg bg-teal-500/15 border border-teal-500/25 flex items-center justify-center flex-shrink-0">
              <FileText className="w-5 h-5 text-teal-400" />
            </div>
            <div className="min-w-0">
              <p className="text-sm font-medium truncate" style={{ color: "hsl(var(--text-primary))" }}>
                {file.name}
              </p>
              <p className="text-xs" style={{ color: "hsl(var(--text-muted))" }}>
                {formatSize(file.size)} • PDF
              </p>
            </div>
          </div>
          {!isAnalyzing && (
            <button
              onClick={() => onFileSelect(null)}
              className="p-1.5 rounded-lg hover:bg-[hsl(var(--bg-hover))] transition-colors"
              aria-label="Remove file"
              id="remove-file-button"
            >
              <X className="w-4 h-4" style={{ color: "hsl(var(--text-muted))" }} />
            </button>
          )}
        </div>
      )}

      {/* Analyze button */}
      <button
        onClick={onAnalyze}
        disabled={!file || isAnalyzing || !kbReady}
        className={clsx(
          "w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-sm font-semibold transition-all duration-200",
          !file || !kbReady
            ? "opacity-40 cursor-not-allowed bg-[hsl(var(--bg-hover))] text-[hsl(var(--text-muted))]"
            : isAnalyzing
              ? "opacity-70 cursor-wait bg-gradient-to-r from-teal-600 to-emerald-600 text-white"
              : "bg-gradient-to-r from-teal-600 to-emerald-600 text-white hover:from-teal-500 hover:to-emerald-500 hover:shadow-[0_0_24px_hsl(174_50%_55%/0.25)] active:scale-[0.98]",
        )}
        id="analyze-button"
      >
        {isAnalyzing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Analyzing Report…
          </>
        ) : (
          <>
            <Search className="w-4 h-4" />
            Analyze Report
          </>
        )}
      </button>

      {!kbReady && file && (
        <p className="text-xs text-amber-400/80 text-center">
          Build the Knowledge Base first to enable analysis.
        </p>
      )}
    </div>
  );
}
