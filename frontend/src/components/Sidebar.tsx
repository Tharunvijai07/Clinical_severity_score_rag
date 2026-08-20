import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Database, Hammer, RefreshCw, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { useState, useEffect } from "react";
import clsx from "clsx";

import { getBuildStatus, buildKnowledgeBase, getKnowledgeBaseStatus } from "../api/client";
import type { BuildStatus } from "../types";

export default function Sidebar() {
  const queryClient = useQueryClient();
  const [forceRebuild, setForceRebuild] = useState(false);

  // KB status query
  const { data: kbStatus } = useQuery({
    queryKey: ["kb-status"],
    queryFn: getKnowledgeBaseStatus,
    refetchInterval: 30_000,
  });

  // Build status polling (enabled when building)
  const { data: buildStatus } = useQuery<BuildStatus>({
    queryKey: ["build-status"],
    queryFn: getBuildStatus,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.running ? 2000 : false;
    },
  });

  const isBuilding = buildStatus?.running ?? false;

  // When build finishes, refetch KB status
  useEffect(() => {
    if (buildStatus && !buildStatus.running && buildStatus.progress === 100) {
      queryClient.invalidateQueries({ queryKey: ["kb-status"] });
    }
  }, [buildStatus, queryClient]);

  // Build mutation
  const buildMutation = useMutation({
    mutationFn: (force: boolean) => buildKnowledgeBase(force),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["build-status"] });
    },
  });

  const handleBuild = () => {
    if (isBuilding) return;
    buildMutation.mutate(forceRebuild);
  };

  const ready = kbStatus?.ready ?? false;
  const chunkCount = kbStatus?.chunk_count ?? buildStatus?.chunk_count ?? 0;

  return (
    <aside className="glass-panel p-5 space-y-5" id="sidebar">
      {/* Title */}
      <div className="flex items-center gap-2">
        <Database className="w-4 h-4 text-teal-400" />
        <h2 className="text-sm font-semibold" style={{ color: "hsl(var(--text-primary))" }}>
          Knowledge Base
        </h2>
      </div>

      {/* Status badge */}
      <div className="glass-panel-sm p-3 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium" style={{ color: "hsl(var(--text-secondary))" }}>Status</span>
          <div className={clsx(
            "flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-semibold",
            ready
              ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25"
              : "bg-amber-500/15 text-amber-400 border border-amber-500/25",
          )}>
            {ready ? <CheckCircle2 className="w-3 h-3" /> : <AlertCircle className="w-3 h-3" />}
            {ready ? "Ready" : "Not Built"}
          </div>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs" style={{ color: "hsl(var(--text-muted))" }}>Chunks indexed</span>
          <span className="text-xs font-semibold tabular-nums" style={{ color: "hsl(var(--text-primary))" }}>
            {chunkCount.toLocaleString()}
          </span>
        </div>

        {kbStatus?.collection_name && (
          <div className="flex items-center justify-between">
            <span className="text-xs" style={{ color: "hsl(var(--text-muted))" }}>Collection</span>
            <span className="text-[11px] font-mono" style={{ color: "hsl(var(--text-secondary))" }}>
              {kbStatus.collection_name}
            </span>
          </div>
        )}
      </div>

      {/* Build progress */}
      {isBuilding && buildStatus && (
        <div className="space-y-2 animate-fade-in-up">
          <div className="flex items-center gap-2 text-xs font-medium text-teal-400">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Building…
          </div>
          <div className="w-full h-2 rounded-full" style={{ background: "hsl(var(--bg-surface))" }}>
            <div
              className="progress-fill h-full"
              style={{ width: `${buildStatus.progress}%` }}
            />
          </div>
          <p className="text-[11px]" style={{ color: "hsl(var(--text-muted))" }}>
            {buildStatus.message}
          </p>
        </div>
      )}

      {/* Build error */}
      {buildStatus?.error && (
        <div className="glass-panel-sm p-3 border-red-500/30 text-xs text-red-400 animate-fade-in-up">
          <p className="font-semibold">Build failed</p>
          <p className="mt-1 opacity-80">{buildStatus.error}</p>
        </div>
      )}

      {/* Force rebuild toggle */}
      <label className="flex items-center gap-2 cursor-pointer group" id="force-rebuild-toggle">
        <input
          type="checkbox"
          checked={forceRebuild}
          onChange={(e) => setForceRebuild(e.target.checked)}
          className="sr-only peer"
        />
        <div className={clsx(
          "relative w-8 h-[18px] rounded-full transition-colors",
          forceRebuild ? "bg-teal-500/40" : "bg-[hsl(var(--border))]",
        )}>
          <div className={clsx(
            "absolute top-[2px] w-[14px] h-[14px] rounded-full transition-all duration-200",
            forceRebuild ? "left-[16px] bg-teal-400" : "left-[2px] bg-[hsl(var(--text-muted))]",
          )} />
        </div>
        <span className="text-xs" style={{ color: "hsl(var(--text-secondary))" }}>
          Force rebuild
        </span>
      </label>

      {/* Build button */}
      <button
        onClick={handleBuild}
        disabled={isBuilding}
        className={clsx(
          "w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200",
          isBuilding
            ? "opacity-50 cursor-not-allowed bg-[hsl(var(--bg-hover))] text-[hsl(var(--text-muted))]"
            : "bg-gradient-to-r from-teal-600 to-emerald-600 text-white hover:from-teal-500 hover:to-emerald-500 hover:shadow-[0_0_20px_hsl(174_50%_55%/0.2)] active:scale-[0.98]",
        )}
        id="build-kb-button"
      >
        {isBuilding ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : forceRebuild ? (
          <RefreshCw className="w-4 h-4" />
        ) : (
          <Hammer className="w-4 h-4" />
        )}
        {isBuilding ? "Building…" : forceRebuild ? "Rebuild Knowledge Base" : "Build Knowledge Base"}
      </button>
    </aside>
  );
}
