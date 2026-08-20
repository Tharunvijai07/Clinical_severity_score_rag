import { useQuery } from "@tanstack/react-query";
import { Activity } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export default function Header() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/health`);
      if (!res.ok) throw new Error("offline");
      return (await res.json()) as { status: string };
    },
    refetchInterval: 15_000,
    retry: false,
  });

  const online = health?.status === "ok";

  return (
    <header className="glass-panel flex items-center justify-between px-6 py-4" id="app-header">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-teal-500/20 to-emerald-500/20 border border-teal-500/30">
          <Activity className="w-5 h-5 text-teal-400 animate-heartbeat" />
        </div>
        <div>
          <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-teal-300 to-emerald-400 bg-clip-text text-transparent">
            MedSeverity AI
          </h1>
          <p className="text-[11px] font-medium tracking-wide uppercase" style={{ color: "hsl(var(--text-muted))" }}>
            Clinical Severity Assessment
          </p>
        </div>
      </div>

      {/* Connection status */}
      <div className="flex items-center gap-2 text-xs font-medium" style={{ color: "hsl(var(--text-secondary))" }}>
        <span className={`status-dot ${online ? "online" : "offline"}`} />
        <span>{online ? "API Connected" : "API Offline"}</span>
      </div>
    </header>
  );
}
