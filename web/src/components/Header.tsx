import { Activity, Zap } from "lucide-react";
import { Badge } from "./ui/badge";
import type { Meta } from "@/lib/api";

export function Header({ meta }: { meta: Meta | null }) {
  const updated = meta?.last_run
    ? new Date(meta.last_run).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
    : "—";

  return (
    <header className="border-b border-[var(--color-border)] bg-[var(--color-surface)]/60 backdrop-blur sticky top-0 z-20">
      <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="grid place-items-center h-9 w-9 rounded-xl bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent-2)] text-black">
            <Zap className="h-5 w-5" strokeWidth={2.5} />
          </div>
          <div>
            <div className="text-lg font-bold tracking-tight leading-none">
              Sports<span className="text-[var(--color-accent)]">IQ</span>
            </div>
            <div className="text-[11px] text-[var(--color-muted)] leading-none mt-0.5">
              MLB Prop Edge Finder
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {meta?.synthetic ? (
            <Badge variant="accent" title="Showing demo lines — add a The Odds API key for live lines">
              <Activity className="h-3 w-3" /> Demo lines
            </Badge>
          ) : (
            <Badge variant="over">
              <Activity className="h-3 w-3" /> Live odds
            </Badge>
          )}
          <span className="text-xs text-[var(--color-muted)] hidden sm:block">
            Updated {updated}
          </span>
        </div>
      </div>
    </header>
  );
}
