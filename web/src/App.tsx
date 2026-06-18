import { useEffect, useMemo, useState } from "react";
import { Search, TrendingUp, Target, Flame } from "lucide-react";
import { Header } from "./components/Header";
import { PickRow } from "./components/PickRow";
import { ModelPerformance } from "./components/ModelPerformance";
import { Select } from "./components/ui/select";
import { Card, CardContent } from "./components/ui/card";
import {
  fetchBoard,
  fetchMarkets,
  fetchMeta,
  type Market,
  type Meta,
  type Pick,
} from "./lib/api";

function StatCard({ icon, label, value, hint }: { icon: React.ReactNode; label: string; value: string; hint?: string }) {
  return (
    <Card>
      <CardContent className="flex items-center gap-3 py-3">
        <div className="grid place-items-center h-9 w-9 rounded-lg bg-[var(--color-surface-2)] text-[var(--color-accent)]">
          {icon}
        </div>
        <div>
          <div className="text-xs text-[var(--color-muted)]">{label}</div>
          <div className="text-lg font-bold leading-tight">{value}</div>
          {hint && <div className="text-[11px] text-[var(--color-muted)]">{hint}</div>}
        </div>
      </CardContent>
    </Card>
  );
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [markets, setMarkets] = useState<Market[]>([]);
  const [picks, setPicks] = useState<Pick[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [view, setView] = useState<"board" | "performance">("board");
  const [market, setMarket] = useState("");
  const [date, setDate] = useState("");
  const [rec, setRec] = useState("");
  const [minEdge, setMinEdge] = useState(0);
  const [search, setSearch] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const [m, mk, b] = await Promise.all([
          fetchMeta(),
          fetchMarkets(),
          fetchBoard({ limit: 1000 }),
        ]);
        setMeta(m);
        setMarkets(mk);
        setPicks(b);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const dates = useMemo(
    () => Array.from(new Set(picks.map((p) => p.game_date))).sort(),
    [picks]
  );

  const filtered = useMemo(() => {
    return picks
      .filter((p) => (market ? p.market === market : true))
      .filter((p) => (date ? p.game_date === date : true))
      .filter((p) => (rec ? p.recommendation === rec : true))
      .filter((p) => p.edge >= minEdge)
      .filter((p) =>
        search ? p.player_name.toLowerCase().includes(search.toLowerCase()) : true
      );
  }, [picks, market, date, rec, minEdge, search]);

  const positiveEV = filtered.filter((p) => p.edge > 0).length;
  const topEdge = filtered.length ? filtered[0].edge_pct : 0;

  return (
    <div className="min-h-screen">
      <Header meta={meta} />

      <main className="mx-auto max-w-7xl px-4 py-6">
        {/* Tabs */}
        <div className="flex gap-1 mb-6 p-1 rounded-xl bg-[var(--color-surface-2)] w-fit">
          {([["board", "Edge Board"], ["performance", "Model Performance"]] as const).map(
            ([key, label]) => (
              <button
                key={key}
                onClick={() => setView(key)}
                className={
                  "px-4 py-1.5 rounded-lg text-sm font-medium transition-colors " +
                  (view === key
                    ? "bg-[var(--color-accent)] text-black"
                    : "text-[var(--color-muted)] hover:text-[#e6edf6]")
                }
              >
                {label}
              </button>
            )
          )}
        </div>

        {view === "performance" && <ModelPerformance />}

        {view === "board" && (
        <>
        <div className="mb-5">
          <h1 className="text-2xl font-bold tracking-tight">Today's Edge Board</h1>
          <p className="text-sm text-[var(--color-muted)]">
            Model projections vs sportsbook lines for upcoming MLB games — ranked by expected value.
          </p>
        </div>

        {/* Summary */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
          <StatCard icon={<Target className="h-5 w-5" />} label="Props analyzed" value={String(picks.length)} />
          <StatCard icon={<TrendingUp className="h-5 w-5" />} label="+EV plays" value={String(positiveEV)} hint="edge above 0%" />
          <StatCard icon={<Flame className="h-5 w-5" />} label="Top edge" value={`${topEdge > 0 ? "+" : ""}${topEdge}%`} />
          <StatCard icon={<Target className="h-5 w-5" />} label="Games" value={String(meta?.games ?? 0)} hint="today + tomorrow" />
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-2 mb-4">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--color-muted)]" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search player…"
              className="h-9 w-48 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] pl-8 pr-3 text-sm outline-none focus:border-[var(--color-accent)]"
            />
          </div>

          <Select
            aria-label="Market"
            value={market}
            onChange={setMarket}
            options={[{ value: "", label: "All markets" }, ...markets.map((m) => ({ value: m.key, label: m.label }))]}
          />
          {dates.length > 1 && (
            <Select
              aria-label="Date"
              value={date}
              onChange={setDate}
              options={[{ value: "", label: "All dates" }, ...dates.map((d) => ({ value: d, label: d }))]}
            />
          )}
          <Select
            aria-label="Side"
            value={rec}
            onChange={setRec}
            options={[
              { value: "", label: "Over & Under" },
              { value: "Over", label: "Over only" },
              { value: "Under", label: "Under only" },
            ]}
          />
          <Select
            aria-label="Minimum edge"
            value={String(minEdge)}
            onChange={(v) => setMinEdge(Number(v))}
            options={[
              { value: "0", label: "+EV only" },
              { value: "-1", label: "All edges" },
              { value: "0.03", label: "Edge ≥ 3%" },
              { value: "0.05", label: "Edge ≥ 5%" },
              { value: "0.1", label: "Edge ≥ 10%" },
            ]}
          />
          <span className="text-xs text-[var(--color-muted)] ml-auto">
            {filtered.length} shown
          </span>
        </div>

        {/* Board */}
        {loading && <div className="text-[var(--color-muted)] py-12 text-center">Loading the board…</div>}
        {error && (
          <Card>
            <CardContent className="py-8 text-center text-[var(--color-neg)]">
              {error}
              <div className="text-[var(--color-muted)] text-sm mt-1">
                Is the API running on :8080? Try <code>uvicorn sportsiq.api:app --port 8080</code>
              </div>
            </CardContent>
          </Card>
        )}
        {!loading && !error && filtered.length === 0 && (
          <Card>
            <CardContent className="py-10 text-center text-[var(--color-muted)]">
              No props match your filters.
            </CardContent>
          </Card>
        )}

        <div className="space-y-2">
          {filtered.map((p) => (
            <PickRow key={`${p.game_pk}-${p.player_id}-${p.market}`} pick={p} />
          ))}
        </div>

        </>
        )}

        <footer className="mt-10 text-center text-xs text-[var(--color-muted)]">
          SportsIQ · projections are model estimates for informational purposes only · not betting advice
        </footer>
      </main>
    </div>
  );
}
