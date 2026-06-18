import { useEffect, useState } from "react";
import {
  CartesianGrid,
  ReferenceLine,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader } from "./ui/card";
import { Badge } from "./ui/badge";
import { fetchModelPerformance, type ModelPerf } from "@/lib/api";

function Metric({ label, value, sub, good }: { label: string; value: string; sub?: string; good?: boolean }) {
  return (
    <div className="rounded-lg bg-[var(--color-surface-2)] p-3">
      <div className="text-[11px] uppercase tracking-wide text-[var(--color-muted)]">{label}</div>
      <div className={`text-xl font-bold tabular-nums ${good ? "text-[var(--color-pos)]" : ""}`}>{value}</div>
      {sub && <div className="text-[11px] text-[var(--color-muted)] mt-0.5">{sub}</div>}
    </div>
  );
}

function PerfCard({ m }: { m: ModelPerf }) {
  const ideal = m.dispersion_r >= 1e6 ? "≈ Poisson" : m.dispersion_r.toFixed(1);
  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <div className="font-semibold">{m.market_label}</div>
        <Badge variant="outline">{m.n_test.toLocaleString()} test games</Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          <Metric
            label="Projection MAE"
            value={m.mae.toFixed(2)}
            sub={`baseline ${m.baseline_mae.toFixed(2)}`}
          />
          <Metric
            label="Skill vs baseline"
            value={`${m.skill_pct > 0 ? "+" : ""}${m.skill_pct}%`}
            good={m.skill_pct > 0}
            sub="lower error = better"
          />
          <Metric label="RMSE" value={m.rmse.toFixed(2)} />
          <Metric
            label="Brier (cal.)"
            value={m.brier_calibrated.toFixed(3)}
            sub={`raw ${m.brier_raw.toFixed(3)}`}
            good={m.brier_calibrated <= m.brier_raw}
          />
          <Metric label="Over-dispersion r" value={ideal} sub="NegBinom" />
          <Metric label="Hit rate (proxy)" value={`${(m.hit_rate * 100).toFixed(1)}%`} />
        </div>

        <div>
          <div className="text-xs text-[var(--color-muted)] mb-1">
            Calibration — predicted vs. actual frequency (closer to the dashed line = better)
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 10, bottom: 4, left: -20 }}>
                <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="pred"
                  domain={[0, 1]}
                  tick={{ fontSize: 10, fill: "var(--color-muted)" }}
                  tickFormatter={(v) => `${Math.round(v * 100)}%`}
                />
                <YAxis
                  type="number"
                  dataKey="actual"
                  domain={[0, 1]}
                  tick={{ fontSize: 10, fill: "var(--color-muted)" }}
                  tickFormatter={(v) => `${Math.round(v * 100)}%`}
                />
                <ZAxis type="number" dataKey="n" range={[30, 240]} />
                <ReferenceLine
                  segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                  stroke="var(--color-muted)"
                  strokeDasharray="4 4"
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{
                    background: "var(--color-surface-2)",
                    border: "1px solid var(--color-border)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v: number, name) => [`${Math.round(v * 100)}%`, name]}
                />
                <Scatter data={m.calibration} fill="var(--color-accent)" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function ModelPerformance() {
  const [data, setData] = useState<ModelPerf[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModelPerformance()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"));
  }, []);

  return (
    <div>
      <div className="mb-5">
        <h1 className="text-2xl font-bold tracking-tight">Model Performance</h1>
        <p className="text-sm text-[var(--color-muted)]">
          Time-split backtest (train on earlier games, evaluate on later ones) — projection
          accuracy, probability calibration, and dispersion per market.
        </p>
      </div>

      {error && <div className="text-[var(--color-neg)] py-8 text-center">{error}</div>}
      {!error && !data && (
        <div className="text-[var(--color-muted)] py-12 text-center">Loading metrics…</div>
      )}
      {data && data.length === 0 && (
        <Card>
          <CardContent className="py-10 text-center text-[var(--color-muted)]">
            No backtest metrics yet. Run <code>python -m courtiq.ml.backtest</code>.
          </CardContent>
        </Card>
      )}
      <div className="grid lg:grid-cols-2 gap-4">
        {data?.map((m) => <PerfCard key={m.market} m={m} />)}
      </div>
    </div>
  );
}
