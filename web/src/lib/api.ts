// Strip any trailing slash so `${API_URL}/board` never becomes `//board` (404).
const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8080").replace(/\/+$/, "");

export interface Pick {
  player_id: number;
  player_name: string;
  headshot_url: string | null;
  team: string | null;
  opponent: string | null;
  market: string;
  market_label: string;
  line: number;
  predicted_value: number;
  prob_over: number;
  prob_under: number;
  over_odds: number | null;
  under_odds: number | null;
  recommendation: "Over" | "Under" | "Pass";
  edge: number;
  edge_pct: number;
  bookmaker: string;
  game_pk: number;
  game_date: string;
}

export interface Market {
  key: string;
  label: string;
  group: string;
}

export interface Meta {
  last_run: string | null;
  synthetic: boolean;
  games: number;
  props?: number;
  predictions: number;
  note?: string;
}

export interface HistoryPoint {
  game_date: string;
  opponent: string;
  value: number;
}

export interface PlayerHistory {
  player_id: number;
  player_name: string;
  stat: string;
  history: HistoryPoint[];
  average: number;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

export interface BoardParams {
  market?: string;
  date?: string;
  recommendation?: string;
  minEdge?: number;
  limit?: number;
}

export function fetchBoard(p: BoardParams = {}): Promise<Pick[]> {
  const q = new URLSearchParams();
  if (p.market) q.set("market", p.market);
  if (p.date) q.set("date", p.date);
  if (p.recommendation) q.set("recommendation", p.recommendation);
  if (p.minEdge !== undefined) q.set("min_edge", String(p.minEdge));
  if (p.limit) q.set("limit", String(p.limit));
  return getJSON<Pick[]>(`/board?${q.toString()}`);
}

export const fetchMarkets = () => getJSON<Market[]>("/markets");
export const fetchMeta = () => getJSON<Meta>("/meta");
export const fetchPlayerHistory = (id: number, stat: string) =>
  getJSON<PlayerHistory>(`/player/${id}/history?stat=${stat}&limit=15`);
