# CourtIQ ⚡ — MLB Prop Edge Finder

CourtIQ predicts MLB player‑prop outcomes with a machine‑learning model, compares
those projections to **live sportsbook lines**, and surfaces the bets with the
highest **expected‑value edge** — ranked, filterable, and explained.

> Reborn from a CS411 class project (originally an NBA app on a now‑defunct cloud
> DB). Rebuilt around live MLB data, a real ML model, and a modern UI. The
> original code is kept locally in `_legacy/` for reference and is **not** part of
> this repo.

---

## What it does

- Pulls the **upcoming MLB slate** (today + tomorrow) with probable pitchers and
  confirmed lineups from the free **MLB StatsAPI**.
- Trains **XGBoost** models (Poisson objective) on each player's recent game logs
  to project **pitcher strikeouts** and **batter Hits+Runs+RBIs (H+R+RBI)** — the
  two markets we focus on to stay within the free odds-API quota (configurable
  via `ODDS_MARKETS`).
- Fetches **live player‑prop lines** from [The Odds API](https://the-odds-api.com).
- Computes each bet's **edge** (expected value vs. the book's price) and ranks the
  board so the highest‑edge plays float to the top.
- Modern React UI: searchable/filterable board, color‑coded edges, and an
  expandable recent‑form chart per player.

No login — the app is open. Players/teams without upcoming games or lines are
ignored by design.

> ℹ️ Without a The Odds API key the pipeline generates clearly‑labeled **demo
> lines** from real schedules + lineups so the app runs end‑to‑end out of the box.

---

## Architecture

```
 Python pipeline (scheduled)                         React + Tailwind (Vercel)
 ┌──────────────────────────┐                        ┌───────────────────────┐
 │ MLB StatsAPI  (schedule, │                        │  Edge Board UI        │
 │   lineups, game logs)    │─┐                       │  filters · charts     │
 │ The Odds API (live lines)│ │   ┌───────────────┐  └──────────▲────────────┘
 │ XGBoost train + predict  │ ├──▶│   Postgres    │◀────────────┘  reads
 │ → edge per player/market │ │   │  (Neon/RDS)   │   FastAPI (Render)
 └──────────────────────────┘─┘   └───────────────┘
       GitHub Actions cron
```

- **backend/** — Python package `courtiq`: data sources, ML, pipeline, FastAPI.
- **web/** — Vite + React + TypeScript + Tailwind v4 frontend.
- **DB** — SQLAlchemy; local dev defaults to SQLite, prod uses Postgres via
  `DATABASE_URL`.

---

## Quick start (local)

### 1. Backend + pipeline

```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env            # optional: add THE_ODDS_API_KEY for live lines

python -m courtiq.pipeline      # builds the board (uses SQLite by default)
uvicorn courtiq.api:app --port 8080 --reload
```

### 2. Frontend

```bash
cd web
npm install
cp .env.example .env            # VITE_API_URL=http://localhost:8080
npm run dev                     # http://localhost:5173
```

Open http://localhost:5173.

---

## Configuration

| Variable             | Where    | Purpose                                                        |
| -------------------- | -------- | -------------------------------------------------------------- |
| `DATABASE_URL`       | backend  | Unset = SQLite dev file. Prod = Neon Postgres URL.             |
| `THE_ODDS_API_KEY`   | backend  | Live lines from The Odds API. Unset = demo lines.              |
| `FRONTEND_ORIGIN`    | backend  | Your Vercel URL, for CORS.                                     |
| `SEASON`             | backend  | MLB season for game logs (default 2026).                       |
| `VITE_API_URL`       | web      | Base URL of the API (default `http://localhost:8080`).         |

---

## How the model works

For each player/market the pipeline builds **leak‑free** features from prior games
(rolling 3/5/10‑game and season averages of the target stat and its volume
drivers, home/away, and a data‑derived opponent factor), then an XGBoost Poisson
regressor predicts the stat. The predicted mean drives a Poisson distribution to
get `P(over the line)`, which is compared to the book's implied probability to
compute **expected value (edge)**. The higher‑EV side (Over/Under) is recommended;
non‑+EV props are marked `Pass`.

This is intentionally a cheap, fast, tabular ML approach (trains in seconds, runs
free) — the right tool for stat projection, vs. an expensive fine‑tuned LLM.

---

## Deployment (free tier)

- **DB:** create a free Postgres on [Neon](https://neon.tech); set `DATABASE_URL`.
- **Pipeline:** GitHub Actions ([.github/workflows/pipeline.yml](.github/workflows/pipeline.yml))
  runs every 3 hours. Add repo secrets `DATABASE_URL` and `THE_ODDS_API_KEY`.
- **API:** deploy `backend/` to [Render](https://render.com) via
  [render.yaml](render.yaml).
- **Frontend:** import the repo on [Vercel](https://vercel.com), set root dir to
  `web/` and `VITE_API_URL` to your Render API URL.

---

## Disclaimer

Projections are model estimates for **informational/educational purposes only**
and are **not betting advice**.
