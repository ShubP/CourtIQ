"""Central configuration. Reads from environment / .env, with sane local defaults."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the backend/ directory if present.
BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

# --- Database ---------------------------------------------------------------
# Dev default: a local SQLite file so the whole pipeline runs with zero setup.
# Prod: set DATABASE_URL to your Neon Postgres URL, e.g.
#   postgresql+psycopg2://user:pass@host/db?sslmode=require
# `or` so an empty DATABASE_URL= in .env still falls back to local SQLite.
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{BACKEND_DIR / 'sportsiq.db'}"
# Accept a Neon/Heroku-style URL verbatim: normalize to the psycopg2 driver.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# --- The Odds API -----------------------------------------------------------
ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_REGION = os.getenv("ODDS_API_REGION", "us")
ODDS_API_BOOKMAKERS = os.getenv("ODDS_API_BOOKMAKERS", "")  # optional CSV filter

# --- MLB StatsAPI (no key required) -----------------------------------------
STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
MLB_SPORT_ID = 1

# --- Pipeline behaviour -----------------------------------------------------
# How many upcoming days of games/lines to consider (today + tomorrow per spec).
LOOKAHEAD_DAYS = int(os.getenv("LOOKAHEAD_DAYS", "2"))
# How many days of history to pull per player for features.
HISTORY_LOOKBACK_DAYS = int(os.getenv("HISTORY_LOOKBACK_DAYS", "365"))
# Current MLB season used for game-log queries.
SEASON = int(os.getenv("SEASON", "2026"))

# Where trained model artifacts live.
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BACKEND_DIR / "models")))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Markets ----------------------------------------------------------------
# Maps a The-Odds-API market key -> how we model it.
#   group: which game-log group the stat lives in (hitting/pitching)
#   stat:  the game-log column we predict
#   label: human-friendly name for the UI
MARKETS: dict[str, dict] = {
    "pitcher_strikeouts":    {"group": "pitching", "stat": "strikeouts",     "label": "Pitcher Ks"},
    "batter_hits_runs_rbis": {"group": "hitting",  "stat": "hits_runs_rbis", "label": "Hits+Runs+RBIs"},
}

ODDS_MARKET_KEYS = ",".join(MARKETS.keys())

# Which markets to actually FETCH from The Odds API. This controls cost:
# credits per run ~= (# of these markets) x (# upcoming events). Free tier is
# 500/month, so we focus on just Ks + H+R+RBIs. Override with ODDS_MARKETS.
ODDS_FETCH_MARKETS = os.getenv(
    "ODDS_MARKETS", "pitcher_strikeouts,batter_hits_runs_rbis"
)

# Sharp markets are strong priors. We blend the model's probability with the
# market's no-vig probability so displayed edges reflect genuine, confident
# disagreement (not model noise). 1.0 = pure model, 0.0 = pure market.
MODEL_VS_MARKET = float(os.getenv("MODEL_VS_MARKET", "0.5"))
