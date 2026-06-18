"""MLB StatsAPI client (free, no key).

Provides: upcoming schedule with probable pitchers + lineups, and per-player
game logs used to build features. Docs: https://statsapi.mlb.com/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import requests

from ..config import MLB_SPORT_ID, SEASON, STATSAPI_BASE

_TIMEOUT = 20


def _get(path: str, **params) -> dict:
    resp = requests.get(f"{STATSAPI_BASE}{path}", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# --- Schedule ---------------------------------------------------------------
@dataclass
class GamePlayer:
    player_id: int
    full_name: str
    position: str | None = None


@dataclass
class ScheduledGame:
    game_pk: int
    game_date: str               # YYYY-MM-DD
    game_datetime: datetime | None
    status: str
    home_team_id: int
    home_team_abbr: str
    home_team_name: str
    away_team_id: int
    away_team_abbr: str
    away_team_name: str
    venue: str | None
    home_probable_pitcher: GamePlayer | None = None
    away_probable_pitcher: GamePlayer | None = None
    home_lineup: list[GamePlayer] = field(default_factory=list)
    away_lineup: list[GamePlayer] = field(default_factory=list)


def _parse_player(obj: dict | None) -> GamePlayer | None:
    if not obj or "id" not in obj:
        return None
    pos = (obj.get("primaryPosition") or {}).get("abbreviation")
    return GamePlayer(player_id=obj["id"], full_name=obj.get("fullName", ""), position=pos)


def get_schedule(days: int = 2, start: date | None = None) -> list[ScheduledGame]:
    """Return games for `days` days starting at `start` (default today)."""
    start = start or date.today()
    end = start + timedelta(days=days - 1)
    data = _get(
        "/schedule",
        sportId=MLB_SPORT_ID,
        startDate=start.isoformat(),
        endDate=end.isoformat(),
        hydrate="probablePitcher,lineups,team",
    )
    games: list[ScheduledGame] = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            away, home = g["teams"]["away"], g["teams"]["home"]
            at, ht = away["team"], home["team"]
            dt = None
            if g.get("gameDate"):
                try:
                    dt = datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00"))
                except ValueError:
                    dt = None
            lineups = g.get("lineups", {}) or {}
            games.append(
                ScheduledGame(
                    game_pk=g["gamePk"],
                    game_date=g.get("officialDate") or day.get("date"),
                    game_datetime=dt,
                    status=(g.get("status") or {}).get("detailedState", ""),
                    home_team_id=ht["id"],
                    home_team_abbr=ht.get("abbreviation", ""),
                    home_team_name=ht.get("name", ""),
                    away_team_id=at["id"],
                    away_team_abbr=at.get("abbreviation", ""),
                    away_team_name=at.get("name", ""),
                    venue=(g.get("venue") or {}).get("name"),
                    home_probable_pitcher=_parse_player(home.get("probablePitcher")),
                    away_probable_pitcher=_parse_player(away.get("probablePitcher")),
                    home_lineup=[p for p in (_parse_player(x) for x in lineups.get("homePlayers", [])) if p],
                    away_lineup=[p for p in (_parse_player(x) for x in lineups.get("awayPlayers", [])) if p],
                )
            )
    return games


# --- Game logs --------------------------------------------------------------
@dataclass
class GameLogRow:
    player_id: int
    game_pk: int | None
    game_date: str
    opponent_team_id: int | None
    is_home: bool | None
    stat_group: str
    stats: dict  # normalized numeric stat columns


def _innings_to_outs(ip: str | float | None) -> float:
    """MLB innings pitched are like '7.0', '6.2' (=6 and 2/3). Convert to outs."""
    if ip is None:
        return 0.0
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole or 0) * 3 + int(frac or 0)
    except (ValueError, TypeError):
        return 0.0


def get_game_log(player_id: int, group: str, season: int = SEASON) -> list[GameLogRow]:
    """Per-game log for one player and group ('hitting' | 'pitching')."""
    data = _get(
        f"/people/{player_id}/stats",
        stats="gameLog",
        season=season,
        group=group,
    )
    rows: list[GameLogRow] = []
    stat_blocks = data.get("stats", [])
    if not stat_blocks:
        return rows
    for split in stat_blocks[0].get("splits", []):
        s = split.get("stat", {})
        if group == "hitting":
            hits = s.get("hits", 0)
            runs = s.get("runs", 0)
            rbis = s.get("rbi", 0)
            norm = {
                "at_bats": s.get("atBats", 0),
                "hits": hits,
                "total_bases": s.get("totalBases", 0),
                "home_runs": s.get("homeRuns", 0),
                "rbis": rbis,
                "runs": runs,
                "hits_runs_rbis": hits + runs + rbis,
                "walks": s.get("baseOnBalls", 0),
                "strikeouts_batting": s.get("strikeOuts", 0),
            }
        else:  # pitching
            norm = {
                "innings_outs": _innings_to_outs(s.get("inningsPitched")),
                "strikeouts": s.get("strikeOuts", 0),
                "earned_runs": s.get("earnedRuns", 0),
                "hits_allowed": s.get("hits", 0),
                "walks_allowed": s.get("baseOnBalls", 0),
                "batters_faced": s.get("battersFaced", 0),
            }
        rows.append(
            GameLogRow(
                player_id=player_id,
                game_pk=(split.get("game") or {}).get("gamePk"),
                game_date=split.get("date", ""),
                opponent_team_id=(split.get("opponent") or {}).get("id"),
                is_home=split.get("isHome"),
                stat_group=group,
                stats=norm,
            )
        )
    return rows


def get_all_players(season: int = SEASON) -> list[dict]:
    """All players for a season — used to resolve odds player names to MLB ids."""
    data = _get(f"/sports/{MLB_SPORT_ID}/players", season=season)
    out = []
    for p in data.get("people", []):
        out.append(
            {
                "player_id": p["id"],
                "full_name": p.get("fullName", ""),
                "team_id": (p.get("currentTeam") or {}).get("id"),
                "position": (p.get("primaryPosition") or {}).get("abbreviation"),
                "bat_side": (p.get("batSide") or {}).get("code"),
                "pitch_hand": (p.get("pitchHand") or {}).get("code"),
                "headshot_url": f"https://midfield.mlbstatic.com/v1/people/{p['id']}/spots/120",
            }
        )
    return out


def get_player_meta(player_id: int) -> dict:
    """Basic player metadata (name, position, handedness, headshot)."""
    data = _get(f"/people/{player_id}")
    people = data.get("people", [])
    if not people:
        return {}
    p = people[0]
    return {
        "player_id": p["id"],
        "full_name": p.get("fullName", ""),
        "position": (p.get("primaryPosition") or {}).get("abbreviation"),
        "bat_side": (p.get("batSide") or {}).get("code"),
        "pitch_hand": (p.get("pitchHand") or {}).get("code"),
        "team_id": (p.get("currentTeam") or {}).get("id"),
        "headshot_url": f"https://midfield.mlbstatic.com/v1/people/{p['id']}/spots/120",
    }
