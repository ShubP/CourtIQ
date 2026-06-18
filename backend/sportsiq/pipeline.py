"""End-to-end pipeline: schedule -> history -> train -> odds -> edges -> DB.

Run with:  python -m sportsiq.pipeline
The board this writes is what the API/frontend read.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from dataclasses import dataclass

from sqlalchemy import delete

from .config import LOOKAHEAD_DAYS, MARKETS, SEASON
from .db import init_db, session_scope
from .models import (
    Game,
    PipelineRun,
    Player,
    PlayerGameLog,
    Prediction,
    Prop,
    Team,
)
from .ml import train as train_mod
from .ml.predict import predict_one
from .ml.probability import american_from_prob, prob_over
from .sources import odds as odds_src
from .sources import statsapi

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sportsiq.pipeline")

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name: str) -> str:
    """Normalize a player name for matching across data sources."""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    n = n.lower()
    n = re.sub(r"[^a-z\s]", "", n)
    parts = [p for p in n.split() if p not in _SUFFIXES]
    return " ".join(parts)


@dataclass
class PredContext:
    player_id: int
    player_name: str
    group: str
    game_pk: int
    game_date: str
    is_home: bool
    team_id: int | None
    team_abbr: str | None
    opponent_team_id: int | None
    opponent_abbr: str | None


def _team_side(team_id: int | None, game: statsapi.ScheduledGame):
    """Return (is_home, opponent_team_id, opp_abbr) or None if team not in game."""
    if team_id == game.home_team_id:
        return True, game.away_team_id, game.away_team_abbr
    if team_id == game.away_team_id:
        return False, game.home_team_id, game.home_team_abbr
    return None


def _gather_backtest_logs(days: int | None = None) -> dict[str, dict[int, list]]:
    """Pull season game logs for all probable pitchers + lineup batters on the
    upcoming slate — a broad, real sample for training/backtesting."""
    games = statsapi.get_schedule(days=days or LOOKAHEAD_DAYS)
    pitcher_ids: set[int] = set()
    batter_ids: set[int] = set()
    for g in games:
        for pp in (g.home_probable_pitcher, g.away_probable_pitcher):
            if pp:
                pitcher_ids.add(pp.player_id)
        for lu in (g.home_lineup, g.away_lineup):
            for p in lu:
                batter_ids.add(p.player_id)

    logs_by_group: dict[str, dict[int, list]] = {"hitting": {}, "pitching": {}}
    log.info("Backtest gather: %d pitchers, %d batters", len(pitcher_ids), len(batter_ids))
    for pid in pitcher_ids:
        try:
            logs_by_group["pitching"][pid] = statsapi.get_game_log(pid, "pitching", SEASON)
        except Exception:  # noqa: BLE001
            pass
    for pid in batter_ids:
        try:
            logs_by_group["hitting"][pid] = statsapi.get_game_log(pid, "hitting", SEASON)
        except Exception:  # noqa: BLE001
            pass
    return logs_by_group


def run_pipeline(limit_games: int | None = None, synthetic_if_no_odds: bool = True) -> dict:
    init_db()

    games = statsapi.get_schedule(days=LOOKAHEAD_DAYS)
    if limit_games:
        games = games[:limit_games]
    log.info("Schedule: %d games over %d days", len(games), LOOKAHEAD_DAYS)

    # Season-wide player directory for resolving odds names + metadata.
    players = statsapi.get_all_players(SEASON)
    meta_by_id = {p["player_id"]: p for p in players}
    name_to_ids: dict[str, list[int]] = {}
    for p in players:
        name_to_ids.setdefault(norm_name(p["full_name"]), []).append(p["player_id"])

    # Index games by the pair of team ids and by normalized team name.
    name_to_team_id: dict[str, int] = {}
    for g in games:
        name_to_team_id[norm_name(g.home_team_name)] = g.home_team_id
        name_to_team_id[norm_name(g.away_team_name)] = g.away_team_id
    game_by_teams: dict[frozenset, statsapi.ScheduledGame] = {
        frozenset({g.home_team_id, g.away_team_id}): g for g in games
    }
    abbr_by_team: dict[int, str] = {}
    for g in games:
        abbr_by_team[g.home_team_id] = g.home_team_abbr
        abbr_by_team[g.away_team_id] = g.away_team_abbr

    # --- Build prediction contexts + the prop lines that go with them --------
    contexts: list[PredContext] = []
    prop_rows: list[dict] = []  # raw line info aligned to contexts by index
    using_synthetic = False

    raw_props = odds_src.get_all_upcoming_props() if odds_src.has_key() else []
    if raw_props:
        log.info("Fetched %d live prop lines from The Odds API", len(raw_props))
        for rp in raw_props:
            if rp.market not in MARKETS:
                continue
            ht = name_to_team_id.get(norm_name(rp.home_team))
            at = name_to_team_id.get(norm_name(rp.away_team))
            if ht is None or at is None:
                continue
            game = game_by_teams.get(frozenset({ht, at}))
            if game is None:
                continue
            ids = name_to_ids.get(norm_name(rp.player_name), [])
            if not ids:
                continue
            pid = ids[0]
            meta = meta_by_id.get(pid, {})
            side = _team_side(meta.get("team_id"), game)
            if side is None:
                continue
            is_home, opp_id, opp_abbr = side
            contexts.append(
                PredContext(
                    player_id=pid,
                    player_name=rp.player_name,
                    group=MARKETS[rp.market]["group"],
                    game_pk=game.game_pk,
                    game_date=game.game_date,
                    is_home=is_home,
                    team_id=meta.get("team_id"),
                    team_abbr=abbr_by_team.get(meta.get("team_id")),
                    opponent_team_id=opp_id,
                    opponent_abbr=opp_abbr,
                )
            )
            prop_rows.append(
                {
                    "market": rp.market,
                    "line": rp.line,
                    "over_odds": rp.over_odds,
                    "under_odds": rp.under_odds,
                    "bookmaker": rp.bookmaker,
                }
            )
    elif synthetic_if_no_odds:
        using_synthetic = True
        log.warning("No Odds API key/props — generating DEMO lines from probables + lineups")
        # Use every game on the slate: posted lineups give a full demo board.
        # With a real Odds API key this branch is skipped and only upcoming
        # games with live lines appear.
        for g in games:
            # Pitcher Ks for both probable pitchers.
            for pp, is_home in ((g.away_probable_pitcher, False), (g.home_probable_pitcher, True)):
                if not pp:
                    continue
                opp_id = g.home_team_id if not is_home else g.away_team_id
                opp_abbr = g.home_team_abbr if not is_home else g.away_team_abbr
                team_id = g.away_team_id if not is_home else g.home_team_id
                contexts.append(PredContext(
                    pp.player_id, pp.full_name, "pitching", g.game_pk, g.game_date,
                    is_home, team_id, abbr_by_team.get(team_id), opp_id, opp_abbr,
                ))
                prop_rows.append({"market": "pitcher_strikeouts"})
            # Batter props for lineup hitters.
            for lineup, is_home in ((g.away_lineup, False), (g.home_lineup, True)):
                opp_id = g.home_team_id if not is_home else g.away_team_id
                opp_abbr = g.home_team_abbr if not is_home else g.away_team_abbr
                team_id = g.away_team_id if not is_home else g.home_team_id
                for player in lineup:
                    for market in ("batter_hits_runs_rbis",):
                        contexts.append(PredContext(
                            player.player_id, player.full_name, "hitting", g.game_pk,
                            g.game_date, is_home, team_id, abbr_by_team.get(team_id),
                            opp_id, opp_abbr,
                        ))
                        prop_rows.append({"market": market})

    # --- Fetch + cache game logs for every (player, group) we need ----------
    log_cache: dict[tuple[int, str], list] = {}
    needed = {(c.player_id, c.group) for c in contexts}
    log.info("Fetching game logs for %d player/group combos...", len(needed))
    for pid, group in needed:
        try:
            log_cache[(pid, group)] = statsapi.get_game_log(pid, group, SEASON)
        except Exception as exc:  # noqa: BLE001
            log.warning("game log failed for %s/%s: %s", pid, group, exc)
            log_cache[(pid, group)] = []

    # --- Train models on the gathered history -------------------------------
    logs_by_group: dict[str, dict[int, list]] = {"hitting": {}, "pitching": {}}
    for (pid, group), rows in log_cache.items():
        if rows:
            logs_by_group[group][pid] = rows
    artifacts = train_mod.train_all(logs_by_group)
    log.info("Trained models: %s", list(artifacts.keys()))

    # --- Predict edges ------------------------------------------------------
    predictions: list[dict] = []
    props_to_store: list[dict] = []
    for ctx, pr in zip(contexts, prop_rows):
        market = pr["market"]
        artifact = artifacts.get(market) or train_mod.load_model(market)
        if artifact is None:
            continue
        logs = log_cache.get((ctx.player_id, ctx.group), [])

        if using_synthetic:
            # Predict first, then set a believable demo line around it.
            base = predict_one(artifact, logs, line=0.5, over_odds=-110,
                               under_odds=-110, is_home=int(ctx.is_home),
                               opponent_team_id=ctx.opponent_team_id)
            if base is None:
                continue
            pv = base["predicted_value"]
            # A 0.5 demo line only makes sense for stats the player actually
            # accrues (skip e.g. HR for most batters); real odds cover these.
            if pv < 1.0:
                continue
            # Realistic demo line: the standard half-point under the projection.
            line = math.floor(pv) + 0.5
            # Price the demo odds off the model's own probability, shaded by a
            # small deterministic amount + vig. This makes the book disagree
            # slightly with us (the source of edge) and yields a believable mix
            # of small Over/Under edges instead of flat -110 artifacts.
            model_p = prob_over(pv, line)
            h = int(hashlib.md5(f"{ctx.player_id}-{market}".encode()).hexdigest(), 16)
            shade = (h % 1600) / 10000 - 0.08  # -0.08 .. +0.08
            market_p = min(max(model_p + shade, 0.05), 0.95)
            vig = 0.022
            over_odds = american_from_prob(market_p + vig)
            under_odds = american_from_prob(1 - market_p + vig)
            bookmaker = "demo (no API key)"
        else:
            line = pr["line"]
            over_odds = pr.get("over_odds")
            under_odds = pr.get("under_odds")
            bookmaker = pr.get("bookmaker", "")

        pred = predict_one(
            artifact, logs, line=line, over_odds=over_odds,
            under_odds=under_odds, is_home=int(ctx.is_home),
            opponent_team_id=ctx.opponent_team_id,
        )
        if pred is None:
            continue

        props_to_store.append({
            "game_pk": ctx.game_pk, "player_id": ctx.player_id,
            "player_name": ctx.player_name, "market": market, "line": line,
            "over_odds": over_odds, "under_odds": under_odds, "bookmaker": bookmaker,
        })
        predictions.append({
            "game_pk": ctx.game_pk, "player_id": ctx.player_id,
            "player_name": ctx.player_name, "team_abbrev": ctx.team_abbr,
            "opponent_abbrev": ctx.opponent_abbr, "market": market, "line": line,
            "over_odds": over_odds, "under_odds": under_odds,
            "bookmaker": bookmaker, "game_date": ctx.game_date, **pred,
        })

    # --- Persist ------------------------------------------------------------
    _persist(games, meta_by_id, log_cache, props_to_store, predictions, using_synthetic)

    result = {
        "games": len(games),
        "props": len(props_to_store),
        "predictions": len(predictions),
        "synthetic": using_synthetic,
    }
    log.info("Pipeline complete: %s", result)
    return result


def _persist(games, meta_by_id, log_cache, props_to_store, predictions, synthetic):
    with session_scope() as s:
        # Teams
        team_seen: dict[int, Team] = {}
        for g in games:
            for tid, abbr, name in (
                (g.home_team_id, g.home_team_abbr, g.home_team_name),
                (g.away_team_id, g.away_team_abbr, g.away_team_name),
            ):
                if tid not in team_seen:
                    team_seen[tid] = Team(team_id=tid, abbrev=abbr, name=name)
        for t in team_seen.values():
            s.merge(t)
        s.flush()  # teams must exist before games/players reference them (Postgres FKs)

        # Games
        for g in games:
            s.merge(Game(
                game_pk=g.game_pk, game_date=g.game_date,
                game_datetime=g.game_datetime, status=g.status,
                home_team_id=g.home_team_id, away_team_id=g.away_team_id,
                home_probable_pitcher_id=(g.home_probable_pitcher.player_id
                                          if g.home_probable_pitcher else None),
                away_probable_pitcher_id=(g.away_probable_pitcher.player_id
                                          if g.away_probable_pitcher else None),
                venue=g.venue,
            ))

        # Players we touched
        touched = {pid for (pid, _g) in log_cache}
        for pid in touched:
            m = meta_by_id.get(pid)
            if not m:
                continue
            # Only set team_id if that team is on the slate (FK safety for
            # recently-traded players whose currentTeam isn't playing today).
            team_id = m.get("team_id")
            if team_id not in team_seen:
                team_id = None
            s.merge(Player(
                player_id=pid, full_name=m["full_name"], position=m.get("position"),
                team_id=team_id, bat_side=m.get("bat_side"),
                pitch_hand=m.get("pitch_hand"), headshot_url=m.get("headshot_url"),
            ))

        # Game logs: refresh for touched players/groups
        for (pid, group), rows in log_cache.items():
            s.execute(delete(PlayerGameLog).where(
                PlayerGameLog.player_id == pid,
                PlayerGameLog.stat_group == group,
            ))
            for r in rows:
                s.add(PlayerGameLog(
                    player_id=r.player_id, game_pk=r.game_pk, game_date=r.game_date,
                    opponent_team_id=r.opponent_team_id, is_home=r.is_home,
                    stat_group=group, **r.stats,
                ))

        # Rebuild the current board
        s.execute(delete(Prop))
        s.execute(delete(Prediction))
        for p in props_to_store:
            s.add(Prop(**p))
        for p in predictions:
            s.add(Prediction(**p))

        s.add(PipelineRun(
            games=len(games), props=len(props_to_store),
            predictions=len(predictions),
            note="synthetic demo lines" if synthetic else "live odds",
        ))


if __name__ == "__main__":
    run_pipeline()
