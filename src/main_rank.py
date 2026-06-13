#main_rank.py
import ast
import json
import logging
import re
import math
from pathlib import Path

import pandas as pd

from graph_engine import (
    build_graph_pools,
    build_overall_stats,
    create_rankings,
    _SlugRegistry,
    _safe_slug,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s"
)

_SINGLES_COLS = [
    "rank",
    "gender",
    "division",
    "flight",
    "name",
    "school",
    "rating",
    "TGRS",
    "reachability",
    "local_reachability",
    "sos",
    "local_sos",
    "quality_wins",
    "ts_rating",
    "ts_mu",
    "local_ts_mu",
    "ts_sigma",
    "matches_played",
    "wins",
    "losses",
    "last_match_date",
]

_DOUBLES_COLS = [
    "rank",
    "gender",
    "division",
    "flight",
    "pair_name",
    "school",
    "rating",
    "TGRS",
    "reachability",
    "local_reachability",
    "sos",
    "local_sos",
    "quality_wins",
    "ts_rating",
    "ts_mu",
    "local_ts_mu",
    "ts_sigma",
    "matches_played",
    "wins",
    "losses",
    "last_match_date",
]

_LIST_COLS = [
    "winner_player_ids",
    "loser_player_ids",
]


def load_matches(path="all_matches.csv") -> list:
    """Load match CSV and restore list columns."""
    import os
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        logging.warning("all_matches.csv is missing or empty — skipping.")
        return []
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        logging.warning("all_matches.csv has no columns — skipping.")
        return []
    for col in _LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: ast.literal_eval(v)
                if isinstance(v, str) and v.startswith("[")
                else []
            )
    return df.to_dict(orient="records")


def load_school_meta(path="school_meta.json", overrides_path="../data/division_overrides.json") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    override_file = Path(overrides_path)
    if override_file.exists():
        with open(override_file, "r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        for school_id, fields in overrides.items():
            sid = str(school_id)
            if sid not in meta:
                meta[sid] = {"id": int(sid)}
            for k, v in fields.items():
                meta[sid][k] = v
        logging.info("Applied %d division overrides", len(overrides))

    return meta


_DIVISION_MAPPING = {
    "d1": "1",
    "d2": "2",
    "d3": "3",
    "d4": "4",
    "d-1": "1",
    "aa": "1",
    "a": "1",
    "d": "4",
    "one": "1",
    "ii": "2",
    "iv": "4",
    "2a": "2",
    "3a": "4",
    "4a": "1",
    "klaa": "1",
    "ok red": "1",
    "mac bl": "1",
    "mac": "2",
    "oaa": "2",
    "lvc": "3",
    "sec wh": "3",
    "bwac": "3",
    "silver": "3",
    "gac": "4",
    "tennis": "4",
}

def _normalize_division(raw):
    s = str(raw or "").strip().lower()

    if s in _DIVISION_MAPPING:
        return _DIVISION_MAPPING[s]

    if s in ("1", "div 1", "div1", "division 1", "division1"):
        return "1"
    if s in ("2", "div 2", "div2", "division 2", "division2"):
        return "2"
    if s in ("3", "div 3", "div3", "division 3", "division3"):
        return "3"
    if s in ("4/other", "4 other", "4other", "other", "division 4", "division4"):
        return "4_other"
    if s == "4":
        return "4"

    return "4_other"


def build_lookups(matches, school_meta):
    player_lookup = {}
    pair_lookup = {}

    def normalize_school_id(sid):
        if sid is None:
            return ""
        return str(sid).split(".")[0]

    def division_for_school(school_id, gender):
        school = school_meta.get(normalize_school_id(school_id), {})
        if gender == "Boys":
            return _normalize_division(school.get("division_boys", ""))
        if gender == "Girls":
            return _normalize_division(school.get("division_girls", ""))
        return "4_other"

    for m in matches:
        gender = m.get("gender", "")
        category = m.get("match_type", "").lower().strip()

        if category == "singles":
            winner_ids = m.get("winner_player_ids", [])
            loser_ids = m.get("loser_player_ids", [])

            if winner_ids:
                pid = str(winner_ids[0])
                player_lookup[pid] = {
                    "name": m.get("winner_names", pid),
                    "school": m.get("winner_school", ""),
                    "division": division_for_school(m.get("winner_school_id"), gender),
                }

            if loser_ids:
                pid = str(loser_ids[0])
                player_lookup[pid] = {
                    "name": m.get("loser_names", pid),
                    "school": m.get("loser_school", ""),
                    "division": division_for_school(m.get("loser_school_id"), gender),
                }

        elif category == "doubles":
            winner_ids = m.get("winner_player_ids", [])
            loser_ids = m.get("loser_player_ids", [])

            if len(winner_ids) == 2:
                key = tuple(sorted(str(x) for x in winner_ids))
                pair_lookup[key] = {
                    "pair_name": m.get("winner_names", " / ".join(key)),
                    "school": m.get("winner_school", ""),
                    "division": division_for_school(m.get("winner_school_id"), gender),
                }

            if len(loser_ids) == 2:
                key = tuple(sorted(str(x) for x in loser_ids))
                pair_lookup[key] = {
                    "pair_name": m.get("loser_names", " / ".join(key)),
                    "school": m.get("loser_school", ""),
                    "division": division_for_school(m.get("loser_school_id"), gender),
                }

    return player_lookup, pair_lookup


def export_split_csvs(rows, columns, prefix):
    """
    Write one CSV per gender + division + flight bucket.
    Adds TGRS_scaled column: lowest floored to 0, highest scaled to 100.

    Raises ValueError if any two distinct gender/division/flight values
    produce the same filename slug (which would silently overwrite a file).
    """
    out_dir = Path("rankings_by_division_flight")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    if df.empty:
        logging.info("No rows to export for %s", prefix)
        return

    # Add TGRS_scaled per bucket
    scaled_col = []
    for (gender, division, flight), group in df.groupby(
        ["gender", "division", "flight"], dropna=False
    ):
        tgrs = group["TGRS"]
        sorted_tgrs = tgrs.sort_values(ascending=False).reset_index(drop=True)
        floor_val = math.floor(sorted_tgrs.iloc[29] if len(sorted_tgrs) >= 30 else sorted_tgrs.iloc[-1])
        shifted = (tgrs - floor_val).clip(lower=0)
        max_shifted = shifted.max()
        if max_shifted > 0:
            scaled = (shifted / max_shifted * 100).round(2)
        else:
            scaled = shifted * 0 + 100.0
        scaled_col.append(scaled)

    df["TGRS_scaled"] = pd.concat(scaled_col)

    cols_with_scaled = []
    for c in columns:
        cols_with_scaled.append(c)
        if c == "TGRS":
            cols_with_scaled.append("TGRS_scaled")

    # Slug registries — one per component that forms part of the filename.
    # Collisions raise ValueError before any file is written.
    gender_slugs   = _SlugRegistry(f"{prefix}/gender")
    division_slugs = _SlugRegistry(f"{prefix}/division")
    flight_slugs   = _SlugRegistry(f"{prefix}/flight")

    for (gender, division, flight), group in df.groupby(
        ["gender", "division", "flight"], dropna=False
    ):
        g_slug = gender_slugs.register(str(gender))
        d_slug = division_slugs.register(str(division))
        f_slug = flight_slugs.register(str(flight))

        out_path = out_dir / (
            f"{prefix}_{g_slug}_division_{d_slug}_flight_{f_slug}.csv"
        )
        group.to_csv(out_path, index=False, columns=cols_with_scaled)
        logging.info("Wrote %s (%d rows)", out_path, len(group))


def build_team_rankings(singles_rows, doubles_rows):
    """
    Team score per slot:
      rank 1        → 12.5 pts
      rank 2        → 10.0 pts
      rank 3-4      →  7.5 pts
      rank 5-8      →  5.0 pts
      rank 9-16     →  2.5 pts
      rank 17-32    →  1.0 pts
      rank 33+      →  0.0 pts

    One entry per school per flight/category (best-ranked player only).
    Sum across all 6 slots (S1,S2,S3,D1,D2,D3) for team total.
    """
    ALLOWED_FLIGHTS = {"1", "2", "3", "4"}
    out_dir = Path("rankings_by_division_flight")
    out_dir.mkdir(parents=True, exist_ok=True)

    def rank_to_points(r):
        if r == 1:   return 12.5
        if r == 2:   return 10.0
        if r <= 4:   return  7.5
        if r <= 8:   return  5.0
        if r <= 16:  return  2.5
        if r <= 32:  return  1.0
        return 0.0

    all_rows = (
        [dict(r, category="singles") for r in singles_rows] +
        [dict(r, category="doubles") for r in doubles_rows]
    )

    df = pd.DataFrame(all_rows)
    if df.empty:
        return

    df = df[df["flight"].astype(str).isin(ALLOWED_FLIGHTS)]
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(999).astype(int)

    best = (
        df.sort_values("rank")
        .groupby(["gender", "division", "flight", "category", "school"])
        .first()
        .reset_index()[["gender", "division", "flight", "category", "school", "rank"]]
    )

    best["points"] = best["rank"].apply(rank_to_points)

    team_scores = (
        best.groupby(["gender", "division", "school"])["points"]
        .sum()
        .reset_index()
        .rename(columns={"points": "team_score"})
    )
    team_scores["team_score"] = team_scores["team_score"].round(2)
    team_scores = team_scores.sort_values(
        ["gender", "division", "team_score"], ascending=[True, True, False]
    )
    team_scores["rank"] = (
        team_scores.groupby(["gender", "division"])["team_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    gender_slugs   = _SlugRegistry("team/gender")
    division_slugs = _SlugRegistry("team/division")

    for (gender, division), group in team_scores.groupby(["gender", "division"]):
        g_slug = gender_slugs.register(str(gender))
        d_slug = division_slugs.register(str(division))
        out_path = out_dir / f"team_{g_slug}_division_{d_slug}.csv"
        group[["rank", "school", "team_score"]].to_csv(out_path, index=False)
        logging.info("Wrote %s (%d schools)", out_path, len(group))


def main():
    logging.info("Loading match data...")
    matches = load_matches("all_matches.csv")
    if not matches:
        logging.warning("No matches loaded — aborting rankings run.")
        return
    logging.info("Loaded %d matches", len(matches))

    school_meta = load_school_meta("school_meta.json")
    player_lookup, pair_lookup = build_lookups(matches, school_meta)

    logging.info("Building graph pools...")
    pools = build_graph_pools(
        matches,
        player_lookup=player_lookup,
        pair_lookup=pair_lookup,
    )

    logging.info("Building overall season stats...")
    singles_overall_stats = build_overall_stats(matches, "singles")
    doubles_overall_stats = build_overall_stats(matches, "doubles")

    logging.info("Generating singles rankings...")
    singles_rows = create_rankings(
        pools,
        "singles",
        overall_stats=singles_overall_stats,
        player_lookup=player_lookup,
        pair_lookup=pair_lookup,
    )
    export_split_csvs(singles_rows, _SINGLES_COLS, "singles")

    logging.info("Generating doubles rankings...")
    doubles_rows = create_rankings(
        pools,
        "doubles",
        overall_stats=doubles_overall_stats,
        player_lookup=player_lookup,
        pair_lookup=pair_lookup,
    )
    export_split_csvs(doubles_rows, _DOUBLES_COLS, "doubles")

    logging.info("Generating team rankings...")
    build_team_rankings(singles_rows, doubles_rows)

    logging.info("Done.")


if __name__ == "__main__":
    main()
