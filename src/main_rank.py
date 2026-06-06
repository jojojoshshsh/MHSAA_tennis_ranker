#main_rank.py
import ast
import json
import logging
import re
from pathlib import Path

import pandas as pd

from graph_engine import (
    build_graph_pools,
    build_overall_stats,
    create_rankings,
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
    "local_sos",       # NEW – SOS within same gender/category/division/flight bucket
    "quality_wins",
    "ts_rating",
    "ts_mu",
    "local_ts_mu",     # NEW – TrueSkill mu from local bucket model
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
    "local_sos",       # NEW
    "quality_wins",
    "ts_rating",
    "ts_mu",
    "local_ts_mu",     # NEW
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
    df = pd.read_csv(path, dtype=str).fillna("")
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


def _safe_slug(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


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
    """
    out_dir = Path("rankings_by_division_flight")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    if df.empty:
        logging.info("No rows to export for %s", prefix)
        return

    for (gender, division, flight), group in df.groupby(
        ["gender", "division", "flight"],
        dropna=False
    ):
        out_path = out_dir / (
            f"{prefix}_{_safe_slug(gender)}_division_{_safe_slug(division)}"
            f"_flight_{_safe_slug(flight)}.csv"
        )
        group.to_csv(out_path, index=False, columns=columns)
        logging.info("Wrote %s (%d rows)", out_path, len(group))


def main():
    logging.info("Loading match data...")
    matches = load_matches("all_matches.csv")
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

    logging.info("Done.")


if __name__ == "__main__":
    main()
