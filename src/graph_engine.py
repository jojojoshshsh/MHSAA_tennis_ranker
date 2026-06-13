# graph_engine.py
from collections import defaultdict
from datetime import datetime
from statistics import mean
import logging

from config import MIN_MATCHES, TARGET_GENDER
from trueskill_engine import compute_trueskill

# ============================================================
# TUNABLE WEIGHTS
# ============================================================
#
# Each weight is a plain named constant. The values below
# reproduce the original effective weights exactly — the only
# change is making the intent of each component legible.
#
# To re-tune: adjust the numbers here and the TGRS formula in
# create_rankings. All weights sum to a meaningful total so
# their relative magnitudes can be compared directly.
#
# Derivation of original weights (for reference):
#   original denominator : 28.25
#   REACH         = (10.00 + 15)    / 28.25  ≈ 0.8850
#   QUALITY_WINS  = (3.00 + 3.5 + 5)/ 28.25  ≈ 0.4071
#   TS_MU         = (3+1+3+2+1+0.4+1+1)/28.25 ≈ 0.4301
#   SOS           = (3.20+2.25+0.10+4.5)/28.25 ≈ 0.3575
#   LOCAL_TS_MU   = (0.30+1+0.5+1+4.25)/28.25  ≈ 0.2513
#   LOCAL_SOS     = 0.2 + 0.1 + 0.1             = 0.4000
#   LOCAL_REACH   = 1.75
#   H2H           = 0.005  (bonus applied after base score is fixed)

TGRS_REACH_WEIGHT        = (10.00 + 15)             / 28.25   # global win-graph reachability
TGRS_QUALITY_WINS_WEIGHT = (3.00 + 3.5 + 5)         / 28.25   # avg TS-mu of top-3 victims (global)
TGRS_TS_MU_WEIGHT        = (3.00+1+3+2+1+0.4+1+1)   / 28.25   # global TrueSkill mean
TGRS_SOS_WEIGHT          = (3.20+2.25+0.10+4.5)      / 28.25   # global strength-of-schedule
TGRS_LOCAL_TS_MU_WEIGHT  = (0.30+1+0.5+1+4.25)       / 28.25   # local-bucket TrueSkill mean
TGRS_LOCAL_SOS_WEIGHT    = 0.2 + 0.1 + 0.1                     # local strength-of-schedule
TGRS_LOCAL_REACH_WEIGHT  = 1.75                                 # local win-graph reachability

# H2H bonus: for each direct win over an eligible opponent, add this
# fraction of that opponent's *pre-bonus* TGRS score.  Applied after
# base scores are fully computed so the bonus values are stable and
# there is no circular dependency between the bonus assignments.
TGRS_H2H_WEIGHT          = 0.005

# ============================================================
# UTILITIES
# ============================================================

def _parse_dt(raw):
    if not raw:
        return datetime.min
    try:
        return datetime.fromisoformat(
            str(raw).replace("Z", "+00:00")
        ).replace(tzinfo=None)
    except Exception:
        return datetime.min


def _entity_from_match(match, category, side, flight=None):
    """
    Return a (flight, division)-scoped entity key so that the same
    player appearing in multiple flights or divisions is treated as a
    separate entity per bucket.

    Singles  → (player_id_str, flight_str, division_str)
    Doubles  → (id_a, id_b, flight_str, division_str)
                sorted pair ids with flight+division appended
    """
    ids = match.get(f"{side}_player_ids") or []
    flight_str   = str(flight) if flight is not None else str(match.get("flight") or "?")
    division_str = str(match.get("_resolved_division") or "?")

    if category == "singles":
        if len(ids) != 1:
            return None
        return (str(ids[0]), flight_str, division_str)

    if len(ids) != 2:
        return None
    sorted_ids = tuple(sorted(str(x) for x in ids))
    return sorted_ids + (flight_str, division_str)


def _pair_key(a, b):
    return tuple(sorted((repr(a), repr(b))))


def _parse_score(score):
    games_for = games_against = sets_for = sets_against = 0
    for token in str(score or "").split():
        try:
            w, l = map(int, token.split("-"))
        except Exception:
            continue
        games_for += w
        games_against += l
        if w > l:
            sets_for += 1
        else:
            sets_against += 1
    return games_for, games_against, sets_for, sets_against


def _entity_label(entity):
    """Human-readable label; strip the trailing flight+division tags."""
    if isinstance(entity, tuple):
        # Singles: ("pid", flight, div)      → pid
        # Doubles: (id_a, id_b, flight, div) → "id_a / id_b"
        if len(entity) == 3:
            return str(entity[0])
        return " / ".join(entity[:-2])
    return str(entity)


def _bare_player_id(entity):
    """Return the raw player/pair ID without flight+division suffix."""
    if isinstance(entity, tuple):
        if len(entity) == 3:          # singles (pid, flight, div)
            return entity[0]
        return entity[:-2]            # doubles (id_a, id_b, flight, div) → (id_a, id_b)
    return entity


def _reachability_score(start, graph):
    """Count unique nodes reachable from `start` via directed win-edges."""
    seen = set()
    stack = list(graph.get(start, []))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for nxt in graph.get(node, []):
            if nxt not in seen:
                stack.append(nxt)
    return len(seen)


def _normalize_division(raw):
    s = str(raw or "").strip().lower()
    if s in {"1", "div 1", "div1", "division 1", "division1"}:
        return "1"
    if s in {"2", "div 2", "div2", "division 2", "division2"}:
        return "2"
    if s in {"3", "div 3", "div3", "division 3", "division3"}:
        return "3"
    if s in {"4", "4/other", "4 other", "4other", "other", "division 4", "division4"}:
        return "4_other"
    return "4_other"


def _match_division(match, category, player_lookup=None, pair_lookup=None):
    player_lookup = player_lookup or {}
    pair_lookup = pair_lookup or {}

    if category == "singles":
        for side in ("winner", "loser"):
            ids = match.get(f"{side}_player_ids") or []
            if len(ids) == 1:
                pid = str(ids[0])
                meta = player_lookup.get(pid, {})
                return _normalize_division(meta.get("division", "4_other"))
        return "4_other"

    if category == "doubles":
        for side in ("winner", "loser"):
            ids = match.get(f"{side}_player_ids") or []
            if len(ids) == 2:
                key = tuple(sorted(str(x) for x in ids))
                meta = pair_lookup.get(key, {})
                return _normalize_division(meta.get("division", "4_other"))
        return "4_other"

    return "4_other"


def _safe_mean(values, default=0.0):
    vals = list(values)
    if not vals:
        return default
    return mean(vals)


def _top_n_average(values, n=5, default=0.0):
    vals = sorted(values, reverse=True)[:n]
    if not vals:
        return default
    return mean(vals)


def _safe_slug(value):
    """
    Convert a value to a filesystem-safe slug.

    Raises ValueError if two semantically different inputs produce the
    same slug, since that would silently overwrite a CSV file.  Callers
    that build filenames from multiple slug components are encouraged to
    pass each component through this function individually and keep
    a registry of seen (component, slug) pairs.
    """
    import re
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


class _SlugRegistry:
    """
    Tracks (original_value → slug) mappings and raises if two distinct
    values collide onto the same slug within the same namespace.
    """
    def __init__(self, namespace: str = ""):
        self._namespace = namespace
        self._seen: dict[str, str] = {}   # slug → original value

    def register(self, value: str) -> str:
        slug = _safe_slug(value)
        original = str(value or "").strip()
        if slug in self._seen and self._seen[slug] != original:
            raise ValueError(
                f"Slug collision in {self._namespace!r}: "
                f"{original!r} and {self._seen[slug]!r} both map to {slug!r}. "
                "Rename one value or extend _safe_slug to disambiguate."
            )
        self._seen[slug] = original
        return slug


# ============================================================
# BUILD POOLS
# ============================================================

def _build_pool(matches, category, flight, division):
    """
    Returns
    -------
    graph    : {entity: set of entities beaten (latest result only)}
    stats    : raw per-entity statistics for this bucket
    ts_pairs : list of (winner, loser) in chronological order

    Entity keys are (flight, division)-scoped via _entity_from_match.
    """
    # Stamp each match with its resolved division so _entity_from_match
    # can embed it into the key without a separate lookup argument.
    for m in matches:
        m["_resolved_division"] = division

    matches = sorted(
        matches,
        key=lambda m: _parse_dt(m.get("match_updated_at"))
    )

    latest_by_pair = {}
    ts_pairs = []

    stats = defaultdict(lambda: {
        "raw_wins":      0,
        "raw_losses":    0,
        "raw_matches":   0,
        "raw_last_date": datetime.min,
        "wins":          0,
        "losses":        0,
        "game_diff":     0,
        "set_diff":      0,
        "opponents":     set(),
        "vs":            {},
    })

    for m in matches:
        winner = _entity_from_match(m, category, "winner", flight)
        loser  = _entity_from_match(m, category, "loser",  flight)
        if winner is None or loser is None:
            continue

        dt = _parse_dt(m.get("match_updated_at"))
        wg, lg, ws, ls = _parse_score(m.get("set_score", ""))

        stats[winner]["raw_wins"] += 1
        stats[winner]["raw_matches"] += 1
        stats[winner]["raw_last_date"] = max(stats[winner]["raw_last_date"], dt)

        stats[loser]["raw_losses"] += 1
        stats[loser]["raw_matches"] += 1
        stats[loser]["raw_last_date"] = max(stats[loser]["raw_last_date"], dt)

        stats[winner]["game_diff"] += (wg - lg)
        stats[loser]["game_diff"] -= (wg - lg)
        stats[winner]["set_diff"] += (ws - ls)
        stats[loser]["set_diff"] -= (ws - ls)

        ts_pairs.append((winner, loser))

        pair = _pair_key(winner, loser)
        latest_by_pair[pair] = {"winner": winner, "loser": loser, "date": dt}

    graph = defaultdict(set)
    for rec in latest_by_pair.values():
        w, l = rec["winner"], rec["loser"]
        graph[w].add(l)
        stats[w]["wins"] += 1
        stats[l]["losses"] += 1
        stats[w]["opponents"].add(l)
        stats[l]["opponents"].add(w)
        stats[w]["vs"][l] = rec
        stats[l]["vs"][w] = rec

    for entity in stats:
        graph.setdefault(entity, set())

    return graph, stats, ts_pairs


def build_overall_stats(matches, category):
    """
    Build season-wide stats for the displayed W/L columns.
    Keyed by (flight+division)-scoped entity key so records
    match the entity keys used everywhere else.
    """
    stats = defaultdict(lambda: {
        "raw_wins":      0,
        "raw_losses":    0,
        "raw_matches":   0,
        "raw_last_date": datetime.min,
    })

    matches = sorted(
        matches,
        key=lambda m: _parse_dt(m.get("match_updated_at"))
    )

    for m in matches:
        flight   = str(m.get("flight") or "?")
        division = str(m.get("_resolved_division") or "?")
        winner   = _entity_from_match(m, category, "winner", flight)
        loser    = _entity_from_match(m, category, "loser",  flight)
        if winner is None or loser is None:
            continue

        dt = _parse_dt(m.get("match_updated_at"))

        stats[winner]["raw_wins"] += 1
        stats[winner]["raw_matches"] += 1
        stats[winner]["raw_last_date"] = max(stats[winner]["raw_last_date"], dt)

        stats[loser]["raw_losses"] += 1
        stats[loser]["raw_matches"] += 1
        stats[loser]["raw_last_date"] = max(stats[loser]["raw_last_date"], dt)

    return dict(stats)


# ============================================================
# MAIN ENGINE
# ============================================================

def build_graph_pools(matches, player_lookup=None, pair_lookup=None):
    """
    Build pools split by: gender -> category -> division -> flight.

    Entity keys are (flight, division)-scoped so the same player in
    different flights OR different divisions is always a distinct entity.
    """
    player_lookup = player_lookup or {}
    pair_lookup   = pair_lookup   or {}

    grouped = defaultdict(list)
    global_ts_rows = defaultdict(list)   # (gender, category) → [(dt, winner, loser)]

    for match in matches:
        gender   = match.get("gender")
        category = match.get("match_type", "").lower().strip()
        flight   = str(match.get("flight") or "?")

        if gender in ("Boys", "Girls") and category in ("singles", "doubles"):
            division = _match_division(match, category, player_lookup, pair_lookup)
            # Stamp division onto match so downstream helpers can read it.
            match["_resolved_division"] = division
            grouped[(gender, category, division, flight)].append(match)

            winner = _entity_from_match(match, category, "winner", flight)
            loser  = _entity_from_match(match, category, "loser",  flight)
            if winner is not None and loser is not None:
                dt = _parse_dt(match.get("match_updated_at"))
                global_ts_rows[(gender, category)].append((dt, winner, loser))

    pools = {
        g: {
            "singles": defaultdict(dict),
            "doubles": defaultdict(dict),
        }
        for g in ("Boys", "Girls")
    }

    for (gender, category, division, flight), group in grouped.items():
        graph, stats, ts_pairs = _build_pool(group, category, flight, division)
        pools[gender][category][division][flight] = {
            "graph":    graph,
            "stats":    stats,
            "ts_pairs": ts_pairs,
        }

    # Global match list per gender+category for the cross-bucket graph
    global_matches_by_key = defaultdict(list)
    for match in matches:
        gender   = match.get("gender")
        category = match.get("match_type", "").lower().strip()
        if gender in ("Boys", "Girls") and category in ("singles", "doubles"):
            global_matches_by_key[(gender, category)].append(match)

    for (gender, category), rows in global_ts_rows.items():
        rows.sort(key=lambda x: x[0])
        global_ts_pairs = [(w, l) for _, w, l in rows]
        pools[gender][category]["_global_ts_pairs"]   = global_ts_pairs
        pools[gender][category]["_global_ts_ratings"] = compute_trueskill(global_ts_pairs)

        # Global win-graph (latest result per pair, across ALL buckets).
        all_global_matches = sorted(
            global_matches_by_key[(gender, category)],
            key=lambda m: _parse_dt(m.get("match_updated_at"))
        )
        global_latest_by_pair = {}
        for m in all_global_matches:
            flight   = str(m.get("flight") or "?")
            winner   = _entity_from_match(m, category, "winner", flight)
            loser    = _entity_from_match(m, category, "loser",  flight)
            if winner is None or loser is None:
                continue
            pk = _pair_key(winner, loser)
            global_latest_by_pair[pk] = {"winner": winner, "loser": loser}

        global_graph = defaultdict(set)
        for rec in global_latest_by_pair.values():
            global_graph[rec["winner"]].add(rec["loser"])

        pools[gender][category]["_global_graph"] = global_graph

    return pools


def create_rankings(
    pools,
    category,
    overall_stats=None,
    player_lookup=None,
    pair_lookup=None,
):
    """
    Rank by TGRS inside each gender/division/flight bucket.

    player_lookup  keyed by bare player ID string
    pair_lookup    keyed by (id_a, id_b) tuple (no flight/division)
    overall_stats  keyed by (flight+division)-scoped entity key
    """
    player_lookup = player_lookup or {}
    pair_lookup   = pair_lookup   or {}
    overall_stats = overall_stats or {}

    rows = []

    genders = (
        (TARGET_GENDER,)
        if TARGET_GENDER
        else ("Boys", "Girls")
    )

    for gender in genders:
        category_pools = pools[gender][category]

        global_ts_pairs   = category_pools.get("_global_ts_pairs", [])
        global_ts_ratings = category_pools.get("_global_ts_ratings", {})
        global_graph      = category_pools.get("_global_graph", defaultdict(set))

        # Build per-entity opponent list for global SOS
        global_opponents_by_entity = defaultdict(list)
        for w, l in global_ts_pairs:
            global_opponents_by_entity[w].append(l)
            global_opponents_by_entity[l].append(w)

        for division, division_pools in category_pools.items():
            if str(division).startswith("_"):
                continue
            for flight, pool in division_pools.items():
                graph    = pool["graph"]
                stats    = pool["stats"]
                ts_pairs = pool["ts_pairs"]

                eligible = {
                    entity
                    for entity, info in stats.items()
                    if info["raw_matches"] >= MIN_MATCHES
                }
                if not eligible:
                    continue

                graph = {
                    node: {x for x in neighbors if x in eligible}
                    for node, neighbors in graph.items()
                    if node in eligible
                }
                for entity in eligible:
                    graph.setdefault(entity, set())

                if not graph:
                    continue

                global_reach = {
                    entity: _reachability_score(entity, global_graph)
                    for entity in eligible
                }

                local_reach = {
                    entity: _reachability_score(entity, graph)
                    for entity in graph
                }

                eligible = {e for e in eligible if local_reach.get(e, 0) > 0}
                if not eligible:
                    continue

                # Local TrueSkill (bucket-scoped, flight+division-scoped entities)
                local_ts_ratings = compute_trueskill(ts_pairs)

                def ts_mu_val(entity):
                    r = local_ts_ratings.get(entity)
                    return r.mu if r is not None else 0.0

                def ts_sigma_val(entity):
                    r = local_ts_ratings.get(entity)
                    return r.sigma if r is not None else 0.0

                def ts_cons(entity):
                    r = local_ts_ratings.get(entity)
                    return r.conservative if r is not None else 0.0

                eligible_ts_cons = [ts_cons(e) for e in eligible]
                field_avg_ts = _safe_mean(eligible_ts_cons, default=0.0)

                # Local opponent map for local SOS
                local_opponents_by_entity = defaultdict(list)
                for w, l in ts_pairs:
                    local_opponents_by_entity[w].append(l)
                    local_opponents_by_entity[l].append(w)

                sos          = {}
                local_sos    = {}
                quality_wins = {}
                base_tgrs    = {}   # scores WITHOUT h2h bonus

                for entity in eligible:
                    global_opp_strengths = [
                        global_ts_ratings[opp].conservative
                        for opp in global_opponents_by_entity.get(entity, [])
                        if opp in global_ts_ratings
                    ]
                    entity_global_sos = _safe_mean(global_opp_strengths, default=field_avg_ts)
                    sos[entity] = entity_global_sos

                    local_opp_strengths = [
                        local_ts_ratings[opp].conservative
                        for opp in local_opponents_by_entity.get(entity, [])
                        if opp in local_ts_ratings
                    ]
                    entity_local_sos = _safe_mean(local_opp_strengths, default=field_avg_ts)
                    local_sos[entity] = entity_local_sos

                    beaten_ratings = [
                        global_ts_ratings[opp].mu
                        for opp in global_graph.get(entity, set())
                        if opp in global_ts_ratings
                    ]
                    entity_quality_wins = _top_n_average(beaten_ratings, n=3, default=0.0)
                    quality_wins[entity] = entity_quality_wins

                    g_ts_r = global_ts_ratings.get(entity)
                    global_ts_mu_val = g_ts_r.mu if g_ts_r is not None else 0.0

                    base_tgrs[entity] = (
                        TGRS_TS_MU_WEIGHT         * global_ts_mu_val
                        + TGRS_SOS_WEIGHT          * entity_global_sos
                        + TGRS_REACH_WEIGHT        * global_reach.get(entity, 0)
                        + TGRS_QUALITY_WINS_WEIGHT * entity_quality_wins
                        + TGRS_LOCAL_SOS_WEIGHT    * entity_local_sos
                        + TGRS_LOCAL_TS_MU_WEIGHT  * ts_mu_val(entity)
                        + TGRS_LOCAL_REACH_WEIGHT  * local_reach.get(entity, 0)
                    )

                # ── Head-to-head bonus ───────────────────────────────────────
                # H2H bonus is computed from BASE scores only (before any bonus
                # is applied), so there is no circular dependency: the bonus
                # for beating player X is always TGRS_H2H_WEIGHT * X's base
                # score, regardless of who X beat.
                tgrs_score = {entity: score for entity, score in base_tgrs.items()}
                for entity in eligible:
                    for beaten in graph.get(entity, set()):
                        if beaten in eligible:
                            tgrs_score[entity] += TGRS_H2H_WEIGHT * base_tgrs[beaten]

                ordered = sorted(
                    eligible,
                    key=lambda x: (
                        -tgrs_score.get(x, -9999.0),
                        -global_reach.get(x, 0),
                        -local_reach.get(x, 0),
                        -quality_wins.get(x, 0.0),
                        -ts_mu_val(x),
                        -ts_cons(x),
                        _entity_label(x),
                    )
                )

                rank = 1
                i = 0
                n = len(ordered)

                while i < n:
                    current_score = tgrs_score[ordered[i]]

                    tie_group = []
                    j = i
                    while j < n and abs(tgrs_score[ordered[j]] - current_score) < 1e-12:
                        tie_group.append(ordered[j])
                        j += 1

                    for entity in tie_group:
                        bucket_info = stats[entity]
                        global_info = overall_stats.get(entity, bucket_info)

                        global_ts_r = global_ts_ratings.get(entity)
                        local_ts_r  = local_ts_ratings.get(entity)
                        score       = tgrs_score.get(entity, 0.0)

                        bare = _bare_player_id(entity)

                        row = {
                            "rank":               rank,
                            "gender":             gender,
                            "division":           division,
                            "flight":             flight,
                            "rating":             round(score, 4),
                            "TGRS":               round(score, 4),
                            "reachability":       global_reach.get(entity, 0),
                            "local_reachability": local_reach.get(entity, 0),
                            "sos":                round(sos.get(entity, field_avg_ts), 4),
                            "local_sos":          round(local_sos.get(entity, field_avg_ts), 4),
                            "quality_wins":       round(quality_wins.get(entity, 0.0), 4),
                            "ts_mu":              round(global_ts_r.mu, 4) if global_ts_r else None,
                            "ts_sigma":           round(global_ts_r.sigma, 4) if global_ts_r else None,
                            "ts_rating":          round(global_ts_r.conservative, 4) if global_ts_r else None,
                            "local_ts_mu":        round(local_ts_r.mu, 4) if local_ts_r else None,
                            "matches_played":     global_info["raw_matches"],
                            "wins":               global_info["raw_wins"],
                            "losses":             global_info["raw_losses"],
                            "last_match_date": (
                                global_info["raw_last_date"].strftime("%Y-%m-%d")
                                if global_info["raw_last_date"] != datetime.min
                                else ""
                            ),
                        }

                        if category == "singles":
                            meta = player_lookup.get(bare, {})
                            row.update({
                                "name":     meta.get("name", bare),
                                "school":   meta.get("school", ""),
                                "division": division,
                            })
                        else:
                            meta = pair_lookup.get(bare, {})
                            row.update({
                                "pair_name": meta.get("pair_name", " / ".join(bare)),
                                "school":    meta.get("school", ""),
                                "division":  division,
                            })

                        rows.append(row)

                    rank += len(tie_group)
                    i = j

    return rows
