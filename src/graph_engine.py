# graph_engine.py

from collections import defaultdict
from datetime import datetime
from statistics import mean

from config import MIN_MATCHES, TARGET_GENDER
from trueskill_engine import compute_trueskill

# ============================================================
# TUNABLE WEIGHTS (unchanged)
# ============================================================
TGRS_LOCAL_REACH_WEIGHT   = 1.25
TGRS_TS_MU_WEIGHT         = (3.00+1+3+2+1+0.4+1+1)/28.25 
TGRS_SOS_WEIGHT           = (3.20+2.25+0.10+4.5)/28.25
TGRS_REACH_WEIGHT         = (10.00+15)/28.25
TGRS_QUALITY_WINS_WEIGHT  = (3.00+3.5+5)/28.25
TGRS_LOCAL_SOS_WEIGHT     = 0.2+0.1+0.1
TGRS_LOCAL_TS_MU_WEIGHT   = (0.30+1+0.5+1+4.25)/28.25

# ============================================================
# UTILITIES (unchanged)
# ============================================================
def _parse_dt(raw):
    if not raw:
        return datetime.min
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime.min

def _entity_from_match(match, category, side):
    ids = match.get(f"{side}_player_ids") or []
    if category == "singles":
        if len(ids) != 1:
            return None
        return str(ids[0])
    if len(ids) != 2:
        return None
    return tuple(sorted(str(x) for x in ids))

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
    if isinstance(entity, tuple):
        return " / ".join(entity)
    return str(entity)

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

# ============================================================
# BUILD POOLS
# ============================================================
def _build_pool(matches, category):
    """
    Returns
    -------
    graph       : {entity: set of entities beaten (latest result only)}
    stats       : raw per-entity statistics for this bucket
    ts_pairs    : list of (winner, loser) in chronological order
                  used by compute_trueskill(); includes ALL matches in bucket.
    """
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

    # Process each match in chronological order
    for m in matches:
        winner = _entity_from_match(m, category, "winner")
        loser = _entity_from_match(m, category, "loser")
        if winner is None or loser is None:
            continue

        dt = _parse_dt(m.get("match_updated_at"))
        wg, lg, ws, ls = _parse_score(m.get("set_score", ""))

        # Update raw stats for the bucket (flight-specific)
        stats[winner]["raw_wins"] += 1
        stats[winner]["raw_matches"] += 1
        stats[winner]["raw_last_date"] = max(stats[winner]["raw_last_date"], dt)
        stats[loser]["raw_losses"] += 1
        stats[loser]["raw_matches"] += 1
        stats[loser]["raw_last_date"] = max(stats[loser]["raw_last_date"], dt)

        # Update game/set differential (bucket-specific)
        stats[winner]["game_diff"] += (wg - lg)
        stats[loser]["game_diff"] -= (wg - lg)
        stats[winner]["set_diff"] += (ws - ls)
        stats[loser]["set_diff"] -= (ws - ls)

        ts_pairs.append((winner, loser))

        # Keep only the latest result per unique pair (for head-to-head graph)
        pair = _pair_key(winner, loser)
        latest_by_pair[pair] = {"winner": winner, "loser": loser, "date": dt}

    # Build directed win-graph using only latest results per pair
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
    Build a season-wide stats table for the given category.
    Used only for the displayed record columns.
    """
    matches = sorted(
        matches,
        key=lambda m: _parse_dt(m.get("match_updated_at"))
    )

    stats = defaultdict(lambda: {
        "raw_wins":      0,
        "raw_losses":    0,
        "raw_matches":   0,
        "raw_last_date": datetime.min,
    })

    for m in matches:
        winner = _entity_from_match(m, category, "winner")
        loser = _entity_from_match(m, category, "loser")
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
    Build pools split by:
      gender -> category -> division -> flight
    Also builds a global TrueSkill pool per gender/category.
    """
    # (This function remains unchanged, as it only groups matches.)
    # ...
    pass  # (Omitted for brevity; same as before)

def create_rankings(
    pools,
    category,
    overall_stats=None,
    player_lookup=None,
    pair_lookup=None,
):
    """
    Rank by TGRS.  Outputs one row per player/pair in each flight.
    """
    player_lookup = player_lookup or {}
    pair_lookup = pair_lookup or {}
    overall_stats = overall_stats or {}

    rows = []
    genders = ( (TARGET_GENDER,) if TARGET_GENDER else ("Boys", "Girls") )

    for gender in genders:
        category_pools = pools[gender][category]

        # Build per-entity opponent list for global SOS
        global_ts_pairs   = category_pools.get("_global_ts_pairs", [])
        global_ts_ratings = category_pools.get("_global_ts_ratings", {})
        global_graph      = category_pools.get("_global_graph", defaultdict(set))
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

                # Only keep entities with >= MIN_MATCHES in this flight
                eligible = {
                    entity
                    for entity, info in stats.items()
                    if info["raw_matches"] >= MIN_MATCHES
                }
                if not eligible:
                    continue

                # Prune graph to eligible players
                graph = {
                    node: {x for x in neighbors if x in eligible}
                    for node, neighbors in graph.items()
                    if node in eligible
                }
                for entity in eligible:
                    graph.setdefault(entity, set())
                if not graph:
                    continue

                # Compute reachability (global vs local)
                global_reach = {entity: _reachability_score(entity, global_graph) for entity in eligible}
                local_reach  = {entity: _reachability_score(entity, graph) for entity in graph}

                # Local TrueSkill (bucket-scoped)
                local_ts_ratings = compute_trueskill(ts_pairs)
                def ts_mu_val(entity):
                    r = local_ts_ratings.get(entity); return r.mu if r else 0.0
                def ts_cons(entity):
                    r = local_ts_ratings.get(entity); return r.conservative if r else 0.0

                eligible_ts_cons = [ts_cons(e) for e in eligible]
                field_avg_ts = _safe_mean(eligible_ts_cons, default=0.0)

                # Build local opponent map for local SOS
                local_opponents_by_entity = defaultdict(list)
                for w, l in ts_pairs:
                    local_opponents_by_entity[w].append(l)
                    local_opponents_by_entity[l].append(w)

                sos          = {}
                local_sos    = {}
                quality_wins = {}
                tgrs_score   = {}

                for entity in eligible:
                    # Global SOS (conservative TS ratings of all opponents played globally)
                    global_opp_strengths = [
                        global_ts_ratings[opp].conservative
                        for opp in global_opponents_by_entity.get(entity, [])
                        if opp in global_ts_ratings
                    ]
                    entity_global_sos = _safe_mean(global_opp_strengths, default=field_avg_ts)
                    sos[entity] = entity_global_sos

                    # Local SOS (conservative TS ratings of opponents in this bucket)
                    local_opp_strengths = [
                        local_ts_ratings[opp].conservative
                        for opp in local_opponents_by_entity.get(entity, [])
                        if opp in local_ts_ratings
                    ]
                    entity_local_sos = _safe_mean(local_opp_strengths, default=field_avg_ts)
                    local_sos[entity] = entity_local_sos

                    # Quality wins (avg of top-3 beaten opponents' global mu)
                    beaten_ratings = [
                        global_ts_ratings[opp].mu
                        for opp in global_graph.get(entity, set())
                        if opp in global_ts_ratings
                    ]
                    entity_quality_wins = _top_n_average(beaten_ratings, n=3, default=0.0)
                    quality_wins[entity] = entity_quality_wins

                    g_ts_r = global_ts_ratings.get(entity)
                    global_ts_mu_val = g_ts_r.mu if g_ts_r else 0.0

                    tgrs_score[entity] = (
                        TGRS_TS_MU_WEIGHT         * global_ts_mu_val +
                        TGRS_SOS_WEIGHT          * entity_global_sos +
                        TGRS_REACH_WEIGHT        * global_reach.get(entity, 0) +
                        TGRS_QUALITY_WINS_WEIGHT * entity_quality_wins +
                        TGRS_LOCAL_SOS_WEIGHT    * entity_local_sos +
                        TGRS_LOCAL_TS_MU_WEIGHT  * ts_mu_val(entity) +
                        TGRS_LOCAL_REACH_WEIGHT  * local_reach.get(entity, 0)
                    )

                # Sort and assign ranks (handling ties)
                ordered = sorted(
                    graph.keys(),
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
                        score = tgrs_score.get(entity, 0.0)

                        row = {
                            "rank":           rank,
                            "gender":         gender,
                            "division":       division,
                            "flight":         flight,
                            "rating":         round(score, 4),
                            "TGRS":           round(score, 4),
                            "reachability":        global_reach.get(entity, 0),
                            "local_reachability":  local_reach.get(entity, 0),
                            "sos":            round(sos.get(entity, field_avg_ts), 4),
                            "local_sos":      round(local_sos.get(entity, field_avg_ts), 4),
                            "quality_wins":   round(quality_wins.get(entity, 0.0), 4),
                            "ts_mu":          round(global_ts_r.mu, 4) if global_ts_r else None,
                            "ts_sigma":       round(global_ts_r.sigma, 4) if global_ts_r else None,
                            "ts_rating":      round(global_ts_r.conservative, 4) if global_ts_r else None,
                            "local_ts_mu":    round(local_ts_r.mu, 4) if local_ts_r else None,
                            # Use flight stats for wins/losses and matches
                            "matches_played": bucket_info["raw_matches"],
                            "wins":           bucket_info["raw_wins"],
                            "losses":         bucket_info["raw_losses"],
                            # Expose overall totals as raw_wins/losses columns
                            "raw_wins":       global_info["raw_wins"],
                            "raw_losses":     global_info["raw_losses"],
                            "last_match_date": (
                                global_info["raw_last_date"].strftime("%Y-%m-%d")
                                if global_info["raw_last_date"] != datetime.min
                                else ""
                            ),
                        }

                        if category == "singles":
                            meta = player_lookup.get(entity, {})
                            row.update({
                                "name":     meta.get("name", entity),
                                "school":   meta.get("school", ""),
                                "division": division,
                            })
                        else:
                            meta = pair_lookup.get(entity, {})
                            row.update({
                                "pair_name": meta.get("pair_name", " / ".join(entity)),
                                "school":    meta.get("school", ""),
                                "division":  division,
                            })

                        rows.append(row)

                    rank += len(tie_group)
                    i = j

    return rows
