import os
import re
import pandas as pd
import numpy as np
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_FILES = [
    (ROOT_DIR / "data/raw/ATP_Singles_pbp.csv", "ATP"),
    (ROOT_DIR / "data/raw/ATP_Singles_pbpx.csv", "ATP"),
    (ROOT_DIR / "data/raw/WTA_Singles_pbp.csv", "WTA"),
    (ROOT_DIR / "data/raw/WTA_Singles_pbpx.csv", "WTA"),
]

OUTPUT_PATH = ROOT_DIR / "data/processed/cleaned_matches.parquet"

def parse_recorded_sets(score_str):
    """
    Parse recorded score string (for example, '6-4 3-6 7-6(4)') into a list of (p1_games, p2_games) tuples.
    Returns None if parsing failed
    """
    if not isinstance(score_str, str):
        return None

    # If a set ends in a tie break, the score is displayed as 7-6(4), we want to remove the extra parenthesis in the score in order to parse the string
    # A string in cleaned will look like "6-4 3-6 7-6"
    cleaned = re.sub(r"\(\d+\)", "", score_str).strip()
    sets = []
    for token in cleaned.split():
        m = re.match(r"^(\d+)-(\d+)$", token)
        if m:
            sets.append((int(m.group(1)), int(m.group(2))))

    if sets:
        return sets
    else:
        return None


def normalize_pbp_string(pbp_str):
    """
    Normalize pbp and pbpx strings into standard Sackmann tokens:
    'S' (server won), 'R' (returner won), ';' (end of game), '.' (end of set), '/' (tiebreak server switch).
    In pbpx strings, point outcomes can be encoded with serve/shot details:
      - Server wins: S, A (ace), W (winner on serve) -> 'S'
      - Returner wins: R, D (double fault) -> 'R'
    """
    if not isinstance(pbp_str, str):
        return ""
    out = []
    for ch in pbp_str:
        if ch in ("S", "A"):
            out.append("S")
        elif ch in ("R", "D"):
            out.append("R")
        elif ch in (";", ".", "/"):
            out.append(ch)
    return "".join(out)


def decode_match_points(pbp_norm, first_server_idx):
    """
    Decode a normalized point string assuming `first_server_idx` (0 for server1, 1 for server2)
    serves the very first game of the match.

    Returns:
        points: list of dicts with server (0 or 1), winner (0 or 1), is_tiebreak (bool),
                and preceding score state (pts_s, pts_r, games_s, games_r, sets_s, sets_r).
        reconstructed_sets: list of (p1_games, p2_games) tuples from server1 / server2 perspective.
    """
    points = []
    reconstructed_sets = []

    sets_won = [0, 0]
    games = [0, 0]
    pts = [0, 0]

    # Track who serves the current regular game (alternates every game, including after tiebreaks)
    game_server = first_server_idx
    in_tiebreak = False
    tb_first_server = first_server_idx
    tb_points_played = 0

    for ch in pbp_norm:
        if ch == ";":
            # End of regular game: determine game winner from point count
            if pts[0] > pts[1]:
                games[0] += 1
            elif pts[1] > pts[0]:
                games[1] += 1
            pts = [0, 0]
            game_server = 1 - game_server
            in_tiebreak = False
            tb_points_played = 0

        elif ch == ".":
            # End of set: handle tiebreak or final game if not closed by ';'
            if in_tiebreak or (pts[0] > 0 or pts[1] > 0):
                if pts[0] > pts[1]:
                    games[0] += 1
                elif pts[1] > pts[0]:
                    games[1] += 1
                # Next set starts with the player who did NOT serve first in the tiebreak
                if in_tiebreak:
                    game_server = 1 - tb_first_server
                else:
                    game_server = 1 - game_server

            reconstructed_sets.append((games[0], games[1]))
            if games[0] > games[1]:
                sets_won[0] += 1
            elif games[1] > games[0]:
                sets_won[1] += 1

            games = [0, 0]
            pts = [0, 0]
            in_tiebreak = False
            tb_points_played = 0

        elif ch == "/":
            # Tiebreak service change marker
            continue

        elif ch in ("S", "R"):
            # Detect tiebreak start at 6-6 (standard tiebreak set)
            if not in_tiebreak and games[0] == 6 and games[1] == 6 and pts == [0, 0]:
                in_tiebreak = True
                tb_first_server = game_server
                tb_points_played = 0

            # Determine active server for this point
            if in_tiebreak:
                # First point served by tb_first_server, then alternates every 2 points
                if tb_points_played == 0:
                    curr_server = tb_first_server
                else:
                    pair_idx = (tb_points_played - 1) // 2
                    curr_server = (1 - tb_first_server) if (pair_idx % 2 == 0) else tb_first_server
            else:
                curr_server = game_server

            curr_returner = 1 - curr_server
            winner = curr_server if ch == "S" else curr_returner

            points.append({
                "server": curr_server,
                "winner": winner,
                "is_tiebreak": in_tiebreak,
                "pts_s": pts[curr_server],
                "pts_r": pts[curr_returner],
                "games_s": games[curr_server],
                "games_r": games[curr_returner],
                "sets_s": sets_won[curr_server],
                "sets_r": sets_won[curr_returner],
            })

            pts[winner] += 1
            if in_tiebreak:
                tb_points_played += 1

    # Flush trailing incomplete set 
    if games[0] > 0 or games[1] > 0 or pts[0] > 0 or pts[1] > 0:
        if pts[0] > pts[1]:
            games[0] += 1
        elif pts[1] > pts[0]:
            games[1] += 1
        reconstructed_sets.append((games[0], games[1]))

    return points, reconstructed_sets


def main():
    # Load raw files
    dfs = []
    for path, tour in RAW_FILES:
        df = pd.read_csv(path)
        df["tour"] = tour
        dfs.append(df)
    raw_df = pd.concat(dfs, ignore_index=True)

    # Standardize column names if needed (check common TennisVisuals headers)
    # Expected columns: date, server1, server2, score, pbp, surface, tour, draw, winner
    pbp_col = "pbp" if "pbp" in raw_df.columns else [c for c in raw_df.columns if "pbp" in c.lower()][0]

    # Remove exact duplicate rows on (date, ordered player names, point string)
    before_dedup = len(raw_df)
    df = raw_df.drop_duplicates(subset=["date", "server1", "server2", pbp_col]).copy()

    # Remove rows that ended in retirements
    before_ret = len(df)
    score_upper = df["score"].astype(str).str.upper()
    is_ret = score_upper.str.contains(r"\bRET\b", regex=True)
    if "status" in df.columns:
        is_ret = is_ret | (df["status"].astype(str).str.upper().str.strip() == "RET")
    df = df[~is_ret].copy()

    # Map 'Davis' and 'Fed' surface labels to 'Unknown'
    if "surface" in df.columns:
        df["surface"] = df["surface"].fillna("Unknown").replace({"Davis": "Unknown", "Fed": "Unknown"})

    if "draw" in df.columns:
        df["draw"] = df["draw"].fillna("Unknown")

    # Decode points and verify first server assignment
    decoded_records = []
    reversed_count = 0
    unmatched_count = 0

    for idx, row in df.iterrows():
        pbp_norm = normalize_pbp_string(row[pbp_col])
        rec_sets = parse_recorded_sets(row["score"])

        pts_0, sets_0 = decode_match_points(pbp_norm, first_server_idx=0)
        pts_1, sets_1 = decode_match_points(pbp_norm, first_server_idx=1)

        if rec_sets == sets_0:
            chosen_pts = pts_0
            first_server = 0
        elif rec_sets == sets_1:
            chosen_pts = pts_1
            first_server = 1
            reversed_count += 1
        else:
            # Fallback prefix match (e.g., for the 2 administrative default matches)
            if rec_sets and sets_0[: len(rec_sets)] == rec_sets:
                chosen_pts = pts_0
                first_server = 0
            elif rec_sets and sets_1[: len(rec_sets)] == rec_sets:
                chosen_pts = pts_1
                first_server = 1
                reversed_count += 1
            else:
                chosen_pts = pts_0
                first_server = 0
                unmatched_count += 1

        # Relabel so Player A is always the actual first server and Player B is second
        if first_server == 0:
            player_a, player_b = row["server1"], row["server2"]
        else:
            player_a, player_b = row["server2"], row["server1"]
            for p in chosen_pts:
                p["server"] = 1 - p["server"]
                p["winner"] = 1 - p["winner"]

        decoded_records.append({
            "match_id": idx,
            "date": pd.to_datetime(row["date"]),
            "tour": row["tour"],
            "surface": row.get("surface", "Unknown"),
            "draw": row.get("draw", "Unknown"),
            "player_a": player_a,
            "player_b": player_b,
            "first_server_reversed": bool(first_server == 1),
            "num_points": len(chosen_pts),
            "points": chosen_pts,
        })

    out_df = pd.DataFrame(decoded_records)
    total_pts = out_df["num_points"].sum()
    print(f"First server reversed in: {reversed_count} matches ({reversed_count / len(out_df):.1%}) (Proposal: 16,093 / 50.1%)")
    print(f"Unmatched score checks: {unmatched_count}")
    print(f"Total decoded points: {total_pts:,} (Proposal: 4,760,298)")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df["surface"] = out_df["surface"].astype(str)
    out_df["draw"] = out_df["draw"].astype(str)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()