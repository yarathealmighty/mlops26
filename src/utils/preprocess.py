"""Preprocessing utilities for the League of Legends ranked matches dataset.

Everything here is lifted from notebooks/EDA.ipynb. Typical use from a notebook:

    import sys; sys.path.append("..")
    from src.preprocess import (
        load_raw_tables, filter_matches, build_player_lookup,
        time_split, build_champion_meta,
    )

    tables = load_raw_tables("../data/raw")
    matches = filter_matches(tables["matches"])
    player_lookup = build_player_lookup(
        tables["participants"], tables["champs"],
        match_ids=set(matches["id"]), verbose=True,
    )
    train_ids, test_ids = time_split(matches)
    champion_meta = build_champion_meta(
        player_lookup, tables["stats"], matches, match_ids=train_ids
    )
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# assumes this file lives in <repo>/src/, so the repo root is one level up
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ITEM_COLS = [f"item{i}" for i in range(1, 7)]

AVG_COLS = {
    # raw column     -> name in the meta table
    "kills": "kills", "deaths": "deaths", "assists": "assists",
    "totdmgdealt": "dmg_total", "magicdmgdealt": "dmg_magic",
    "physicaldmgdealt": "dmg_phys", "truedmgdealt": "dmg_true",
    "totdmgtochamp": "champ_dmg_total", "magicdmgtochamp": "champ_dmg_magic",
    "physdmgtochamp": "champ_dmg_phys", "truedmgtochamp": "champ_dmg_true",
    "totheal": "heal",
    "dmgtoobj": "dmg_obj", "dmgtoturrets": "dmg_turret",
    "totdmgtaken": "taken_total", "magicdmgtaken": "taken_magic",
    "physdmgtaken": "taken_phys", "truedmgtaken": "taken_true",
    "goldspent": "gold_spent",
    "champlvl": "champ_lvl",
    "totcctimedealt": "cc_time",
    "wardsplaced": "wards_placed", "wardskilled": "wards_killed",
}

__all__ = [
    "RAW_DIR", "ITEM_COLS", "AVG_COLS",
    "load_raw_tables", "filter_matches", "build_player_lookup",
    "time_split", "build_champion_meta",
]


def load_raw_tables(raw_dir: str | Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read the raw CSVs. stats1 and stats2 are two halves of one table, so they are stacked."""
    raw_dir = Path(raw_dir)
    stats = pd.concat(
        [
            pd.read_csv(raw_dir / "stats1.csv"),
            pd.read_csv(raw_dir / "stats2.csv", low_memory=False),
        ],
        ignore_index=True,
    )
    return {
        "champs": pd.read_csv(raw_dir / "champs.csv"),
        "matches": pd.read_csv(raw_dir / "matches.csv"),
        "participants": pd.read_csv(raw_dir / "participants.csv"),
        "stats": stats,
        "teambans": pd.read_csv(raw_dir / "teambans.csv"),
    }


def filter_matches(matches: pd.DataFrame, platform: str = "EUW1") -> pd.DataFrame:
    """Keep matches from a single platform (only EUW matches checked, drops ~10k rows)."""
    return matches[matches["platformid"] == platform]


def build_player_lookup(
    participants: pd.DataFrame,
    champs: pd.DataFrame,
    match_ids=None,
    verbose: bool = False,
) -> pd.DataFrame:
    """One row per player-game: team, summoner spells, position and champion name.

    If `match_ids` is given, only players from those matches are kept.
    """
    champ_names = champs.set_index("id")["name"]

    # DUO_SUPPORT / DUO_CARRY on the BOT lane -> support / adc; other lanes keep their own name
    position = participants["role"].map({"DUO_SUPPORT": "support", "DUO_CARRY": "adc"})
    not_bot = participants["position"] != "BOT"
    position = position.where(~not_bot, participants["position"].str.lower())

    player_lookup = pd.DataFrame({
        "player_id":    participants["id"],
        "match_id":     participants["matchid"],
        "team":         np.where(participants["player"] < 6, "blue", "red"),
        "ss1":          participants["ss1"],
        "ss2":          participants["ss2"],
        "position":     position,
        "played_champ": participants["championid"].map(champ_names),
    })

    if match_ids is not None:
        before = len(player_lookup)
        player_lookup = (
            player_lookup[player_lookup["match_id"].isin(match_ids)]
            .reset_index(drop=True)
        )
        if verbose:
            print(f"kept {len(player_lookup):,} of {before:,} rows "
                  f"({len(player_lookup) / before:.1%})")
    return player_lookup


def time_split(matches: pd.DataFrame, train_frac: float = 0.8) -> tuple[set, set]:
    """Chronological split of match ids (oldest -> train, newest -> test).

    Build champion stats from the train ids only, otherwise each match's own
    result leaks into its features.
    """
    order = matches.sort_values("creation")["id"]
    cut = int(len(order) * train_frac)
    return set(order[:cut]), set(order[cut:])


def build_champion_meta(
    player_lookup: pd.DataFrame,
    stats: pd.DataFrame,
    matches: pd.DataFrame,
    match_ids=None,
    prior_games: int = 100,
) -> pd.DataFrame:
    """One row per champion, built only from the matches in `match_ids`."""
    pl = player_lookup
    if match_ids is not None:
        pl = pl[pl["match_id"].isin(match_ids)]

    df = (
        pl[["player_id", "match_id", "played_champ", "position"]]
        .merge(stats[["id", "win", *ITEM_COLS, *AVG_COLS]],
               left_on="player_id", right_on="id", how="inner")
        .merge(matches[["id", "duration"]].rename(columns={"id": "match_id"}),
               on="match_id", how="left")
        .dropna(subset=["played_champ"])
    )
    df["win"] = df["win"].astype(float)
    g = df.groupby("played_champ")

    # rarer champs handled with smoothing
    meta = pd.DataFrame({"games": g.size()})
    meta["pick_rate"] = meta["games"] / df["match_id"].nunique()
    meta["winrate"] = g["win"].mean()
    meta["winrate_smooth"] = (
        (g["win"].sum() + prior_games * df["win"].mean()) / (meta["games"] + prior_games)
    )

    meta = meta.join(g[list(AVG_COLS)].mean().rename(columns=AVG_COLS))

    # kda needs spec cuz 0 death
    tot = g[["kills", "deaths", "assists"]].sum()
    meta["kda"] = (tot["kills"] + tot["assists"]) / tot["deaths"].clip(lower=1)

    for kind in ("magic", "phys", "true"):
        meta[f"champ_dmg_{kind}_share"] = meta[f"champ_dmg_{kind}"] / meta["champ_dmg_total"]
    meta["gold_per_kill"] = meta["gold_spent"] / meta["kills"].replace(0, np.nan)

    filled = df[ITEM_COLS].fillna(0).ne(0)
    filled.columns = [f"has_item_{i}" for i in range(1, 7)]
    meta = meta.join(filled.groupby(df["played_champ"]).mean())

    # is item in slot n
    meta["items_filled"] = meta[[f"has_item_{i}" for i in range(1, 7)]].sum(axis=1)

    pos = pd.crosstab(df["played_champ"], df["position"], normalize="index")
    meta = meta.join(pos.add_prefix("pos_"))

    # scaling: winrate in the shortest vs longest third of games, might need a better approach
    length = pd.qcut(df["duration"], 3, labels=["short", "mid", "long"])
    by_len = (df.groupby(["played_champ", length], observed=True)["win"]
                .mean().unstack().add_prefix("winrate_"))
    meta = meta.join(by_len)
    meta["scaling"] = meta["winrate_long"] - meta["winrate_short"]
    return meta