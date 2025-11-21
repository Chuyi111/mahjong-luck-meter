#!/usr/bin/env python
import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Iterable, Tuple

# If you want to parse the original proto instead of JSON dicts:
# from google.protobuf.json_format import Parse
# import ms.protocol_pb2 as pb


# ====== Tile / Shanten / Ukeire utilities ======

# In Mahjong Soul the tile "instance" IDs are typically 0..135 and
# tile-type = instance_id // 4 gives a number 0..33:
#   0..8   = 1m..9m
#   9..17  = 1p..9p
#   18..26 = 1s..9s
#   27..33 = honors
def tile_instance_to_type(tile_id: int) -> int:
    return tile_id // 4


def hand_types_to_counts(tiles: List[int]) -> List[int]:
    """
    Convert a list of tile types 0..33 into 34-length count array.
    """
    counts = [0] * 34
    for t in tiles:
        if 0 <= t < 34:
            counts[t] += 1
    return counts


def is_terminal_or_honor(t: int) -> bool:
    # 0..8 man, 9..17 pin, 18..26 sou, 27..33 honors
    if t >= 27:
        return True
    num = t % 9
    return num == 0 or num == 8  # 1 or 9 in that suit


# --- Shanten calculation (standard hand, chiitoi, kokushi) ---


def shanten_normal(counts: List[int]) -> int:
    """
    Standard 4-meld + 1-pair hand shanten.
    This is a reasonably standard DFS-based shanten.
    """
    min_shanten = 8  # maximum possible shanten (13 tiles) is 8

    # copy so we can mutate
    c = counts[:]

    def dfs(idx: int, melds: int, pairs: int, taatsu: int):
        nonlocal min_shanten, c

        # theoretical max melds/pairs we can still form
        # Quick pruning: if we already can't beat current best, stop.
        # (optional micro-optimization; safe to omit)
        # compute current shanten upper bound
        m = melds
        t = taatsu
        p = pairs
        if m > 4:
            m = 4
        if m + t > 4:
            t = 4 - m

        sh = 8 - m * 2 - t - p
        if sh >= min_shanten:
            return

        # skip exhausted tiles
        while idx < 34 and c[idx] == 0:
            idx += 1

        if idx >= 34:
            # Recompute final shanten again (with clamping)
            m = melds
            t = taatsu
            p = pairs
            if m > 4:
                m = 4
            if m + t > 4:
                t = 4 - m
            sh = 8 - m * 2 - t - p
            min_shanten = min(min_shanten, sh)
            return

        # Try meld (triplet)
        if c[idx] >= 3:
            c[idx] -= 3
            dfs(idx, melds + 1, pairs, taatsu)
            c[idx] += 3

        # Try sequence meld (only for suits)
        if idx < 27:
            s = idx % 9
            if s <= 6 and c[idx] >= 1 and c[idx + 1] >= 1 and c[idx + 2] >= 1:
                c[idx] -= 1
                c[idx + 1] -= 1
                c[idx + 2] -= 1
                dfs(idx, melds + 1, pairs, taatsu)
                c[idx] += 1
                c[idx + 1] += 1
                c[idx + 2] += 1

        # Try pair
        if c[idx] >= 2:
            c[idx] -= 2
            dfs(idx + 1, melds, pairs + 1, taatsu)
            c[idx] += 2

        # Try taatsu (incomplete melds)
        # Closed kan counted as meld already above (triplet).
        if idx < 27:
            s = idx % 9
            if s <= 7 and c[idx] >= 1 and c[idx + 1] >= 1:
                c[idx] -= 1
                c[idx + 1] -= 1
                dfs(idx, melds, pairs, taatsu + 1)
                c[idx] += 1
                c[idx + 1] += 1
            if s <= 6 and c[idx] >= 1 and c[idx + 2] >= 1:
                c[idx] -= 1
                c[idx + 2] -= 1
                dfs(idx, melds, pairs, taatsu + 1)
                c[idx] += 1
                c[idx + 2] += 1

        # Skip tile
        dfs(idx + 1, melds, pairs, taatsu)

    dfs(0, 0, 0, 0)
    return min_shanten


def shanten_chiitoi(counts: List[int]) -> int:
    """
    7 pairs shanten.
    """
    pairs = 0
    singles = 0
    for n in counts:
        if n >= 2:
            pairs += 1
        elif n == 1:
            singles += 1
    # base shanten for chiitoi with k pairs = 6 - k,
    # but can't use more than 7 distinct tiles
    return 6 - pairs + max(0, 7 - pairs - singles)


def shanten_kokushi(counts: List[int]) -> int:
    """
    Kokushi musou shanten.
    """
    terminals = 0
    pair = 0
    for t in range(34):
        if is_terminal_or_honor(t) and counts[t] > 0:
            terminals += 1
            if counts[t] >= 2:
                pair = 1
    return 13 - terminals - pair


def shanten_number(tiles: List[int]) -> int:
    """
    Overall shanten, minimum of (normal, chiitoi, kokushi).
    """
    counts = hand_types_to_counts(tiles)
    s_normal = shanten_normal(counts)
    s_chiitoi = shanten_chiitoi(counts)
    s_kokushi = shanten_kokushi(counts)
    return min(s_normal, s_chiitoi, s_kokushi)


def ukeire_tiles(tiles: List[int]) -> List[int]:
    """
    Return list of tile types that strictly improve shanten
    if drawn next (assuming 4-copy limit).
    """
    base_shanten = shanten_number(tiles)
    counts = hand_types_to_counts(tiles)
    result = []

    for t in range(34):
        if counts[t] >= 4:
            continue
        # simulate draw
        tiles.append(t)
        new_sh = shanten_number(tiles)
        tiles.pop()
        if new_sh < base_shanten:
            result.append(t)

    return result


# ====== In-memory models for analysis ======

@dataclass
class DrawStep:
    # state just BEFORE the draw (tiles in hand, type 0..33)
    tiles_before: List[int]
    # tile type drawn (0..33)
    tile_drawn: int


@dataclass
class RiichiInstance:
    # Whether this riichi ultimately resulted in ippatsu
    ippatsu_win: bool = False
    # Whether it was a win at all (for ippatsu denominator)
    win: bool = False


@dataclass
class WinEvent:
    winner_seat: int
    loser_seat: Optional[int]  # None for tsumo
    # points lost/gained for hero, already scored from log
    hero_point_delta: int
    # flags
    is_tsumo: bool
    # riichi / ura info if hero is winner & riichi
    hero_was_riichi: bool = False
    ura_indicators: int = 0
    ura_hits: int = 0
    ippatsu: bool = False


@dataclass
class RoundModel:
    # hero seat 0..3
    hero_seat: int

    # Whether hero is dealer this round
    hero_is_dealer: bool

    # Starting hand (13 tiles) for hero as tile types 0..33
    starting_hand_types: List[int]

    # Dead wall tiles (full dead wall at end of round), as tile types 0..33
    dead_wall_types: List[int] = field(default_factory=list)

    # All draw steps for hero in this round
    hero_draws: List[DrawStep] = field(default_factory=list)

    # All riichi declarations by hero in this round
    hero_riichi_instances: List[RiichiInstance] = field(default_factory=list)

    # If round ends in win/draw, record it here
    win_event: Optional[WinEvent] = None


# ====== Luck metrics accumulator ======

@dataclass
class LuckMetrics:
    # 1. starting shanten
    sum_shanten_dealer: int = 0
    count_start_dealer: int = 0
    sum_shanten_nondealer: int = 0
    count_start_nondealer: int = 0

    # 2. ukeire hit rate
    ukeire_hits: int = 0
    ukeire_opportunities: int = 0  # draws where ukeire > 0

    # 3. ukeire improvement when it increases
    sum_ukeire_gain: int = 0
    count_ukeire_gain: int = 0
    total_draws: int = 0

    # 4. deal-in while riichi
    deal_in_while_riichi: int = 0
    riichi_discard_turns: int = 0

    # 5. wanted tiles in dead wall
    sum_wanted_in_deadwall: int = 0
    tenpai_rounds_for_deadwall: int = 0  # rounds where hero had a final wait

    # 6. extra points lost because of dealership
    sum_extra_loss_dealer_tsumo: int = 0
    count_dealer_tsumo_against_hero: int = 0

    # 7. ippatsu rate
    ippatsu_wins: int = 0
    riichi_count_for_ippatsu: int = 0

    # 8. ura dora hits
    ura_hits_total: int = 0
    ura_indicators_total: int = 0
    riichi_wins_count: int = 0

    def update_from_round(self, rnd: RoundModel):
        # 1. starting hand shanten
        s = shanten_number(rnd.starting_hand_types)
        if rnd.hero_is_dealer:
            self.sum_shanten_dealer += s
            self.count_start_dealer += 1
        else:
            self.sum_shanten_nondealer += s
            self.count_start_nondealer += 1

        # 2 & 3. Ukeire hit / improvement per draw
        # We recompute ukeire from tiles_before each draw
        for step in rnd.hero_draws:
            self.total_draws += 1
            tiles_before = step.tiles_before
            tile_drawn = step.tile_drawn

            ukeire_before = ukeire_tiles(tiles_before)
            u_before_set = set(ukeire_before)
            u_before_size = len(u_before_set)

            # new tiles after draw
            tiles_after = tiles_before + [tile_drawn]
            ukeire_after = ukeire_tiles(tiles_after)
            u_after_set = set(ukeire_after)

            if u_before_size > 0:
                self.ukeire_opportunities += 1
                if tile_drawn in u_before_set:
                    self.ukeire_hits += 1

            gain = len(u_after_set) - u_before_size
            if gain > 0:
                self.sum_ukeire_gain += gain
                self.count_ukeire_gain += 1

        # For metrics 4–8 we rely on RoundModel being fully populated
        # These parts assume you've correctly wired log parsing.

        # 4. Deal-in while riichi:
        # This needs: for each deal-in event you mark whether hero was riichi.
        # Here we assume that for the final win_event, if loser_seat == hero_seat
        # then hero dealt in, and that you counted how many of hero's discards
        # while in riichi we had (riichi_discard_turns).
        # For now we only treat the final losing event; if you want per-discard
        # analysis, enhance the model.
        if rnd.win_event and rnd.win_event.loser_seat == rnd.hero_seat:
            # Did hero happen to be in riichi at the moment of deal-in?
            # You need to set this in your extraction logic; here we
            # approximate by: if hero ever riichied in this round.
            hero_ever_riichi = any(True for _ in rnd.hero_riichi_instances)
            if hero_ever_riichi:
                self.deal_in_while_riichi += 1

        # For denominator, approximate riichi discard turns by number of
        # riichi instances (you can refine later with per-turn data).
        self.riichi_discard_turns += len(rnd.hero_riichi_instances)

        # 5. Wanted tiles in dead wall:
        # "Wanted tiles" = final winning wait set if hero finished in tenpai.
        # That requires your parsing logic to compute hero's final wait.
        # Here we just define a hook: you should store
        # rnd.final_wait_tile_types: List[int] when hero is tenpai at end.
        # For now we'll assume a field exists or is None.
        final_wait = getattr(rnd, "final_wait_tile_types", None)
        if final_wait:
            # Count how many of these wait tiles are in the dead wall
            deadwall_counts = hand_types_to_counts(rnd.dead_wall_types)
            wait_counts = hand_types_to_counts(final_wait)
            stuck = 0
            for t in range(34):
                if wait_counts[t] > 0:
                    # max copies we can "want" of this tile is wait_counts[t],
                    # but stuck copies in deadwall is deadwall_counts[t]
                    stuck += min(wait_counts[t], deadwall_counts[t])
            self.sum_wanted_in_deadwall += stuck
            self.tenpai_rounds_for_deadwall += 1

        # 6. Extra points lost because of dealership if others tsumo
        if (
            rnd.win_event
            and rnd.win_event.is_tsumo
            and rnd.win_event.winner_seat != rnd.hero_seat
        ):
            # When hero is dealer, actual loss is hero_point_delta (negative).
            # We want: extra loss = actual_loss - hypothetical_loss_if_child.
            # You need to supply hero_point_delta and a hypothetical
            # child loss from your parsing logic. For now we expect
            # you to attach 'hypo_child_loss' to win_event.
            if rnd.hero_is_dealer:
                actual_loss = -rnd.win_event.hero_point_delta  # make positive
                hypo = getattr(rnd.win_event, "hypo_child_loss", None)
                if hypo is not None:
                    extra = actual_loss - hypo
                    if extra > 0:
                        self.sum_extra_loss_dealer_tsumo += extra
                        self.count_dealer_tsumo_against_hero += 1

        # 7. Ippatsu rate:
        # For every riichi instance, check if it was an ippatsu win
        for inst in rnd.hero_riichi_instances:
            self.riichi_count_for_ippatsu += 1
            if inst.ippatsu_win:
                self.ippatsu_wins += 1

        # 8. Ura dora hits:
        # If hero wins by riichi, we expect win_event.ura_indicators and
        # win_event.ura_hits to be filled in your parsing logic.
        if rnd.win_event and rnd.win_event.hero_was_riichi:
            self.riichi_wins_count += 1
            self.ura_hits_total += rnd.win_event.ura_hits
            self.ura_indicators_total += rnd.win_event.ura_indicators

    def as_summary_dict(self) -> Dict[str, float]:
        """
        Return a dictionary with human-friendly aggregate numbers.
        """
        res: Dict[str, float] = {}

        # 1. average starting shanten
        if self.count_start_dealer > 0:
            res["avg_start_shanten_dealer"] = (
                self.sum_shanten_dealer / self.count_start_dealer
            )
        if self.count_start_nondealer > 0:
            res["avg_start_shanten_nondealer"] = (
                self.sum_shanten_nondealer / self.count_start_nondealer
            )

        # 2. ukeire hit rate
        if self.ukeire_opportunities > 0:
            res["ukeire_hit_rate"] = (
                self.ukeire_hits / self.ukeire_opportunities
            )

        # 3. average improvement per draw when it improves
        if self.count_ukeire_gain > 0:
            res["avg_ukeire_gain_when_positive"] = (
                self.sum_ukeire_gain / self.count_ukeire_gain
            )

        # 4. deal-in while riichi (count + rate)
        res["deal_in_while_riichi_count"] = float(self.deal_in_while_riichi)
        if self.riichi_discard_turns > 0:
            res["deal_in_while_riichi_rate"] = (
                self.deal_in_while_riichi / self.riichi_discard_turns
            )

        # 5. wanted tiles in dead wall
        res["wanted_in_deadwall_total"] = float(self.sum_wanted_in_deadwall)
        if self.tenpai_rounds_for_deadwall > 0:
            res["wanted_in_deadwall_per_tenpai_round"] = (
                self.sum_wanted_in_deadwall / self.tenpai_rounds_for_deadwall
            )

        # 6. average extra points lost as dealer
        if self.count_dealer_tsumo_against_hero > 0:
            res["avg_extra_loss_dealer_tsumo"] = (
                self.sum_extra_loss_dealer_tsumo
                / self.count_dealer_tsumo_against_hero
            )

        # 7. ippatsu rate
        res["ippatsu_wins"] = float(self.ippatsu_wins)
        res["riichi_count_for_ippatsu"] = float(self.riichi_count_for_ippatsu)
        if self.riichi_count_for_ippatsu > 0:
            res["ippatsu_rate"] = (
                self.ippatsu_wins / self.riichi_count_for_ippatsu
            )

        # 8. ura dora hits
        res["ura_hits_total"] = float(self.ura_hits_total)
        res["ura_indicators_total"] = float(self.ura_indicators_total)
        res["riichi_wins_count"] = float(self.riichi_wins_count)
        if self.ura_indicators_total > 0:
            res["ura_hit_rate_per_indicator"] = (
                self.ura_hits_total / self.ura_indicators_total
            )
        if self.riichi_wins_count > 0:
            res["ura_hits_per_riichi_win"] = (
                self.ura_hits_total / self.riichi_wins_count
            )

        return res


# ====== Game parsing glue (TO BE WIRED TO MAHJONG SOUL RECORDS) ======

def extract_rounds_from_game(
    game_json: dict,
    hero_seat: int,
) -> List[RoundModel]:
    """
    Convert a single GameDetailRecords JSON (from download_logs.py)
    into a list of RoundModel objects for the given hero_seat (0..3).

    IMPORTANT:
        This function is a TEMPLATE only. The exact structure of
        game_json depends on the Mahjong Soul protocol version and
        how MessageToJson serialized your GameDetailRecords.

        You need to:
          - Inspect one of your saved logs (open the JSON in an editor).
          - Find the equivalent of .records[i].name, .records[i].data
            for:
                - RecordNewRound
                - RecordDealTile
                - RecordDiscardTile
                - RecordHule (win)
                - RecordNoTile / RecordLiuJu (draw)
          - For RecordNewRound:
                - derive starting hands for all seats
                - derive dead wall tiles
                - derive dealer seat
          - For per-tile events (deal/discard/chi/pon/kan):
                - maintain hero's hand as a list of tile types (0..33)
                - whenever hero draws (deal tile) add DrawStep(...)
          - For riichi:
                - detect when hero declares riichi
                - create RiichiInstance and later mark ippatsu_win if win in ippatsu
          - For wins:
                - determine who won, who lost, points changes
                - fill WinEvent fields:
                    winner_seat, loser_seat, hero_point_delta,
                    is_tsumo, hero_was_riichi (if hero is winner),
                    ura_indicators, ura_hits, ippatsu,
                    hypo_child_loss (optional, for extra dealer loss)

        Once this function returns a correct list of RoundModel,
        the rest of analyze_log.py will compute all your metrics.
    """
    # --- Minimal placeholder: return [] so the script runs without crashing.
    # Replace EVERYTHING below with real parsing.
    # You have all the math infra above; this is the only piece that depends
    # on the specific protobuf/JSON layout.
    return []


# ====== CLI / main ======

def analyze_path(path: Path, hero_seat: int) -> Dict[str, float]:
    metrics = LuckMetrics()

    files: List[Path] = []
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    else:
        files = [path]

    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
            game_json = json.loads(text)
        except Exception as e:
            print(f"[WARN] Failed to read/parse {f}: {e}")
            continue

        rounds = extract_rounds_from_game(game_json, hero_seat)
        for rnd in rounds:
            metrics.update_from_round(rnd)

    return metrics.as_summary_dict()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Mahjong Soul logs and compute luck metrics."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a single JSON log file or a directory containing logs.",
    )
    parser.add_argument(
        "--seat",
        type=int,
        default=0,
        help="Your seat index (0..3, where 0 usually = East). "
             "Used to interpret which player is 'you'.",
    )

    args = parser.parse_args()
    p = Path(args.path)

    if not p.exists():
        print(f"Path does not exist: {p}")
        return

    summary = analyze_path(p, hero_seat=args.seat)

    print("=== Luck Metrics Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
