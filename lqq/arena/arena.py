"""Arena — Multi-bot tournament system for Luqiangqi.

Supports round-robin, double-round-robin, and random-matching modes with
parallel game execution and ELO rating computation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing
import random
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SEED = 20260520
DEFAULT_ELO = 1500.0
DEFAULT_K = 32.0
DEFAULT_TURN_LIMIT = 400


def _worker_run_match(args: tuple[str, str, int, int, float, float, int]) -> dict[str, Any]:
    """Worker function (runs in subprocess) — executes one game via battle_bots_once.
    
    Receives (bot0_path, bot1_path, bot0_seat, seed, referee_timeout, perceived_timeout, turn_limit).
    referee_timeout = hard thread-join limit; perceived_timeout = what the bot sees.
    """
    bot0_path, bot1_path, bot0_seat, seed, decision_timeout, perceived_timeout, turn_limit = args

    from env.lqq_env import battle_bots_once

    return battle_bots_once(
        bot0_path,
        bot1_path,
        bot0_seat=bot0_seat,
        seed=seed,
        keep_log=False,
        turn_limit=turn_limit,
        decision_timeout=decision_timeout,
        perceived_timeout=perceived_timeout,
    )


@dataclass
class Rating:
    elo: float = DEFAULT_ELO
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class Arena:
    """Orchestrate tournaments across multiple bots, compute ELO ratings."""

    def __init__(
        self,
        bots: list[Path],
        *,
        mode: str = "round-robin",
        rounds: int = 4,
        decision_timeout: float | None = 1.0,
        turn_limit: int = DEFAULT_TURN_LIMIT,
        seed: int = DEFAULT_SEED,
        initial_elo: float = DEFAULT_ELO,
        k_factor: float = DEFAULT_K,
        workers: int = 1,
        output: Path | None = None,
    ):
        if len(bots) < 2:
            raise ValueError("need at least 2 bots")
        if mode not in ("round-robin", "double-round-robin", "random"):
            raise ValueError(f"unknown mode: {mode!r}")
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        if workers < 1:
            raise ValueError("workers must be >= 1")

        self.bots = bots
        self.mode = mode
        self.rounds = rounds
        self.decision_timeout = decision_timeout
        self._referee_timeout = (decision_timeout * 3.0) if decision_timeout is not None else None
        self.turn_limit = turn_limit
        self.seed = seed
        self.k_factor = k_factor
        self.workers = workers
        self.output = output

        self.ratings: dict[str, Rating] = {}
        self._match_log: list[dict[str, Any]] = []

        self._load_bot_names()
        self._init_ratings(initial_elo)

    def _load_bot_names(self) -> None:
        from env.lqq_env import load_bot

        self._bot_names: dict[str, str] = {}
        self._bot_labels: dict[str, str] = {}
        for p in self.bots:
            key = str(p)
            try:
                bot = load_bot(p)
                name = getattr(bot, "name", p.stem)
            except Exception:
                name = p.stem
            # Derive a short label from the relative path.
            try:
                repo = Path(__file__).resolve().parent.parent
                rel = p.resolve().relative_to(repo)
            except ValueError:
                rel = p
            label = str(rel.with_suffix("")).replace("\\", "/").replace("baseline/", "")
            self._bot_names[key] = str(name)
            self._bot_labels[key] = label
            if label not in self.ratings:
                self.ratings[label] = Rating()

    def _bot_label(self, path: str) -> str:
        return self._bot_labels.get(path, path)

    def _bot_name(self, path: str) -> str:
        return self._bot_names.get(path, path)

    def _init_ratings(self, initial_elo: float) -> None:
        for p in self.bots:
            label = self._bot_label(str(p))
            if label not in self.ratings:
                self.ratings[label] = Rating(elo=initial_elo)

    def _schedule_matches(self) -> list[tuple[str, str, int, int]]:
        """Return (bot0_path, bot1_path, bot0_seat, seed) for each game."""
        tasks: list[tuple[str, str, int, int]] = []
        bot_paths = [str(p) for p in self.bots]

        if self.mode in ("round-robin", "double-round-robin"):
            pairs = list(itertools.combinations(range(len(bot_paths)), 2))
            effective_rounds = self.rounds * (2 if self.mode == "double-round-robin" else 1)
            game_idx = 0
            for i, j in pairs:
                for r in range(effective_rounds):
                    bot0_seat = r % 2
                    tasks.append((bot_paths[i], bot_paths[j], bot0_seat, self.seed + game_idx))
                    game_idx += 1
        elif self.mode == "random":
            total = self.rounds * len(bot_paths) * (len(bot_paths) - 1) // 2
            rng = random.Random(self.seed)
            for idx in range(total):
                i, j = rng.sample(range(len(bot_paths)), 2)
                bot0_seat = rng.randint(0, 1)
                tasks.append((bot_paths[i], bot_paths[j], bot0_seat, self.seed + idx))

        return tasks

    def _expected_score(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + math.pow(10.0, (rb - ra) / 400.0))

    def _update_elo(self, label_a: str, label_b: str, score_a: float) -> None:
        ra = self.ratings[label_a]
        rb = self.ratings[label_b]
        ea = self._expected_score(ra.elo, rb.elo)
        eb = 1.0 - ea
        ra.elo += self.k_factor * (score_a - ea)
        rb.elo += self.k_factor * ((1.0 - score_a) - eb)

    def _result_to_score(self, result: dict[str, Any]) -> tuple[float, int | None]:
        winner_bot = result.get("winner_bot")
        if winner_bot == 0:
            return 1.0, 0
        if winner_bot == 1:
            return 0.0, 1
        return 0.5, None

    def run(self) -> dict[str, Any]:
        tasks = self._schedule_matches()
        total_games = len(tasks)

        print(f"Arena: {len(self.bots)} bots, {total_games} games")
        print(f"  mode     = {self.mode}")
        print(f"  timeout  = {self.decision_timeout}s")
        print(f"  workers  = {self.workers}")
        print(f"  k-factor = {self.k_factor}")
        print()
        for p in self.bots:
            label = self._bot_label(str(p))
            name = self._bot_name(str(p))
            print(f"  {label}  ({name})")
        print()

        completed = 0
        t_start = time.perf_counter()

        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(_worker_run_match, (b0, b1, seat, sd, self._referee_timeout, self.decision_timeout, self.turn_limit)): (b0, b1, seat, sd)
                for b0, b1, seat, sd in tasks
            }

            for future in as_completed(futures):
                b0, b1, seat, sd = futures[future]
                try:
                    result = future.result(timeout=(self._referee_timeout or 10) * 400 + 60)
                except Exception:
                    result = {
                        "status": "worker_error",
                        "error": traceback.format_exc(),
                        "winner": None,
                        "winner_bot": None,
                        "turns": 0,
                        "bot_seats": [seat, 1 - seat],
                    }

                label_a = self._bot_label(b0)
                label_b = self._bot_label(b1)
                score_a, winner_bot = self._result_to_score(result)
                self._update_elo(label_a, label_b, score_a)

                for label in (label_a, label_b):
                    self.ratings[label].games += 1
                if winner_bot == 0:
                    self.ratings[label_a].wins += 1
                    self.ratings[label_b].losses += 1
                elif winner_bot == 1:
                    self.ratings[label_b].wins += 1
                    self.ratings[label_a].losses += 1
                else:
                    self.ratings[label_a].draws += 1
                    self.ratings[label_b].draws += 1

                match_entry = {
                    "bot_a": label_a,
                    "bot_b": label_b,
                    "bot_a_seat": result.get("bot_seats", [None, None])[0],
                    "bot_b_seat": result.get("bot_seats", [None, None])[1] if result.get("bot_seats") and len(result["bot_seats"]) > 1 else None,
                    "winner": None if winner_bot is None else (label_a if winner_bot == 0 else label_b),
                    "status": result.get("status", "unknown"),
                    "turns": result.get("turns", 0),
                    "seed": sd,
                }
                self._match_log.append(match_entry)

                completed += 1
                if completed % 10 == 0 or completed == total_games:
                    elapsed = time.perf_counter() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_games - completed) / rate if rate > 0 else 0
                    print(f"[{completed}/{total_games}] {rate:.1f} games/s  ETA {eta:.0f}s")

        elapsed = time.perf_counter() - t_start
        print(f"\nDone. {total_games} games in {elapsed:.1f}s  ({total_games / elapsed:.1f} games/s)")

        self._print_rankings()
        payload = self._export()
        if self.output:
            self.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"\nExported to {self.output}")
        return payload

    def _print_rankings(self) -> None:
        ranked = sorted(self.ratings.items(), key=lambda kv: kv[1].elo, reverse=True)
        print()
        print("Rank  Elo     W-L-D    Games  Bot")
        print("----  ------  -------  -----  ---")
        for rank, (label, r) in enumerate(ranked, start=1):
            record = f"{r.wins}-{r.losses}-{r.draws}"
            print(f"{rank:>4}  {r.elo:>6.1f}  {record:>7}  {r.games:>5}  {label}")

    def _export(self) -> dict[str, Any]:
        ranked = sorted(self.ratings.items(), key=lambda kv: kv[1].elo, reverse=True)
        return {
            "mode": self.mode,
            "rounds": self.rounds,
            "decision_timeout": self.decision_timeout,
            "turn_limit": self.turn_limit,
            "seed": self.seed,
            "initial_elo": DEFAULT_ELO,
            "k_factor": self.k_factor,
            "workers": self.workers,
            "total_games": len(self._match_log),
            "rankings": [
                {
                    "rank": rank,
                    "bot": label,
                    "bot_name": self._bot_name(str(self._find_bot_path(label))),
                    "elo": round(r.elo, 1),
                    "games": r.games,
                    "wins": r.wins,
                    "losses": r.losses,
                    "draws": r.draws,
                }
                for rank, (label, r) in enumerate(ranked, start=1)
            ],
            "matches": self._match_log,
        }

    def _find_bot_path(self, label: str) -> str:
        for p in self.bots:
            if self._bot_label(str(p)) == label:
                return str(p)
        return ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Luqiangqi Arena — Multi-bot ELO tournament")
    parser.add_argument("--bots", nargs="+", required=True, help="bot .py files to compete")
    parser.add_argument("--mode", choices=["round-robin", "double-round-robin", "random"], default="round-robin")
    parser.add_argument("--rounds", type=int, default=4, help="games per pair (round-robin) or total games multiplier")
    parser.add_argument("--decision-timeout", type=float, default=1.0)
    parser.add_argument("--turn-limit", type=int, default=DEFAULT_TURN_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--initial-elo", type=float, default=DEFAULT_ELO)
    parser.add_argument("--k-factor", type=float, default=DEFAULT_K)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", help="JSON output path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    bot_paths: list[Path] = []
    for bot_arg in args.bots:
        p = Path(bot_arg)
        if not p.is_absolute():
            p = (repo / p).resolve()
        if not p.exists():
            print(f"ERROR: bot not found: {p}", file=sys.stderr)
            return 1
        bot_paths.append(p)

    output = Path(args.output) if args.output else None
    if output and not output.is_absolute():
        output = Path.cwd() / output

    arena = Arena(
        bots=bot_paths,
        mode=args.mode,
        rounds=args.rounds,
        decision_timeout=args.decision_timeout,
        turn_limit=args.turn_limit,
        seed=args.seed,
        initial_elo=args.initial_elo,
        k_factor=args.k_factor,
        workers=args.workers,
        output=output,
    )
    arena.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
