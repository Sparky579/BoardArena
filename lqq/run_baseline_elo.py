"""Run a round-robin Elo evaluation for all Luqiangqi baseline bots."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from env.lqq_env import battle_bots_once, load_bot
except ModuleNotFoundError:
    from .env.lqq_env import battle_bots_once, load_bot  # type: ignore


DEFAULT_SEED = 20260520


@dataclass
class BotEntry:
    label: str
    path: Path
    module_name: str


@dataclass
class Rating:
    elo: float = 1500.0
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    statuses: Counter[str] = field(default_factory=Counter)


def discover_bots(baseline_dir: Path) -> list[BotEntry]:
    bots: list[BotEntry] = []
    for path in sorted(baseline_dir.rglob("*.py")):
        if "__pycache__" in path.parts or path.name.startswith("_"):
            continue
        module_name = getattr(load_bot(path), "name", path.stem)
        rel = path.relative_to(baseline_dir).as_posix()
        label = rel[:-3] if rel.endswith(".py") else rel
        if path.name == "bot.py" and path.parent != baseline_dir:
            label = path.parent.relative_to(baseline_dir).as_posix()
        bots.append(BotEntry(label=label, path=path, module_name=str(module_name)))
    return bots


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(rating_a: Rating, rating_b: Rating, score_a: float, k_factor: float) -> None:
    expected_a = expected_score(rating_a.elo, rating_b.elo)
    expected_b = 1.0 - expected_a
    rating_a.elo += k_factor * (score_a - expected_a)
    rating_b.elo += k_factor * ((1.0 - score_a) - expected_b)


def score_from_result(result: dict[str, Any]) -> tuple[float, int | None]:
    winner_bot = result.get("winner_bot")
    if winner_bot == 0:
        return 1.0, 0
    if winner_bot == 1:
        return 0.0, 1
    return 0.5, None


def print_table(rows: list[tuple[int, BotEntry, Rating]]) -> None:
    print()
    print("Rank  Elo     W-L-D   Games  Bot")
    print("----  ------  ------  -----  ---")
    for rank, bot, rating in rows:
        record = f"{rating.wins}-{rating.losses}-{rating.draws}"
        print(f"{rank:>4}  {rating.elo:>6.1f}  {record:>6}  {rating.games:>5}  {bot.label}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_lqq_dir = Path(__file__).resolve().parent
    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.is_absolute():
        baseline_dir = (repo_lqq_dir / baseline_dir).resolve()

    bots = discover_bots(baseline_dir)
    if len(bots) < 2:
        raise SystemExit(f"need at least 2 bots under {baseline_dir}")

    ratings = {bot.label: Rating(elo=args.initial_elo) for bot in bots}
    pair_count = len(bots) * (len(bots) - 1) // 2
    game_count = pair_count * args.rounds

    print(f"Discovered {len(bots)} bots in {baseline_dir}")
    for bot in bots:
        print(f"- {bot.label} ({bot.module_name})")
    print(f"Running {game_count} games: {pair_count} pairs x {args.rounds} rounds")
    print(f"Decision timeout: {args.decision_timeout:g}s")

    game_index = 0
    match_rows: list[dict[str, Any]] = []

    for pair_index, (bot_a, bot_b) in enumerate(itertools.combinations(bots, 2), start=1):
        pair_wins = [0, 0]
        pair_draws = 0
        pair_statuses: Counter[str] = Counter()

        for round_index in range(args.rounds):
            seed = args.seed + game_index
            bot0_seat = round_index % 2
            result = battle_bots_once(
                bot_a.path,
                bot_b.path,
                bot0_seat=bot0_seat,
                seed=seed,
                keep_log=False,
                turn_limit=args.turn_limit,
                decision_timeout=args.decision_timeout,
            )
            score_a, winner_bot = score_from_result(result)
            rating_a = ratings[bot_a.label]
            rating_b = ratings[bot_b.label]
            update_elo(rating_a, rating_b, score_a, args.k_factor)

            for rating in (rating_a, rating_b):
                rating.games += 1
                rating.statuses[result["status"]] += 1
            if winner_bot == 0:
                rating_a.wins += 1
                rating_b.losses += 1
                pair_wins[0] += 1
            elif winner_bot == 1:
                rating_b.wins += 1
                rating_a.losses += 1
                pair_wins[1] += 1
            else:
                rating_a.draws += 1
                rating_b.draws += 1
                pair_draws += 1

            pair_statuses[result["status"]] += 1
            match_rows.append(
                {
                    "bot_a": bot_a.label,
                    "bot_b": bot_b.label,
                    "round": round_index + 1,
                    "seed": seed,
                    "bot_a_seat": result["bot_seats"][0],
                    "bot_b_seat": result["bot_seats"][1],
                    "winner": None if winner_bot is None else (bot_a.label if winner_bot == 0 else bot_b.label),
                    "status": result["status"],
                    "turns": result["turns"],
                }
            )
            game_index += 1

        status_text = ", ".join(f"{key}:{value}" for key, value in sorted(pair_statuses.items()))
        print(
            f"[{pair_index}/{pair_count}] {bot_a.label} vs {bot_b.label}: "
            f"{pair_wins[0]}-{pair_wins[1]}-{pair_draws} ({status_text})"
        )

    ranked = sorted(bots, key=lambda bot: ratings[bot.label].elo, reverse=True)
    rows = [(rank, bot, ratings[bot.label]) for rank, bot in enumerate(ranked, start=1)]
    print_table(rows)

    payload = {
        "baseline_dir": str(baseline_dir),
        "rounds": args.rounds,
        "decision_timeout": args.decision_timeout,
        "turn_limit": args.turn_limit,
        "seed": args.seed,
        "initial_elo": args.initial_elo,
        "k_factor": args.k_factor,
        "rankings": [
            {
                "rank": rank,
                "bot": bot.label,
                "path": str(bot.path),
                "module_name": bot.module_name,
                "elo": round(rating.elo, 1),
                "games": rating.games,
                "wins": rating.wins,
                "losses": rating.losses,
                "draws": rating.draws,
                "statuses": dict(sorted(rating.statuses.items())),
            }
            for rank, bot, rating in rows
        ],
        "matches": match_rows,
    }
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = (Path.cwd() / output).resolve()
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {output}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lqq baseline round-robin Elo evaluation")
    parser.add_argument("--baseline-dir", default="baseline", help="baseline directory relative to lqq/")
    parser.add_argument("--rounds", type=int, default=4, help="games per bot pair")
    parser.add_argument("--decision-timeout", type=float, default=1.0, help="seconds per choose_action call")
    parser.add_argument("--turn-limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--initial-elo", type=float, default=1500.0)
    parser.add_argument("--k-factor", type=float, default=32.0)
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    if args.decision_timeout <= 0:
        parser.error("--decision-timeout must be positive")
    if args.turn_limit < 1:
        parser.error("--turn-limit must be >= 1")
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
