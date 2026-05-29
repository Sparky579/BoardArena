"""Round-robin tournament for Nimmt baseline bots.
100 games, 3s decision timeout, ELO calculation.

All baseline bots play together simultaneously in each game.
Pairwise results are derived from scores to compute ELO.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from env.nimmt_env import load_bot, run_match

GAMES = 100
DECISION_TIMEOUT = 3.0
INITIAL_ELO = 1500
K_FACTOR = 32


def discover_bots(baseline_dir: Path) -> dict[str, Path]:
    bots = {}
    for py_file in sorted(baseline_dir.rglob("*.py")):
        if py_file.name == "__init__.py" or "__pycache__" in py_file.parts:
            continue
        label = "/".join(py_file.relative_to(baseline_dir).parts).removesuffix(".py").replace("\\", "/")
        bots[label] = py_file
    return bots


def derive_pairwise(bot_names: list[str], scores_list: list[list[int]]) -> dict:
    """Derive pairwise wins/losses/draws from multi-player game scores.
    Lower score = better in Nimmt.
    Returns {(i,j): [wins_i, wins_j, draws]} for all i < j."""
    n = len(bot_names)
    results = {}
    for i in range(n):
        for j in range(i + 1, n):
            results[(i, j)] = [0, 0, 0]

    for scores in scores_list:
        for i in range(n):
            for j in range(i + 1, n):
                if scores[i] < scores[j]:
                    results[(i, j)][0] += 1
                elif scores[i] > scores[j]:
                    results[(i, j)][1] += 1
                else:
                    results[(i, j)][2] += 1

    return results


def compute_elos(bot_names: list[str], matchup_results: dict) -> dict[str, float]:
    elos = {name: INITIAL_ELO for name in bot_names}

    for _ in range(20):
        deltas = {name: 0.0 for name in bot_names}
        counts = {name: 0 for name in bot_names}

        for i in range(len(bot_names)):
            for j in range(i + 1, len(bot_names)):
                wi, wj, dr = matchup_results.get((i, j), [0, 0, 0])
                total = wi + wj + dr
                if total == 0:
                    continue
                e_i = 1.0 / (1.0 + 10.0 ** ((elos[bot_names[j]] - elos[bot_names[i]]) / 400.0))
                s_i = (wi + 0.5 * dr) / total
                delta = K_FACTOR * (s_i - e_i)
                deltas[bot_names[i]] += delta
                deltas[bot_names[j]] -= delta
                counts[bot_names[i]] += 1
                counts[bot_names[j]] += 1

        max_delta = max((abs(d) for d in deltas.values()), default=0.0)
        for name in bot_names:
            if counts[name] > 0:
                elos[name] += deltas[name] / counts[name]

        if max_delta < 0.01:
            break

    return elos


def main():
    parser = argparse.ArgumentParser(description="Nimmt baseline bot tournament")
    parser.add_argument("--games", type=int, default=GAMES)
    parser.add_argument("--timeout", type=float, default=DECISION_TIMEOUT)
    parser.add_argument("--output", default="tourney_results.json")
    args = parser.parse_args()

    baseline_dir = HERE / "baseline"
    bots = discover_bots(baseline_dir)

    if len(bots) < 2:
        print("Need at least 2 bots for a tournament.")
        return

    bot_names = sorted(bots.keys())
    print(f"Found {len(bots)} bots in baseline/:")
    for name in bot_names:
        print(f"  {name}")

    print(f"\nGames: {args.games} (all bots play together simultaneously)")
    print(f"Decision timeout: {args.timeout}s per move")

    bots_loaded = {name: load_bot(path) for name, path in bots.items()}

    scores_list = []
    t0 = time.time()

    for g in range(args.games):
        seed = 42 + g
        try:
            result = run_match(
                [bots_loaded[n] for n in bot_names],
                seed=seed,
                keep_log=False,
                decision_timeout=args.timeout,
            )
            scores_list.append(result["scores"])
        except Exception as e:
            print(f"  game {g + 1}: ERROR {e}")
            continue

        if (g + 1) % 10 == 0 or g == args.games - 1:
            elapsed = time.time() - t0
            print(f"  [{g + 1}/{args.games}] ({elapsed:.0f}s)")
            sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\nCompleted {len(scores_list)}/{args.games} games in {elapsed:.0f}s")

    matchup_results = derive_pairwise(bot_names, scores_list)
    elos = compute_elos(bot_names, matchup_results)

    print(f"\n{'=' * 60}")
    print("FINAL STANDINGS (ELO)")
    print(f"{'=' * 60}")
    sorted_elos = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    for name, elo in sorted_elos:
        print(f"  {name:<45}  {elo:>7.1f}")

    result = {
        "game": "nimmt",
        "games": args.games,
        "games_completed": len(scores_list),
        "decision_timeout": args.timeout,
        "bots": bot_names,
        "matchups": [
            {
                "bot_a": bot_names[i], "bot_b": bot_names[j],
                "a_wins": wi, "b_wins": wj, "draws": dr,
            }
            for (i, j), (wi, wj, dr) in matchup_results.items()
        ],
        "standings": {name: round(elo, 1) for name, elo in sorted_elos},
    }
    out_path = HERE / args.output
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
