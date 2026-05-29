"""Round-robin tournament for Gomoku baseline bots.
100 games per matchup, 3s decision timeout, ELO calculation.
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

from env.gomoku_env import GomokuEnv, load_bot, DEFAULT_MAX_PLIES

GAMES_PER_MATCHUP = 100
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


def play_one_game(bot_black, bot_white, seed: int, decision_timeout: float, max_plies: int):
    env = GomokuEnv(seed=seed, max_plies=max_plies)
    bots = [bot_black, bot_white]

    while True:
        state = env.state(env.actor)
        if state["phase"] == "game_over":
            break
        if env.max_plies is not None and env.plies >= env.max_plies:
            break
        legal = state["legal_actions"]
        if not legal:
            break
        state["decision_timeout"] = decision_timeout
        state["time_limit"] = decision_timeout
        try:
            action = bots[env.actor].choose_action(state)
        except Exception:
            action = legal[0]
        if action not in legal:
            action = legal[0]
        env.step(action)

    final = env.state()
    return final.get("winner")


def run_round_robin(bot_paths: dict[str, Path], games_per_matchup: int, decision_timeout: float, max_plies: int):
    bot_names = list(bot_paths.keys())
    bots_loaded = {}
    for name, path in bot_paths.items():
        try:
            bots_loaded[name] = load_bot(path)
        except Exception as e:
            print(f"  SKIP {name}: load error ({e})")
            continue

    valid_names = [n for n in bot_names if n in bots_loaded]
    if len(valid_names) < 2:
        print("Need at least 2 valid bots for a tournament.")
        return [], {}, {}

    n = len(valid_names)
    total_matchups = n * (n - 1) // 2
    matchup_idx = 0

    matchup_results = {}
    for i in range(n):
        for j in range(i + 1, n):
            matchup_results[(i, j)] = [0, 0, 0]

    for i in range(n):
        for j in range(i + 1, n):
            matchup_idx += 1
            name_i, name_j = valid_names[i], valid_names[j]
            bot_i = bots_loaded[name_i]
            bot_j = bots_loaded[name_j]

            print(f"\n[{matchup_idx}/{total_matchups}] {name_i} vs {name_j} ({games_per_matchup} games)")
            sys.stdout.flush()

            wi, wj, dr = 0, 0, 0
            t0 = time.time()

            for g in range(games_per_matchup):
                seed = 42 + matchup_idx * 10000 + g
                if g % 2 == 0:
                    winner = play_one_game(bot_i, bot_j, seed, decision_timeout, max_plies)
                    if winner == 0:
                        wi += 1
                    elif winner == 1:
                        wj += 1
                    else:
                        dr += 1
                else:
                    winner = play_one_game(bot_j, bot_i, seed, decision_timeout, max_plies)
                    if winner == 0:
                        wj += 1
                    elif winner == 1:
                        wi += 1
                    else:
                        dr += 1

                if (g + 1) % 20 == 0 or g == games_per_matchup - 1:
                    elapsed = time.time() - t0
                    print(f"  [{g + 1}/{games_per_matchup}] {name_i} {wi}-{wj}-{dr} {name_j}  ({elapsed:.0f}s)")
                    sys.stdout.flush()

            matchup_results[(i, j)] = [wi, wj, dr]

    return valid_names, bots_loaded, matchup_results


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
    parser = argparse.ArgumentParser(description="Gomoku baseline bot round-robin tournament")
    parser.add_argument("--games", type=int, default=GAMES_PER_MATCHUP)
    parser.add_argument("--timeout", type=float, default=DECISION_TIMEOUT)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--output", default="tourney_results.json")
    args = parser.parse_args()

    baseline_dir = HERE / "baseline"
    bots = discover_bots(baseline_dir)

    print(f"Found {len(bots)} bots in baseline/:")
    for name in sorted(bots.keys()):
        print(f"  {name}")

    print(f"\nGames per matchup: {args.games}")
    print(f"Decision timeout: {args.timeout}s per move")
    print(f"Max plies: {args.max_plies}")

    bot_names, _, matchup_results = run_round_robin(
        bots, args.games, args.timeout, args.max_plies,
    )

    if not bot_names:
        return

    elos = compute_elos(bot_names, matchup_results)

    print(f"\n{'=' * 60}")
    print("FINAL STANDINGS (ELO)")
    print(f"{'=' * 60}")
    sorted_elos = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    for name, elo in sorted_elos:
        print(f"  {name:<45}  {elo:>7.1f}")

    result = {
        "game": "gomoku",
        "games_per_matchup": args.games,
        "decision_timeout": args.timeout,
        "max_plies": args.max_plies,
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
