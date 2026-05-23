"""Quick bot-vs-bot duel harness for chess.

Usage:
    python duel.py path/to/bot_a.py path/to/bot_b.py [--games N] [--seed S]
                  [--decision-timeout T] [--max-plies P]

Defaults: 20 games, alternating colors. Reports wins/losses/draws by bot.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

import chess

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from env.chess_env import (
    BotTimeoutError,
    ChessEnv,
    DEFAULT_MAX_PLIES,
    choose_action_with_timeout,
    load_bot,
)


def play_one(
    bot_white,
    bot_black,
    *,
    seed: int,
    max_plies: int,
    decision_timeout: float | None,
) -> dict:
    env = ChessEnv(seed=seed, max_plies=max_plies)
    bots = [bot_white, bot_black]
    status = "ok"
    while True:
        state = env.state(env.actor)
        if state["phase"] == "game_over":
            break
        if env.max_plies is not None and env.plies >= env.max_plies:
            status = "turn_limit"
            break
        actor = state["actor"]
        legal = state["legal_actions"]
        if not legal:
            status = "no_legal_actions"
            break
        try:
            action = choose_action_with_timeout(bots[actor], state, decision_timeout)
        except BotTimeoutError:
            status = "timeout"
            break
        except Exception:  # noqa: BLE001
            status = "bot_exception"
            break
        if not isinstance(action, str) or action not in legal:
            status = "invalid_action"
            break
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            status = info.get("status", "ok") if not truncated else "turn_limit"
            break

    final_state = env.state()
    winner = final_state["winner"]
    if status in {"bot_exception", "invalid_action", "timeout"}:
        loser = env.actor
        winner = 1 - loser
    elif status == "turn_limit":
        winner = None
    return {
        "winner": winner,
        "status": status,
        "result": final_state["result"],
        "plies": env.plies,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bot_a")
    parser.add_argument("bot_b")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--decision-timeout", type=float, default=2.0)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--alternate", action="store_true", default=True)
    args = parser.parse_args()

    bot_a = load_bot(args.bot_a)
    bot_b = load_bot(args.bot_b)

    rng = random.Random(args.seed)
    a_wins = b_wins = draws = 0
    statuses = {}

    def print_summary():
        total = a_wins + b_wins + draws
        if total == 0:
            return
        print(json.dumps({
            "bot_a": args.bot_a,
            "bot_b": args.bot_b,
            "games": total,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "a_win_rate": a_wins / total,
            "a_score_rate": (a_wins + 0.5 * draws) / total,
            "statuses": statuses,
        }, indent=2), flush=True)

    import atexit
    atexit.register(print_summary)

    for i in range(args.games):
        # Alternate seats: even games -> bot_a is white; odd -> bot_b is white.
        if i % 2 == 0:
            white, black = bot_a, bot_b
            a_white = True
        else:
            white, black = bot_b, bot_a
            a_white = False
        seed = rng.randrange(1 << 30)
        try:
            r = play_one(
                white, black,
                seed=seed,
                max_plies=args.max_plies,
                decision_timeout=args.decision_timeout,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  game {i+1:2d}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        if r["winner"] is None:
            draws += 1
            outcome = "draw"
        else:
            # winner is 0=white or 1=black.
            a_won = (r["winner"] == 0) == a_white
            if a_won:
                a_wins += 1
                outcome = "A"
            else:
                b_wins += 1
                outcome = "B"
        print(
            f"  game {i+1:2d}: {outcome:4s} "
            f"({r['result']}, {r['status']}, {r['plies']} plies, "
            f"A={'W' if a_white else 'B'})",
            flush=True,
        )

    total = args.games
    print(json.dumps({
        "bot_a": args.bot_a,
        "bot_b": args.bot_b,
        "games": total,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_win_rate": a_wins / total,
        "a_score_rate": (a_wins + 0.5 * draws) / total,
        "statuses": statuses,
    }, indent=2))


if __name__ == "__main__":
    main()
