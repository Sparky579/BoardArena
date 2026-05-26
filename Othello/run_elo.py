"""Othello ELO estimation using Edax bots as fixed-ELO anchors.

Plays each target bot against every Edax anchor bot, then derives a
performance ELO from the observed win/loss/draw counts.  Edax anchor
ELOs are treated as ground truth and never updated.

Usage (from the repo root):
    set EDAX_PATH=C:\\Users\\24789\\edax-4.6\\wEdax-x86-64.exe
    set EDAX_DATA_DIR=C:\\Users\\24789\\edax-4.6
    python Othello/run_elo.py ^
        --targets Othello/baseline/gpt5p5/bot_hard.py ^
                  Othello/baseline/claude_opus4p7/bot_hard.py ^
        --games 4 --timeout 10.0

Or from inside the Othello/ directory:
    python run_elo.py --targets baseline/gpt5p5/bot_hard.py ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

# Allow "from env.othello_env import ..." regardless of cwd.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from env.othello_env import OthelloEnv, load_bot  # noqa: E402

# Approximate Edax level → Elo (community benchmarks at 1 s/move).
ANCHOR_ELOS: dict[str, float] = {
    "bot_edax_1300": 1300.0,
    "bot_edax_1500": 1500.0,
    "bot_edax_1700": 1700.0,
    "bot_edax_1900": 1900.0,
    "bot_edax_2100": 2100.0,
    "bot_edax_2300": 2300.0,
    "bot_edax_2500": 2500.0,
}


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def _play_one(bot_black, bot_white, seed: int, decision_timeout: float) -> tuple[int, int]:
    """Play one game.  Returns (black_score, white_score)."""
    env = OthelloEnv(seed=seed)
    env.reset()
    while not env.is_done():
        state = env.state(env.actor)
        state["decision_timeout"] = decision_timeout
        state["time_limit"] = decision_timeout
        bot = bot_black if env.actor == 0 else bot_white
        try:
            action = bot.choose_action(state)
        except Exception:
            # Bot crashed — forfeit its turn by picking first legal action.
            legal = state.get("legal_actions", [])
            action = legal[0] if legal else "PASS"
        if action not in state.get("legal_actions", []):
            legal = state.get("legal_actions", [])
            action = legal[0] if legal else "PASS"
        env.step(action)
    final = env.state(0)
    scores = final["scores"]
    return scores[0], scores[1]


def _run_match(
    target_bot,
    anchor_bot,
    games: int,
    decision_timeout: float,
    seed: int,
) -> tuple[int, int, int]:
    """Play *games* games between target and anchor (alternating colors).

    Returns (wins, losses, draws) from target's perspective.
    """
    wins = losses = draws = 0
    for i in range(games):
        game_seed = seed + i
        try:
            if i % 2 == 0:
                bs, ws = _play_one(target_bot, anchor_bot, game_seed, decision_timeout)
                target_score, anchor_score = bs, ws
            else:
                bs, ws = _play_one(anchor_bot, target_bot, game_seed, decision_timeout)
                target_score, anchor_score = ws, bs

            if target_score > anchor_score:
                wins += 1
            elif target_score < anchor_score:
                losses += 1
            else:
                draws += 1
        except Exception as exc:
            print(f"      game {i + 1} error: {exc}")
            traceback.print_exc()
    return wins, losses, draws


# ---------------------------------------------------------------------------
# ELO math
# ---------------------------------------------------------------------------

def _performance_elo(wins: int, losses: int, draws: int, anchor_elo: float) -> float | None:
    """Performance ELO against one anchor.

    Uses the standard formula:  perf = anchor + 400·log₁₀(score/(1-score))
    Clamps score to [0.005, 0.995] to avoid ±∞.
    Returns None when no games were played.
    """
    total = wins + losses + draws
    if total == 0:
        return None
    score_frac = (wins + 0.5 * draws) / total
    score_frac = max(0.005, min(0.995, score_frac))
    return anchor_elo + 400.0 * math.log10(score_frac / (1.0 - score_frac))


def _aggregate_elo(perf_elos: list[float]) -> float:
    """Average performance ELO across all anchors."""
    return sum(perf_elos) / len(perf_elos)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    external_dir = Path(args.external_dir)
    if not external_dir.is_absolute():
        external_dir = (_HERE / external_dir).resolve()

    # Discover available anchor bots.
    anchors: list[tuple[str, Path, float]] = []
    for name, elo in sorted(ANCHOR_ELOS.items(), key=lambda x: x[1]):
        path = external_dir / f"{name}.py"
        if path.exists():
            anchors.append((name, path, elo))
        else:
            print(f"[warn] anchor not found: {path}")

    if not anchors:
        raise SystemExit(f"No anchor bots found in {external_dir}")

    print(f"Anchor bots  : {len(anchors)} Edax levels")
    print(f"Games/matchup: {args.games} (alternating colors)")
    print(f"Timeout      : {args.timeout}s per move")
    print(f"Seed         : {args.seed}")
    print()

    all_results: dict = {"settings": vars(args), "bots": []}

    for target_path_str in args.targets:
        target_path = Path(target_path_str)
        if not target_path.is_absolute():
            target_path = (_HERE / target_path).resolve()

        label = "/".join(target_path.parts[-2:]).removesuffix(".py")
        print(f"{'=' * 60}")
        print(f"Target: {label}")
        print(f"{'=' * 60}")

        try:
            target_bot = load_bot(target_path)
        except Exception as exc:
            print(f"  ERROR loading bot: {exc}")
            all_results["bots"].append({"label": label, "error": str(exc)})
            continue

        perf_elos: list[float] = []
        matchup_rows: list[dict] = []
        total_w = total_l = total_d = 0

        for anchor_name, anchor_path, anchor_elo in anchors:
            print(f"  vs {anchor_name} (Elo {anchor_elo:.0f}) ... ", end="", flush=True)
            try:
                anchor_bot = load_bot(anchor_path)
            except Exception as exc:
                print(f"SKIP — load error: {exc}")
                continue

            w, l, d = _run_match(
                target_bot, anchor_bot,
                args.games, args.timeout, args.seed,
            )
            perf = _performance_elo(w, l, d, anchor_elo)
            total_w += w
            total_l += l
            total_d += d
            if perf is not None:
                perf_elos.append(perf)

            perf_str = f"{perf:.1f}" if perf is not None else "N/A"
            print(f"W{w}/L{l}/D{d}  perf={perf_str}")
            matchup_rows.append({
                "anchor": anchor_name,
                "anchor_elo": anchor_elo,
                "wins": w,
                "losses": l,
                "draws": d,
                "performance_elo": round(perf, 1) if perf is not None else None,
            })

        print()
        if perf_elos:
            estimated = _aggregate_elo(perf_elos)
            total_games = total_w + total_l + total_d
            win_pct = 100.0 * total_w / total_games if total_games else 0.0
            print(f"  Total games : {total_games}  W{total_w}/L{total_l}/D{total_d}  ({win_pct:.1f}% wins)")
            print(f"  Estimated ELO: {estimated:.1f}")
            print()
            all_results["bots"].append({
                "label": label,
                "estimated_elo": round(estimated, 1),
                "total_wins": total_w,
                "total_losses": total_l,
                "total_draws": total_d,
                "matchups": matchup_rows,
            })
        else:
            print("  No valid results — ELO could not be estimated.")
            all_results["bots"].append({
                "label": label,
                "estimated_elo": None,
                "matchups": matchup_rows,
            })

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Bot':<40}  {'ELO':>7}")
    print(f"{'-' * 40}  {'-' * 7}")
    for entry in sorted(
        [b for b in all_results["bots"] if b.get("estimated_elo") is not None],
        key=lambda b: b["estimated_elo"],
        reverse=True,
    ):
        print(f"{entry['label']:<40}  {entry['estimated_elo']:>7.1f}")

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nResults written to {out_path}")

    return all_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Othello bot ELO using Edax anchors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--targets", nargs="+", required=True,
        help="Bot file paths to evaluate (relative to Othello/ dir or absolute)",
    )
    parser.add_argument(
        "--external-dir", default="baseline/external",
        help="Directory containing bot_edax_*.py anchor files",
    )
    parser.add_argument(
        "--games", type=int, default=4,
        help="Games per (target, anchor) matchup (alternating colors)",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0,
        help="Decision timeout passed to bots (seconds); target bots self-limit internally",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON output path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(_parse_args()))
