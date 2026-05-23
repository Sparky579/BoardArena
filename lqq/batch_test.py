import subprocess
import json
import sys
import argparse

OPPONENTS = [
    "lqq/baseline/gemini-MCTS/bot_v2.py",
    "lqq/baseline/gemini-MCTS/bot_v5.py",
    "lqq/baseline/gemini-MCTS/bot_v6.py",
    "lqq/baseline/claude-opus4p7/bot_hard_v2.py",
]

def run_match(target_bot, opponent, games):
    cmd = [
        sys.executable, "lqq/lqq_multi.py", "duel",
        "--bot0", target_bot,
        "--bot1", opponent,
        "--games", str(games)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse JSON output from stdout
        data = json.loads(res.stdout.strip())
        target_wins = data["wins_by_bot"][0]
        opp_wins = data["wins_by_bot"][1]
        return opponent, target_wins, opp_wins
    except subprocess.CalledProcessError as e:
        print(f"Error running against {opponent}: Command failed with exit code {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return opponent, 0, 0
    except json.JSONDecodeError as e:
        print(f"Error running against {opponent}: Invalid JSON output.")
        print(f"Output: {res.stdout}")
        return opponent, 0, 0
    except Exception as e:
        print(f"Error running against {opponent}: {e}")
        return opponent, 0, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Path to the bot to test")
    parser.add_argument("--games", type=int, default=4, help="Games per opponent")
    args = parser.parse_args()

    print(f"Testing {args.target} against {len(OPPONENTS)} opponents ({args.games} games each)...")
    
    total_wins = 0
    total_games = 0

    results = []
    
    for opp in OPPONENTS:
        print(f"Playing vs {opp} ... ", end="", flush=True)
        _, w, l = run_match(args.target, opp, args.games)
        total_wins += w
        total_games += (w + l)
        results.append((opp, w, l))
        print(f"{w} - {l}")

    print("\n--- Final Results ---")
    for opp, w, l in results:
        # Just grab the last part of the path for cleaner output
        short_opp = opp.split('/')[-1]
        if 'claude' in opp:
            short_opp = 'claude/' + short_opp
        print(f"vs {short_opp:20s} : {w} - {l}")
    
    if total_games > 0:
        print(f"\nOverall Win Rate: {total_wins / total_games * 100:.1f}% ({total_wins}/{total_games})")

if __name__ == "__main__":
    main()
