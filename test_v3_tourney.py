import subprocess
import sys
import json

TARGET = "lqq/baseline/gemini-v3.0/bot.py"
OPPONENTS = [
    "lqq/baseline/gemini-MCTS/bot_v2.py",
    "lqq/baseline/claude-opus4p7/bot_hard.py",
    "lqq/baseline/bot_v3.py"
]

results = []
for opp in OPPONENTS:
    print(f"Running 6 games vs {opp}...", flush=True)
    cmd = [sys.executable, "lqq/lqq_multi.py", "duel", "--bot0", opp, "--bot1", TARGET, "--games", "6"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(res.stdout.strip())
        results.append({"opp": opp, "v3_wins": data["wins_by_bot"][1], "opp_wins": data["wins_by_bot"][0]})
        print(f"Finished {opp}: v3.0 won {data['wins_by_bot'][1]}, opp won {data['wins_by_bot'][0]}", flush=True)
    except Exception as e:
        print(f"Failed {opp}: {e}", flush=True)
        print(res.stdout)

print("Tests complete.", flush=True)
