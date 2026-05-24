import subprocess
import sys
import time
import json

TARGET = "lqq/baseline/gemini-v3.0/bot.py"
OPPONENTS = {
    "gemini_v2": "lqq/baseline/gemini-MCTS/bot_v2.py",
    "gemini_v2.1": "lqq/baseline/gemini-MCTS/bot_v2.1.py",
    "claude_hard": "lqq/baseline/claude-opus4p7/bot_hard.py",
    "bot_v3": "lqq/baseline/bot_v3.py",
    "bot_mcts": "lqq/baseline/gemini-MCTS/bot.py"
}

results = []
for name, path in OPPONENTS.items():
    print(f"Testing vs {name}...", flush=True)
    cmd = [
        sys.executable, "lqq/lqq_multi.py", "duel",
        "--bot0", TARGET,
        "--bot1", path,
        "--games", "2",
        "--decision-timeout", "1.0"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout.strip())
        print(f"v3.0 won {data['wins_by_bot'][0]}, {name} won {data['wins_by_bot'][1]}", flush=True)
    except Exception as e:
        print(f"Error testing against {name}: {e}", flush=True)

print("Tests completed.")
