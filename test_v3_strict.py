import subprocess
import sys
import json

TARGET = "lqq/baseline/gemini-v3.0/bot.py"
# The user specifically mentioned Opus v3. Let's test against bot_hard_v3.py
OPPONENT = "lqq/baseline/claude-opus4p7/bot_hard_v3.py"

print(f"STRICT CHALLENGE: gemini_v3.0 vs claude_opus_v3 (bot_hard_v3.py)", flush=True)
cmd = [
    sys.executable, "lqq/lqq_multi.py", "duel",
    "--bot0", TARGET,
    "--bot1", OPPONENT,
    "--games", "10",
    "--decision-timeout", "1.0"
]

try:
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout.strip())
    print(f"Results: gemini_v3.0 won {data['wins_by_bot'][0]}, claude_v3 won {data['wins_by_bot'][1]}")
except Exception as e:
    print(f"Error: {e}")
