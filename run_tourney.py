import subprocess
import sys
import json

BOTS = {
    "gemini_v3.0": "lqq/baseline/gemini-v3.0/bot.py",
    "gemini_v2": "lqq/baseline/gemini-MCTS/bot_v2.py",
    "gemini_v2.1": "lqq/baseline/gemini-MCTS/bot_v2.1.py",
    "claude_hard": "lqq/baseline/claude-opus4p7/bot_hard.py",
    "bot_v3": "lqq/baseline/bot_v3.py"
}

games_per_matchup = 10
timeout = 1.0

results = {name: {"wins": 0, "losses": 0} for name in BOTS}
matchup_results = []

target = "gemini_v3.0"
for opp_name, opp_path in BOTS.items():
    if opp_name == target: continue
    
    print(f"Running {games_per_matchup} games: {target} vs {opp_name}...", flush=True)
    cmd = [
        sys.executable, "lqq/lqq_multi.py", "duel",
        "--bot0", BOTS[target],
        "--bot1", opp_path,
        "--games", str(games_per_matchup),
        "--decision-timeout", str(timeout)
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout.strip())
        
        target_wins = data["wins_by_bot"][0]
        opp_wins = data["wins_by_bot"][1]
        
        results[target]["wins"] += target_wins
        results[target]["losses"] += opp_wins
        results[opp_name]["wins"] += opp_wins
        results[opp_name]["losses"] += target_wins
        
        print(f"  Result: {target} {target_wins} - {opp_wins} {opp_name}", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)

print("\n--- Final Results for gemini_v3.0 ---")
print(f"Total Wins: {results[target]['wins']}")
print(f"Total Losses: {results[target]['losses']}")
print(f"Win Rate: {results[target]['wins'] / (results[target]['wins'] + results[target]['losses']) * 100:.1f}%")
