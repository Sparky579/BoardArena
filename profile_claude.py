import cProfile
import pstats
import importlib.util

spec = importlib.util.spec_from_file_location("claude_hard", "lqq/baseline/claude-opus4p7/bot_hard.py")
claude_hard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(claude_hard)

state_dict = {
    "player_id": 0,
    "actor": 0,
    "turn": 10,
    "positions": [[4, 4], [4, 5]],
    "goal_rows": [0, 8],
    "walls_remaining": [8, 8],
    "walls": [],
    "legal_actions": ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]
}

b = claude_hard.Bot()
b.time_limit = 1.0

profiler = cProfile.Profile()
profiler.enable()
action = b.choose_action(state_dict)
profiler.disable()

print("Chosen action:", action)
stats = pstats.Stats(profiler).sort_stats('tottime')
stats.print_stats(15)
