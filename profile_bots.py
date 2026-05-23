import cProfile
import pstats
import importlib.util
import time

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v2 = load_module("v2", "lqq/baseline/gemini-MCTS/bot_v2.py")
v6 = load_module("v6", "lqq/baseline/gemini-MCTS/bot_v6.py")

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

def profile_bot(mod, name):
    print(f"\nProfiling {name}...")
    bot = mod.Bot()
    bot.mcts.time_limit = 1.0
    
    profiler = cProfile.Profile()
    profiler.enable()
    bot.choose_action(state_dict)
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats('tottime')
    print(f"--- Top 10 functions in {name} ---")
    stats.print_stats(15)
    print(f"Total MCTS iterations: {bot.mcts.root.N}")

if __name__ == "__main__":
    profile_bot(v2, "Bot V2")
    profile_bot(v6, "Bot V6")
