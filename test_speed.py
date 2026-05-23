import time
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("v5", "lqq/baseline/gemini-MCTS/bot_v5.py")
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)

s = v5.State()
s.pos = [76, 4]
s.goals = (0, 8)
s.walls_rem = [10, 10]
s.actor = 0
s.turn = 0
s.h_mask = 0
s.v_mask = 0
s.h_walls = 0
s.v_walls = 0

mcts = v5.MCTS(0.85)

t0 = time.perf_counter()
mcts.search(s)
t1 = time.perf_counter()

print(f"MCTS v5 completed {mcts.root.N} iterations in {t1-t0:.4f} seconds")
