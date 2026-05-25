import sys
sys.path.insert(0, ".")
from env.othello_env import OthelloEnv
from baseline.DeepSeek.bot import _generate_moves_flips, _bit_to_square

env = OthelloEnv(seed=42)
state, info = env.reset()
errors = 0

for step in range(60):
    if env.is_done():
        break
    board = state["board"]
    legal = state["legal_actions"]
    actor = state["actor"]
    
    black = 0
    white = 0
    for r in range(8):
        for c in range(8):
            ch = board[r][c]
            if ch == '.':
                continue
            bit = 1 << ((7 - r) * 8 + c)
            if ch == 'B':
                black |= bit
            elif ch == 'W':
                white |= bit
    
    p, o = (black, white) if actor == 0 else (white, black)
    my_moves = _generate_moves_flips(p, o)
    my_squares = set(_bit_to_square(m) for m, f in my_moves)
    engine_squares = set(legal)
    
    if my_squares != engine_squares:
        # PASS representation differs - engine uses ['PASS'], we use empty
        if engine_squares == {"PASS"} and my_squares == set():
            continue
        errors += 1
        if errors <= 3:
            print(f"ERROR step={step}: eng={sorted(engine_squares)} my={sorted(my_squares)}")
            for row in board:
                print(f"  {row}")
    
    if legal == ["PASS"]:
        state = env.step("PASS")[0]
    else:
        state = env.step(sorted(legal)[0])[0]

if errors == 0:
    print(f"ALL {step+1} POSITIONS CORRECT")
else:
    print(f"{errors} ERRORS out of {step+1} positions")
