"""Bot-vs-Bot tournament runner. Pits developer bot against any opponent bot."""
import sys
import time
sys.path.insert(0, ".")

from env.othello_env import OthelloEnv, load_bot

def play_one_game(bot_a, bot_b, seed, decision_timeout):
    """Play one game: bot_a is black, bot_b is white."""
    env = OthelloEnv(seed=seed)
    state = env.reset()
    
    while not env.is_done():
        if env.actor == 0:
            s = env.state(0)
            s["decision_timeout"] = decision_timeout
            action = bot_a.choose_action(s)
        else:
            s = env.state(1)
            s["decision_timeout"] = decision_timeout
            action = bot_b.choose_action(s)
        
        state, reward, terminated, truncated, info = env.step(action)
    
    final_state = env.state(0)
    scores = final_state["scores"]
    black_score = scores[0]
    white_score = scores[1]
    return black_score, white_score, black_score > white_score, white_score > black_score

def run_match(my_bot_path, opp_bot_path, games=10, decision_timeout=1.5, base_seed=0):
    my_bot = load_bot(my_bot_path)
    opp_bot = load_bot(opp_bot_path)
    
    my_name = getattr(my_bot, 'name', 'MyBot')
    opp_name = getattr(opp_bot, 'name', 'OppBot')
    
    my_wins = 0
    opp_wins = 0
    draws = 0
    
    t0 = time.time()
    
    for i in range(games):
        seed = base_seed + i
        if i % 2 == 0:
            bs, ws, bw, ww = play_one_game(my_bot, opp_bot, seed, decision_timeout)
            if bw:
                my_wins += 1
            elif ww:
                opp_wins += 1
            else:
                draws += 1
        else:
            bs, ws, bw, ww = play_one_game(opp_bot, my_bot, seed, decision_timeout)
            if ww:
                my_wins += 1
            elif bw:
                opp_wins += 1
            else:
                draws += 1
        
        elapsed = time.time() - t0
        print(f"[{i+1}/{games}] {my_name} wins={my_wins} {opp_name} wins={opp_wins} draws={draws}  ({elapsed:.1f}s)")
    
    elapsed = time.time() - t0
    win_rate = my_wins / games if games > 0 else 0
    print(f"\n=== {my_name} vs {opp_name} ===")
    print(f"Games: {games}, Win rate: {win_rate:.1%} ({my_wins}/{games})")
    print(f"Opp wins: {opp_wins}, Draws: {draws}")
    print(f"Time: {elapsed:.1f}s")
    
    return win_rate

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--my-bot", default="baseline/DeepSeek/bot.py")
    parser.add_argument("--opp-bot", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_match(args.my_bot, args.opp_bot, args.games, args.timeout, args.seed)
