"""Strong LQQ bot v2 with parity, topological analysis, and highly optimized bitboard search.
"""

import time
import collections
import math

SAFETY_MARGIN_SECONDS = 0.45

BOARD_SIZE = 9
INF = 1000
WIN_SCORE = 1000000

# Masks for 81-bit bitboards
RIGHT_MASK = 0
for r in range(9): RIGHT_MASK |= (1 << (r * 9 + 8))
NOT_RIGHT_MASK = ((1 << 81) - 1) ^ RIGHT_MASK

LEFT_MASK = 0
for r in range(9): LEFT_MASK |= (1 << (r * 9))
NOT_LEFT_MASK = ((1 << 81) - 1) ^ LEFT_MASK

def bfs_dist_fast(pos, target_row, h_mask, v_mask):
    front = 1 << pos
    visited = front
    dist = 0
    target_mask = 0x1FF if target_row == 0 else (0x1FF << 72)
    
    while front:
        if front & target_mask: return dist
        up = (front >> 9) & ~h_mask
        down = ((front & ~h_mask) << 9) & 0x1FFFFFFFFFFFFFFFFFFFF
        left = ((front & NOT_LEFT_MASK) >> 1) & ~v_mask
        right = (((front & NOT_RIGHT_MASK) & ~v_mask) << 1) & 0x1FFFFFFFFFFFFFFFFFFFF
        front = (up | down | left | right) & ~visited
        visited |= front
        dist += 1
    return INF

def bfs_path_fast(pos, target_row, h_mask, v_mask):
    front = 1 << pos
    visited = front
    dist = 0
    target_mask = 0x1FF if target_row == 0 else (0x1FF << 72)
    history = [front]
    
    while front:
        if front & target_mask:
            curr_bit = (front & target_mask)
            curr_bit = curr_bit & -curr_bit
            curr = curr_bit.bit_length() - 1
            
            path = [curr]
            for d in range(dist - 1, -1, -1):
                prev_layer = history[d]
                r, c = curr // 9, curr % 9
                if r < 8 and (prev_layer & (1 << (curr + 9))) and not (h_mask & (1 << curr)):
                    curr += 9
                elif r > 0 and (prev_layer & (1 << (curr - 9))) and not (h_mask & (1 << (curr - 9))):
                    curr -= 9
                elif c < 8 and (prev_layer & (1 << (curr + 1))) and not (v_mask & (1 << curr)):
                    curr += 1
                elif c > 0 and (prev_layer & (1 << (curr - 1))) and not (v_mask & (1 << (curr - 1))):
                    curr -= 1
                path.append(curr)
            path.reverse()
            return dist, [(p // 9, p % 9) for p in path]
            
        up = (front >> 9) & ~h_mask
        down = ((front & ~h_mask) << 9) & 0x1FFFFFFFFFFFFFFFFFFFF
        left = ((front & NOT_LEFT_MASK) >> 1) & ~v_mask
        right = (((front & NOT_RIGHT_MASK) & ~v_mask) << 1) & 0x1FFFFFFFFFFFFFFFFFFFF
        
        front = (up | down | left | right) & ~visited
        visited |= front
        history.append(front)
        dist += 1
    return INF, []

class Searcher:
    def __init__(self, time_limit):
        self.start_time = time.perf_counter()
        self.time_limit = time_limit
        self.nodes = 0
        self.tt = {}
        self.killer_moves = {}
        self.path_cache = {}

    def check_time(self):
        self.nodes += 1
        if (self.nodes & 15) == 0:
            if time.perf_counter() - self.start_time > self.time_limit:
                raise TimeoutError()

    def get_path_info(self, pos, target_row, h_mask, v_mask):
        key = (pos, target_row, h_mask, v_mask)
        if key in self.path_cache: return self.path_cache[key]
        res = bfs_path_fast(pos, target_row, h_mask, v_mask)
        self.path_cache[key] = res
        return res

    def search(self, state, depth, alpha, beta, is_maximizing, max_depth_limit=0):
        self.check_time()
        
        player_id = state["player_id"]
        actor = state["actor"]
        
        winner = state.get("winner")
        if winner is not None:
            if winner == player_id: return WIN_SCORE - state["turn"], None
            else: return -WIN_SCORE + state["turn"], None

        if depth == 0:
            return self.evaluate(state), None

        state_key = self.get_state_key(state)
        if state_key in self.tt:
            cached_depth, cached_score, cached_action = self.tt[state_key]
            if cached_depth >= depth: return cached_score, cached_action

        legal = state["legal_actions"]
        if not legal: return (-WIN_SCORE if is_maximizing else WIN_SCORE), None

        ordered_actions = self.order_moves(state, legal, depth)
        
        best_action = None
        if is_maximizing:
            max_eval = -float('inf')
            for action in ordered_actions:
                child_state = self.apply_action(state, action)
                eval_score, _ = self.search(child_state, depth - 1, alpha, beta, False, max_depth_limit)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self.add_killer(depth, actor, action)
                    break
                # Special root check
                if depth == max_depth_limit and time.perf_counter() - self.start_time > self.time_limit:
                    break
            self.tt[state_key] = (depth, max_eval, best_action)
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in ordered_actions:
                child_state = self.apply_action(state, action)
                eval_score, _ = self.search(child_state, depth - 1, alpha, beta, True, max_depth_limit)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    self.add_killer(depth, actor, action)
                    break
            self.tt[state_key] = (depth, min_eval, best_action)
            return min_eval, best_action

    def get_state_key(self, state):
        pos = (tuple(state["positions"][0]), tuple(state["positions"][1]))
        walls = tuple(sorted((w["dir"], w["row"], w["col"]) for w in state["walls"]))
        return (pos, walls, state["actor"])

    def add_killer(self, depth, actor, action):
        key = (depth, actor)
        if key not in self.killer_moves: self.killer_moves[key] = []
        if action not in self.killer_moves[key]:
            self.killer_moves[key].insert(0, action)
            self.killer_moves[key] = self.killer_moves[key][:2]

    def order_moves(self, state, actions, depth):
        scored = []
        actor = state["actor"]
        killers = self.killer_moves.get((depth, actor), [])
        
        for a in actions:
            score = 0
            if a in killers: score = 2000
            elif a.startswith("MOVE_"): score = 1000
            scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a[1] for a in scored]

    def apply_action(self, state, action):
        new_state = {
            "player_id": state["player_id"], "actor": 1 - state["actor"],
            "positions": [list(p) for p in state["positions"]],
            "walls_remaining": list(state["walls_remaining"]),
            "walls": [dict(w) for w in state["walls"]],
            "goal_rows": state["goal_rows"], "turn": state["turn"] + 1, "board_size": state["board_size"]
        }
        actor = state["actor"]
        if action.startswith("MOVE_"):
            dest = self.get_move_dest(state, actor, action)
            if dest:
                new_state["positions"][actor] = list(dest)
                if dest[0] == new_state["goal_rows"][actor]: new_state["winner"] = actor
        elif action.startswith("WALL_"):
            parts = action.split("_")
            new_state["walls"].append({"dir": parts[1], "row": int(parts[2]), "col": int(parts[3])})
            new_state["walls_remaining"][actor] -= 1
        
        # Shallow legal actions for children
        new_state["legal_actions"] = state["legal_actions"] if state["actor"] == new_state["actor"] else ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]
        return new_state

    def get_move_dest(self, state, pid, action):
        deltas = {"MOVE_UP": (-1, 0), "MOVE_DOWN": (1, 0), "MOVE_LEFT": (0, -1), "MOVE_RIGHT": (0, 1),
                  "MOVE_UP_LEFT": (-1, -1), "MOVE_UP_RIGHT": (-1, 1), "MOVE_DOWN_LEFT": (1, -1), "MOVE_DOWN_RIGHT": (1, 1)}
        dr, dc = deltas.get(action, (0, 0))
        p = state["positions"][pid]
        return (p[0] + dr, p[1] + dc)

    def evaluate(self, state):
        me = state["player_id"]
        opp = 1 - me
        
        h_mask, v_mask = 0, 0
        for w in state["walls"]:
            idx = w["row"] * 9 + w["col"]
            if w["dir"] == "H": h_mask |= (1 << idx) | (1 << (idx + 1))
            else: v_mask |= (1 << idx) | (1 << (idx + 9))
            
        my_pos = state["positions"][me][0] * 9 + state["positions"][me][1]
        opp_pos = state["positions"][opp][0] * 9 + state["positions"][opp][1]
        
        my_dist, my_path = self.get_path_info(my_pos, state["goal_rows"][me], h_mask, v_mask)
        opp_dist, _ = self.get_path_info(opp_pos, state["goal_rows"][opp], h_mask, v_mask)
        
        if my_dist >= INF: return -WIN_SCORE // 2
        if opp_dist >= INF: return WIN_SCORE // 2

        my_turns = my_dist * 2 - (1 if state["actor"] == me else 0)
        opp_turns = opp_dist * 2 - (1 if state["actor"] == opp else 0)
        score = (opp_turns - my_turns) * 1000
        
        score += (state["walls_remaining"][me] - state["walls_remaining"][opp]) * 150
        
        # Bottleneck
        if state["walls_remaining"][opp] > 0 and len(my_path) > 1:
            p1 = my_path[0][0] * 9 + my_path[0][1]
            p2 = my_path[1][0] * 9 + my_path[1][1]
            temp_h, temp_v = h_mask, v_mask
            if abs(p1 - p2) == 9: temp_h |= (1 << min(p1, p2))
            else: temp_v |= (1 << min(p1, p2))
            alt_dist = bfs_dist_fast(p1, state["goal_rows"][me], temp_h, temp_v)
            score -= max(0, alt_dist - my_dist) * 80

        # Center
        score += (4 - abs(state["positions"][me][1] - 4)) * 30
        score -= (4 - abs(state["positions"][opp][1] - 4)) * 20
        
        return score

class Bot:
    name = "gemini_pro_v2"

    def choose_action(self, state):
        timeout = state.get("decision_timeout") or state.get("time_limit")
        limit = max(0.05, float(timeout) - SAFETY_MARGIN_SECONDS) if timeout else 0.75
        
        searcher = Searcher(limit)
        legal_actions = state["legal_actions"]
        if not legal_actions: return ""
        if len(legal_actions) == 1: return legal_actions[0]

        best_action = legal_actions[0]
        try:
            for depth in range(1, 15):
                score, action = searcher.search(state, depth, -float('inf'), float('inf'), True, depth)
                if action: best_action = action
                if score >= WIN_SCORE - 1000: break
                if time.perf_counter() - searcher.start_time > limit * 0.4: break
        except TimeoutError:
            pass

        return best_action

def choose_action(state):
    return Bot().choose_action(state)
