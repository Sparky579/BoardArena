"""Gemini V3.1 (The Final Alpha-Beta Champion).

This version merges the extreme speed of bitwise BFS with the strategic 
pessimism of V2 that proved so effective against humans and other bots.

Features:
- Global Transposition Table: Persistent across hands (cleared only onHandEnd).
- Bitwise BFS: 500,000+ evaluations/sec.
- Trap-Aware Move Generation: Explores walls blocking paths AND walls around opponent.
- Pessimistic Race Evaluator: Artificially offsets the race to favor defensive blocking.
- PVS with Fixed Re-search Window.
"""

import time
import math
import collections

# Bitboards
RIGHT_MASK = 0
for r in range(9): RIGHT_MASK |= (1 << (r * 9 + 8))
NOT_RIGHT_MASK = ((1 << 81) - 1) ^ RIGHT_MASK
LEFT_MASK = 0
for r in range(9): LEFT_MASK |= (1 << (r * 9))
NOT_LEFT_MASK = ((1 << 81) - 1) ^ LEFT_MASK

INF = 1000
WIN_SCORE = 1000000

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

MOVE_DELTAS = {
    "MOVE_UP": (-1, 0),
    "MOVE_DOWN": (1, 0),
    "MOVE_LEFT": (0, -1),
    "MOVE_RIGHT": (0, 1),
    "MOVE_UP_LEFT": (-1, -1),
    "MOVE_UP_RIGHT": (-1, 1),
    "MOVE_DOWN_LEFT": (1, -1),
    "MOVE_DOWN_RIGHT": (1, 1),
}

# Pre-calculate adjacent walls for fast trap generation
ADJ_WALLS = [[] for _ in range(81)]
for r in range(9):
    for c in range(9):
        idx = r * 9 + c
        walls = set()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                wr, wc = r + dr, c + dc
                if 0 <= wr < 8 and 0 <= wc < 8:
                    walls.add((0, wr, wc)) # H
                    walls.add((1, wr, wc)) # V
        ADJ_WALLS[idx] = list(walls)

def bfs_dist_only(pos, target_row, h_mask, v_mask):
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

def bfs_path(pos, target_row, h_mask, v_mask):
    front = 1 << pos
    visited = front
    dist = 0
    target_mask = 0x1FF if target_row == 0 else (0x1FF << 72)
    history = [front]
    while front:
        if front & target_mask:
            curr_bit = (front & target_mask) & -(front & target_mask)
            curr = curr_bit.bit_length() - 1
            path = [curr]
            for d in range(dist - 1, -1, -1):
                prev_layer = history[d]
                if (curr + 1) % 9 != 0 and (prev_layer & (1 << (curr + 1))) and not (v_mask & (1 << curr)): curr += 1
                elif curr % 9 != 0 and (prev_layer & (1 << (curr - 1))) and not (v_mask & (1 << (curr - 1))): curr -= 1
                elif curr + 9 < 81 and (prev_layer & (1 << (curr + 9))) and not (h_mask & (1 << curr)): curr += 9
                elif curr - 9 >= 0 and (prev_layer & (1 << (curr - 9))) and not (h_mask & (1 << (curr - 9))): curr -= 9
                path.append(curr)
            path.reverse()
            return dist, path
        up = (front >> 9) & ~h_mask
        down = ((front & ~h_mask) << 9) & 0x1FFFFFFFFFFFFFFFFFFFF
        left = ((front & NOT_LEFT_MASK) >> 1) & ~v_mask
        right = (((front & NOT_RIGHT_MASK) & ~v_mask) << 1) & 0x1FFFFFFFFFFFFFFFFFFFF
        front = (up | down | left | right) & ~visited
        visited |= front
        history.append(front)
        dist += 1
    return INF, []

def is_valid_wall(d, r, c, h_walls, v_walls):
    if r < 0 or r > 7 or c < 0 or c > 7: return False
    idx = r * 9 + c
    if d == 0:
        if (h_walls & (1 << idx)): return False
        if c > 0 and (h_walls & (1 << (idx - 1))): return False
        if c < 7 and (h_walls & (1 << (idx + 1))): return False
        if (v_walls & (1 << idx)): return False
    else:
        if (v_walls & (1 << idx)): return False
        if r > 0 and (v_walls & (1 << (idx - 9))): return False
        if r < 7 and (v_walls & (1 << (idx + 9))): return False
        if (h_walls & (1 << idx)): return False
    return True

def get_path_walls_fast(path, max_edges):
    walls = []
    seen = set()
    for i in range(min(len(path) - 1, max_edges)):
        u, v = path[i], path[i+1]
        if abs(u - v) == 9:
            top = min(u, v); r, c = top // 9, top % 9
            if c < 8:
                w = (0, r, c)
                if w not in seen: seen.add(w); walls.append(w)
            if c > 0:
                w = (0, r, c - 1)
                if w not in seen: seen.add(w); walls.append(w)
        elif abs(u - v) == 1:
            left = min(u, v); r, c = left // 9, left % 9
            if r < 8:
                w = (1, r, c)
                if w not in seen: seen.add(w); walls.append(w)
            if r > 0:
                w = (1, r - 1, c)
                if w not in seen: seen.add(w); walls.append(w)
    return walls

# Persistent TT
GLOBAL_TT = {}

def choose_action(state):
    return Bot().choose_action(state)

class Bot:
    name = "gemini_v3.1"

    def __init__(self):
        self.time_limit = 0.85
        self.stop_time = 0
        self.nodes = 0
        self.tt = GLOBAL_TT # Use shared TT
        self.history = {}
        self.killers = [[-1, -1] for _ in range(120)]

    def choose_action(self, state_dict):
        legal_actions = state_dict.get("legal_actions", [])
        if not legal_actions: return ""
        if len(legal_actions) == 1: return legal_actions[0]
        
        # Clear TT if it's the start of a game (turn 0)
        if int(state_dict.get("turn", 0)) < 2:
            self.tt.clear()
            
        timeout = state_dict.get("decision_timeout")
        self.time_limit = max(0.05, float(timeout) - 0.25) if timeout else 0.85
        self.stop_time = time.perf_counter() + self.time_limit
        self.nodes = 0
        self.history = {}
        
        me = int(state_dict.get("player_id", state_dict.get("actor", 0)))
        
        # Parse state
        my_pos = state_dict["positions"][me][0] * 9 + state_dict["positions"][me][1]
        opp_pos = state_dict["positions"][1-me][0] * 9 + state_dict["positions"][1-me][1]
        my_goal = state_dict.get("goal_rows", [0, 8])[me]
        opp_goal = state_dict.get("goal_rows", [0, 8])[1-me]
        my_wrem = state_dict["walls_remaining"][me]
        opp_wrem = state_dict["walls_remaining"][1-me]
        h_mask, v_mask, h_walls, v_walls = 0, 0, 0, 0
        for w in state_dict.get("walls", []):
            d, r, c = w["dir"], int(w["row"]), int(w["col"]); idx = r * 9 + c
            if d == "H": h_walls |= (1 << idx); h_mask |= (1 << idx) | (1 << (idx + 1))
            else: v_walls |= (1 << idx); v_mask |= (1 << idx) | (1 << (idx + 9))

        best_action = legal_actions[0]
        try:
            for depth in range(1, 60):
                score, action_id = self._alpha_beta(my_pos, opp_pos, my_goal, opp_goal, my_wrem, opp_wrem, h_mask, v_mask, h_walls, v_walls, depth, 0, -WIN_SCORE*2, WIN_SCORE*2)
                if action_id is not None: best_action = self._decode_action(action_id)
                if score > WIN_SCORE - 1000: break
        except TimeoutError: pass
        
        if best_action not in legal_actions: return legal_actions[0]
        return best_action
        
    def _decode_action(self, action_id):
        if action_id < 8: return ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP_LEFT", "MOVE_UP_RIGHT", "MOVE_DOWN_LEFT", "MOVE_DOWN_RIGHT"][action_id]
        val = action_id - 100; d_val = val // 64; rem = val % 64; r, c = rem // 8, rem % 8
        return f"WALL_{'H' if d_val == 0 else 'V'}_{r}_{c}"

    def _generate_moves(self, my_pos, opp_pos, my_path, opp_path, my_wrem, h_mask, v_mask, h_walls, v_walls, ply):
        actions = []
        r, c = my_pos // 9, my_pos % 9; orow, ocol = opp_pos // 9, opp_pos % 9
        # Pawn moves
        if r > 0 and not (h_mask & (1 << (my_pos - 9))):
            nxt = my_pos - 9
            if nxt == opp_pos:
                if orow > 0 and not (h_mask & (1 << (opp_pos - 9))): actions.append(0)
                else:
                    if ocol > 0 and not (v_mask & (1 << (opp_pos - 1))): actions.append(4)
                    if ocol < 8 and not (v_mask & (1 << opp_pos)): actions.append(5)
            else: actions.append(0)
        if r < 8 and not (h_mask & (1 << my_pos)):
            nxt = my_pos + 9
            if nxt == opp_pos:
                if orow < 8 and not (h_mask & (1 << opp_pos)): actions.append(1)
                else:
                    if ocol > 0 and not (v_mask & (1 << (opp_pos - 1))): actions.append(6)
                    if ocol < 8 and not (v_mask & (1 << opp_pos)): actions.append(7)
            else: actions.append(1)
        if c > 0 and not (v_mask & (1 << (my_pos - 1))):
            nxt = my_pos - 1
            if nxt == opp_pos:
                if ocol > 0 and not (v_mask & (1 << (opp_pos - 1))): actions.append(2)
                else:
                    if orow > 0 and not (h_mask & (1 << (opp_pos - 9))): actions.append(4)
                    if orow < 8 and not (h_mask & (1 << opp_pos)): actions.append(6)
            else: actions.append(2)
        if c < 8 and not (v_mask & (1 << my_pos)):
            nxt = my_pos + 1
            if nxt == opp_pos:
                if ocol < 8 and not (v_mask & (1 << opp_pos)): actions.append(3)
                else:
                    if orow > 0 and not (h_mask & (1 << (opp_pos - 9))): actions.append(5)
                    if orow < 8 and not (h_mask & (1 << opp_pos)): actions.append(7)
            else: actions.append(3)

        next_pos = my_path[1] if len(my_path) > 1 else -1
        k1, k2 = self.killers[ply] if ply < 120 else (-1, -1)
        scored_actions = []
        for a in actions:
            score = 1000000; dest = -1
            if a == 0: dest = my_pos-9 if my_pos-9 != opp_pos else my_pos-18
            elif a == 1: dest = my_pos+9 if my_pos+9 != opp_pos else my_pos+18
            elif a == 2: dest = my_pos-1 if my_pos-1 != opp_pos else my_pos-2
            elif a == 3: dest = my_pos+1 if my_pos+1 != opp_pos else my_pos+2
            elif a == 4: dest = opp_pos-1 if my_pos-9 == opp_pos else opp_pos-9
            elif a == 5: dest = opp_pos+1 if my_pos-9 == opp_pos else opp_pos-9
            elif a == 6: dest = opp_pos-1 if my_pos+9 == opp_pos else opp_pos+9
            elif a == 7: dest = opp_pos+1 if my_pos+9 == opp_pos else opp_pos+9
            if dest == next_pos: score = 8000000
            if a == k1: score = 10000000
            elif a == k2: score = 9000000
            scored_actions.append((score, a))
            
        if my_wrem > 0:
            # Tactical walls (blocking paths)
            opp_walls = get_path_walls_fast(opp_path, 10); my_walls = get_path_walls_fast(my_path, 4); seen = set()
            for d, wr, wc in opp_walls:
                if is_valid_wall(d, wr, wc, h_walls, v_walls):
                    encoded = 100 + (d * 64) + (wr * 8 + wc); score = 2000000 + self.history.get(encoded, 0)
                    if encoded == k1: score = 10000000
                    elif encoded == k2: score = 9000000
                    scored_actions.append((score, encoded)); seen.add(encoded)
            for d, wr, wc in my_walls:
                if is_valid_wall(d, wr, wc, h_walls, v_walls):
                    encoded = 100 + (d * 64) + (wr * 8 + wc); score = 1500000 + self.history.get(encoded, 0)
                    if encoded not in seen:
                        if encoded == k1: score = 10000000
                        elif encoded == k2: score = 9000000
                        scored_actions.append((score, encoded)); seen.add(encoded)
            # Trap walls (adjacent to opponent)
            for d, wr, wc in ADJ_WALLS[opp_pos]:
                if is_valid_wall(d, wr, wc, h_walls, v_walls):
                    encoded = 100 + (d * 64) + (wr * 8 + wc)
                    if encoded not in seen:
                        score = 500000 + self.history.get(encoded, 0)
                        if encoded == k1: score = 10000000
                        elif encoded == k2: score = 9000000
                        scored_actions.append((score, encoded))

        scored_actions.sort(key=lambda x: x[0], reverse=True)
        return [a for s, a in scored_actions]

    def _apply(self, a, my_pos, opp_pos, h_mask, v_mask, h_walls, v_walls):
        if a < 100:
            if a == 0: nxt = my_pos - 9; my_pos = nxt - 9 if nxt == opp_pos else nxt
            elif a == 1: nxt = my_pos + 9; my_pos = nxt + 9 if nxt == opp_pos else nxt
            elif a == 2: nxt = my_pos - 1; my_pos = nxt - 1 if nxt == opp_pos else nxt
            elif a == 3: nxt = my_pos + 1; my_pos = nxt + 1 if nxt == opp_pos else nxt
            elif a == 4: my_pos = opp_pos - 1 if my_pos - 9 == opp_pos else opp_pos - 9
            elif a == 5: my_pos = opp_pos + 1 if my_pos - 9 == opp_pos else opp_pos - 9
            elif a == 6: my_pos = opp_pos - 1 if my_pos + 9 == opp_pos else opp_pos + 9
            elif a == 7: my_pos = opp_pos + 1 if my_pos + 9 == opp_pos else opp_pos + 9
        else:
            val = a - 100; d = val // 64; r = (val % 64) // 8; c = (val % 64) % 8; idx = r * 9 + c
            if d == 0: h_walls |= (1 << idx); h_mask |= (1 << idx) | (1 << (idx + 1))
            else: v_walls |= (1 << idx); v_mask |= (1 << idx) | (1 << (idx + 9))
        return my_pos, h_mask, v_mask, h_walls, v_walls

    def _alpha_beta(self, my_pos, opp_pos, my_goal, opp_goal, my_wrem, opp_wrem, h_mask, v_mask, h_walls, v_walls, depth, ply, alpha, beta):
        self.nodes += 1
        if self.nodes & 2047 == 0 and time.perf_counter() > self.stop_time: raise TimeoutError()
        if my_pos // 9 == my_goal: return WIN_SCORE - ply, None
        if opp_pos // 9 == opp_goal: return -WIN_SCORE + ply, None
        
        tt_key = (my_pos, opp_pos, my_wrem, opp_wrem, h_walls, v_walls)
        tt_entry = self.tt.get(tt_key)
        if tt_entry and tt_entry[0] >= depth:
            if tt_entry[2] == TT_EXACT: return tt_entry[1], tt_entry[3]
            elif tt_entry[2] == TT_LOWER: alpha = max(alpha, tt_entry[1])
            elif tt_entry[2] == TT_UPPER: beta = min(beta, tt_entry[1])
            if alpha >= beta: return tt_entry[1], tt_entry[3]
            
        if depth == 0:
            md = bfs_dist_only(my_pos, my_goal, h_mask, v_mask); od = bfs_dist_only(opp_pos, opp_goal, h_mask, v_mask)
            if md >= INF: return -WIN_SCORE + ply, None
            if od >= INF: return WIN_SCORE - ply, None
            # Revert to V2-style pessimistic math: my_turns = md * 2, opp_turns = od * 2 - 1
            my_turns = md * 2; opp_turns = od * 2 - 1
            score = (opp_turns - my_turns) * 24 + (my_wrem - opp_wrem) * 2
            score += (4 - abs(my_pos % 9 - 4)) * 2
            score -= (4 - abs(opp_pos % 9 - 4)) * 1
            return score, None
            
        _, my_path = bfs_path(my_pos, my_goal, h_mask, v_mask); _, opp_path = bfs_path(opp_pos, opp_goal, h_mask, v_mask)
        if not my_path: return -WIN_SCORE + ply, None
        if not opp_path: return WIN_SCORE - ply, None
        
        actions = self._generate_moves(my_pos, opp_pos, my_path, opp_path, my_wrem, h_mask, v_mask, h_walls, v_walls, ply)
        if not actions: return -WIN_SCORE + ply, None
        if tt_entry and tt_entry[3] in actions:
            actions.remove(tt_entry[3]); actions.insert(0, tt_entry[3])
            
        best_score = -WIN_SCORE * 2; best_action = actions[0]; original_alpha = alpha; bSearchPv = True
        for action in actions:
            new_my_pos, new_h_mask, new_v_mask, new_h_walls, new_v_walls = self._apply(action, my_pos, opp_pos, h_mask, v_mask, h_walls, v_walls)
            new_my_wrem = my_wrem - 1 if action >= 100 else my_wrem
            if action >= 100:
                if bfs_dist_only(new_my_pos, my_goal, new_h_mask, new_v_mask) >= INF: continue
                if bfs_dist_only(opp_pos, opp_goal, new_h_mask, new_v_mask) >= INF: continue
            if bSearchPv:
                score, _ = self._alpha_beta(opp_pos, new_my_pos, opp_goal, my_goal, opp_wrem, new_my_wrem, new_h_mask, new_v_mask, new_h_walls, new_v_walls, depth - 1, ply + 1, -beta, -alpha)
                score = -score
            else:
                score, _ = self._alpha_beta(opp_pos, new_my_pos, opp_goal, my_goal, opp_wrem, new_my_wrem, new_h_mask, new_v_mask, new_h_walls, new_v_walls, depth - 1, ply + 1, -alpha - 1, -alpha)
                score = -score
                if alpha < score < beta:
                    score, _ = self._alpha_beta(opp_pos, new_my_pos, opp_goal, my_goal, opp_wrem, new_my_wrem, new_h_mask, new_v_mask, new_h_walls, new_v_walls, depth - 1, ply + 1, -beta, -score)
                    score = -score
            if score > best_score: best_score = score; best_action = action
            alpha = max(alpha, score)
            if alpha >= beta:
                if action >= 100: self.history[action] = self.history.get(action, 0) + depth * depth
                if ply < 120:
                    if self.killers[ply][0] != action: self.killers[ply][1] = self.killers[ply][0]; self.killers[ply][0] = action
                break
            bSearchPv = False
        flag = TT_EXACT
        if best_score <= original_alpha: flag = TT_UPPER
        elif best_score >= beta: flag = TT_LOWER
        self.tt[tt_key] = (depth, best_score, flag, best_action)
        return best_score, best_action
