"""Gemini V4.0 - The Bitboard Sentinel.

Key features:
- Full Bitboard engine.
- Zobrist Hashing for persistent TT across turns.
- Aspiration Windows & PVS.
- Enhanced Evaluation: Distance, Wall economy, and Path connectivity.
- Optimized Wall Candidate Generation: Focused on bottlenecks.
- Move Ordering: TT > Killers > History > Heuristics.
"""

import time
import random

# ---------- Constants ----------
BOARD_SIZE = 9
INF = 1000000
WIN_SCORE = 2000000
MAX_PLY = 120

# Bitmasks
BOARD_MASK = (1 << 81) - 1
ROW0_MASK = 0x1FF
ROW8_MASK = 0x1FF << 72
COL0_MASK = 0
for r in range(9): COL0_MASK |= (1 << (r * 9))
COL8_MASK = COL0_MASK << 8
NOT_COL0_MASK = BOARD_MASK ^ COL0_MASK
NOT_COL8_MASK = BOARD_MASK ^ COL8_MASK

# TT Flags
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# ---------- Zobrist Hashing ----------
random.seed(42)
ZOBRIST_POS = [[random.getrandbits(64) for _ in range(81)] for _ in range(2)]
ZOBRIST_WALL_H = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_WALL_V = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_WALL_REM = [[random.getrandbits(64) for _ in range(11)] for _ in range(2)]
ZOBRIST_TURN = random.getrandbits(64)

def get_zobrist_hash(p0, p1, w0, w1, h_walls, v_walls, turn):
    h = ZOBRIST_POS[0][p0] ^ ZOBRIST_POS[1][p1]
    h ^= ZOBRIST_WALL_REM[0][w0] ^ ZOBRIST_WALL_REM[1][w1]
    if turn == 1: h ^= ZOBRIST_TURN
    
    # h_walls and v_walls are 64-bit integers (8x8)
    temp_h = h_walls
    while temp_h:
        lowbit = temp_h & -temp_h
        idx = lowbit.bit_length() - 1
        h ^= ZOBRIST_WALL_H[idx]
        temp_h ^= lowbit
        
    temp_v = v_walls
    while temp_v:
        lowbit = temp_v & -temp_v
        idx = lowbit.bit_length() - 1
        h ^= ZOBRIST_WALL_V[idx]
        temp_v ^= lowbit
    return h

# ---------- BFS & Pathfinding ----------
def get_dist(pos, target_row, h_mask, v_mask):
    """Bitboard BFS for shortest distance."""
    target_mask = ROW0_MASK if target_row == 0 else ROW8_MASK
    front = 1 << pos
    if front & target_mask: return 0
    visited = front
    dist = 0
    while front:
        dist += 1
        # Up
        up = (front >> 9) & ~h_mask
        # Down
        down = ((front & ~h_mask) << 9) & BOARD_MASK
        # Left
        left = ((front & NOT_COL0_MASK) >> 1) & ~v_mask
        # Right
        right = (((front & NOT_COL8_MASK) & ~v_mask) << 1) & BOARD_MASK
        
        front = (up | down | left | right) & ~visited
        if front & target_mask: return dist
        visited |= front
    return 255 # Should be INF, but using 255 for byte-sized comparison if needed

def get_path(pos, target_row, h_mask, v_mask):
    """Bitboard BFS for shortest path."""
    target_mask = ROW0_MASK if target_row == 0 else ROW8_MASK
    front = 1 << pos
    if front & target_mask: return [pos]
    visited = front
    history = [front]
    dist = 0
    found = False
    while front:
        dist += 1
        up = (front >> 9) & ~h_mask
        down = ((front & ~h_mask) << 9) & BOARD_MASK
        left = ((front & NOT_COL0_MASK) >> 1) & ~v_mask
        right = (((front & NOT_COL8_MASK) & ~v_mask) << 1) & BOARD_MASK
        front = (up | down | left | right) & ~visited
        history.append(front)
        if front & target_mask:
            found = True
            break
        visited |= front
    
    if not found: return []
    
    # Backtrack path
    curr_bit = (front & target_mask) & -(front & target_mask)
    curr = curr_bit.bit_length() - 1
    path = [curr]
    for d in range(dist - 1, -1, -1):
        prev_layer = history[d]
        r, c = curr // 9, curr % 9
        # Check all 4 directions for connectivity and existence in prev_layer
        # Check Up (coming from Down)
        if r < 8 and (prev_layer & (1 << (curr + 9))) and not (h_mask & (1 << curr)):
            curr += 9
        # Check Down (coming from Up)
        elif r > 0 and (prev_layer & (1 << (curr - 9))) and not (h_mask & (1 << (curr - 9))):
            curr -= 9
        # Check Left (coming from Right)
        elif c < 8 and (prev_layer & (1 << (curr + 1))) and not (v_mask & (1 << curr)):
            curr += 1
        # Check Right (coming from Left)
        elif c > 0 and (prev_layer & (1 << (curr - 1))) and not (v_mask & (1 << (curr - 1))):
            curr -= 1
        path.append(curr)
    path.reverse()
    return path

# ---------- Action Coding ----------
# MOVE: 0-7 (UP, DOWN, LEFT, RIGHT, UL, UR, DL, DR)
# WALL: 100 + dir*64 + row*8 + col (dir: 0=H, 1=V)

MOVE_ACTION_NAMES = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP_LEFT", "MOVE_UP_RIGHT", "MOVE_DOWN_LEFT", "MOVE_DOWN_RIGHT"]

def decode_action(action_id):
    if action_id < 8: return MOVE_ACTION_NAMES[action_id]
    val = action_id - 100
    d = val // 64
    rem = val % 64
    r, c = rem // 8, rem % 8
    return f"WALL_{'H' if d == 0 else 'V'}_{r}_{c}"

def encode_wall(d, r, c):
    return 100 + (d * 64) + (r * 8 + c)

# ---------- Search Engine ----------
class Engine:
    def __init__(self):
        self.tt = {} # Hash -> (depth, score, flag, best_action)
        self.history = {} # Action -> score
        self.killers = [[0, 0] for _ in range(MAX_PLY)]
        self.nodes = 0
        self.stop_time = 0
        self.max_tt_size = 500000

    def evaluate(self, p0, p1, w0, w1, h_mask, v_mask, turn):
        d0 = get_dist(p0, 0, h_mask, v_mask)
        d1 = get_dist(p1, 8, h_mask, v_mask)
        
        if d0 >= 255: return -WIN_SCORE + 100 # P0 blocked
        if d1 >= 255: return WIN_SCORE - 100  # P1 blocked
        
        # Base score: distance difference
        # We use turn-aware distance (tempi)
        t0 = d0 * 2 - (1 if turn == 0 else 0)
        t1 = d1 * 2 - (1 if turn == 1 else 0)
        
        score = (t1 - t0) * 100
        
        # Wall economy
        score += (w0 - w1) * 120 # Increased weight
        
        # Positioning: prefer center columns
        score += (4 - abs(p0 % 9 - 4)) * 10
        score -= (4 - abs(p1 % 9 - 4)) * 10
        
        return score

    def _is_wall_legal(self, d, r, c, h_walls, v_walls, h_mask, v_mask):
        idx = r * 8 + c
        if d == 0: # Horizontal
            if (h_walls & (1 << idx)): return False
            if c > 0 and (h_walls & (1 << (idx - 1))): return False
            if c < 7 and (h_walls & (1 << (idx + 1))): return False
            if (v_walls & (1 << idx)): return False
        else: # Vertical
            if (v_walls & (1 << idx)): return False
            if r > 0 and (v_walls & (1 << (idx - 8))): return False
            if r < 7 and (v_walls & (1 << (idx + 8))): return False
            if (h_walls & (1 << idx)): return False
        return True

    def _get_legal_moves(self, my_pos, opp_pos, h_mask, v_mask):
        actions = []
        r, c = my_pos // 9, my_pos % 9
        orow, ocol = opp_pos // 9, opp_pos % 9
        
        # Up
        if r > 0 and not (h_mask & (1 << (my_pos - 9))):
            nxt = my_pos - 9
            if nxt == opp_pos:
                if orow > 0 and not (h_mask & (1 << (opp_pos - 9))): actions.append(0) # Jump UP
                else:
                    if ocol > 0 and not (v_mask & (1 << (opp_pos - 1))): actions.append(4) # UP-LEFT
                    if ocol < 8 and not (v_mask & (1 << opp_pos)): actions.append(5) # UP-RIGHT
            else: actions.append(0)
        # Down
        if r < 8 and not (h_mask & (1 << my_pos)):
            nxt = my_pos + 9
            if nxt == opp_pos:
                if orow < 8 and not (h_mask & (1 << opp_pos)): actions.append(1) # Jump DOWN
                else:
                    if ocol > 0 and not (v_mask & (1 << (opp_pos - 1))): actions.append(6) # DOWN-LEFT
                    if ocol < 8 and not (v_mask & (1 << opp_pos)): actions.append(7) # DOWN-RIGHT
            else: actions.append(1)
        # Left
        if c > 0 and not (v_mask & (1 << (my_pos - 1))):
            nxt = my_pos - 1
            if nxt == opp_pos:
                if ocol > 0 and not (v_mask & (1 << (opp_pos - 1))): actions.append(2) # Jump LEFT
                else:
                    if orow > 0 and not (h_mask & (1 << (opp_pos - 9))): actions.append(4) # UP-LEFT
                    if orow < 8 and not (h_mask & (1 << opp_pos)): actions.append(6) # DOWN-LEFT
            else: actions.append(2)
        # Right
        if c < 8 and not (v_mask & (1 << my_pos)):
            nxt = my_pos + 1
            if nxt == opp_pos:
                if ocol < 8 and not (v_mask & (1 << opp_pos)): actions.append(3) # Jump RIGHT
                else:
                    if orow > 0 and not (h_mask & (1 << (opp_pos - 9))): actions.append(5) # UP-RIGHT
                    if orow < 8 and not (h_mask & (1 << opp_pos)): actions.append(7) # DOWN-RIGHT
            else: actions.append(3)
        return actions

    def _get_focused_walls(self, my_path, opp_path, h_walls, v_walls, h_mask, v_mask):
        candidates = set()
        # Walls blocking opponent's path
        for i in range(len(opp_path) - 1):
            u, v = opp_path[i], opp_path[i+1]
            ur, uc = u // 9, u % 9
            vr, vc = v // 9, v % 9
            if ur == vr: # Horizontal move, block with vertical wall
                left_c = min(uc, vc)
                if left_c < 8:
                    if ur > 0: candidates.add((1, ur - 1, left_c))
                    if ur < 8: candidates.add((1, ur, left_c))
            else: # Vertical move, block with horizontal wall
                top_r = min(ur, vr)
                if top_r < 8:
                    if uc > 0: candidates.add((0, top_r, uc - 1))
                    if uc < 8: candidates.add((0, top_r, uc))
        
        # Walls blocking my own path
        for i in range(min(len(my_path) - 1, 4)):
            u, v = my_path[i], my_path[i+1]
            ur, uc = u // 9, u % 9
            vr, vc = v // 9, v % 9
            if ur == vr:
                left_c = min(uc, vc)
                if left_c < 8:
                    if ur > 0: candidates.add((1, ur - 1, left_c))
                    if ur < 8: candidates.add((1, ur, left_c))
            else:
                top_r = min(ur, vr)
                if top_r < 8:
                    if uc > 0: candidates.add((0, top_r, uc - 1))
                    if uc < 8: candidates.add((0, top_r, uc))
                
        legal_candidates = []
        for d, r, c in candidates:
            if self._is_wall_legal(d, r, c, h_walls, v_walls, h_mask, v_mask):
                legal_candidates.append(encode_wall(d, r, c))
        return legal_candidates

    def _apply_action(self, action, my_pos, opp_pos, w_rem, h_walls, v_walls, h_mask, v_mask):
        if action < 8:
            # Move
            if action == 0: # UP
                nxt = my_pos - 9
                my_pos = nxt - 9 if nxt == opp_pos else nxt
            elif action == 1: # DOWN
                nxt = my_pos + 9
                my_pos = nxt + 9 if nxt == opp_pos else nxt
            elif action == 2: # LEFT
                nxt = my_pos - 1
                my_pos = nxt - 1 if nxt == opp_pos else nxt
            elif action == 3: # RIGHT
                nxt = my_pos + 1
                my_pos = nxt + 1 if nxt == opp_pos else nxt
            elif action == 4: # UP-LEFT
                my_pos = opp_pos - 1 if my_pos - 9 == opp_pos else opp_pos - 9
            elif action == 5: # UP-RIGHT
                my_pos = opp_pos + 1 if my_pos - 9 == opp_pos else opp_pos - 9
            elif action == 6: # DOWN-LEFT
                my_pos = opp_pos - 1 if my_pos + 9 == opp_pos else opp_pos + 9
            elif action == 7: # DOWN-RIGHT
                my_pos = opp_pos + 1 if my_pos + 9 == opp_pos else opp_pos + 9
        else:
            # Wall
            val = action - 100
            d = val // 64
            rem = val % 64
            r, c = rem // 8, rem % 8
            w_rem -= 1
            if d == 0:
                h_walls |= (1 << (r * 8 + c))
                idx9 = r * 9 + c
                h_mask |= (1 << idx9) | (1 << (idx9 + 1))
            else:
                v_walls |= (1 << (r * 8 + c))
                idx9 = r * 9 + c
                v_mask |= (1 << idx9) | (1 << (idx9 + 9))
        return my_pos, w_rem, h_walls, v_walls, h_mask, v_mask

    def search(self, p0, p1, w0, w1, h_walls, v_walls, h_mask, v_mask, turn, depth, ply, alpha, beta):
        self.nodes += 1
        if self.nodes & 1023 == 0 and time.perf_counter() > self.stop_time:
            raise TimeoutError()
            
        # Terminal checks
        if p0 // 9 == 0: return WIN_SCORE - ply
        if p1 // 9 == 8: return -WIN_SCORE + ply
        
        # TT Lookup
        h = get_zobrist_hash(p0, p1, w0, w1, h_walls, v_walls, turn)
        entry = self.tt.get(h)
        if entry and entry[0] >= depth:
            e_depth, e_score, e_flag, e_best = entry
            score = e_score
            if score > WIN_SCORE - 200: score -= ply
            elif score < -WIN_SCORE + 200: score += ply
            
            if e_flag == TT_EXACT: return score
            if e_flag == TT_LOWER and score >= beta: return score
            if e_flag == TT_UPPER and score <= alpha: return score
            
        if depth <= 0:
            return self.evaluate(p0, p1, w0, w1, h_mask, v_mask, turn)
            
        # Move Generation
        my_pos = p0 if turn == 0 else p1
        opp_pos = p1 if turn == 0 else p0
        my_goal = 0 if turn == 0 else 8
        opp_goal = 8 if turn == 0 else 0
        my_wrem = w0 if turn == 0 else w1
        
        my_path = get_path(my_pos, my_goal, h_mask, v_mask)
        opp_path = get_path(opp_pos, opp_goal, h_mask, v_mask)
        
        if not my_path: return -WIN_SCORE + ply if turn == 0 else WIN_SCORE - ply
        if not opp_path: return WIN_SCORE - ply if turn == 0 else -WIN_SCORE + ply
        
        actions = self._get_legal_moves(my_pos, opp_pos, h_mask, v_mask)
        if my_wrem > 0:
            actions.extend(self._get_focused_walls(my_path, opp_path, h_walls, v_walls, h_mask, v_mask))
            
        if not actions:
            return -WIN_SCORE + ply if turn == 0 else WIN_SCORE - ply
            
        # Move Ordering
        tt_best = entry[3] if entry else -1
        k1, k2 = self.killers[ply]
        
        def score_move(a):
            if a == tt_best: return 1000000
            if a == k1: return 900000
            if a == k2: return 800000
            if a < 8:
                # Prioritize move on shortest path
                dest = -1
                if a == 0: dest = my_pos - 9 if my_pos - 9 != opp_pos else my_pos - 18
                elif a == 1: dest = my_pos + 9 if my_pos + 9 != opp_pos else my_pos + 18
                elif a == 2: dest = my_pos - 1 if my_pos - 1 != opp_pos else my_pos - 2
                elif a == 3: dest = my_pos + 1 if my_pos + 1 != opp_pos else my_pos + 2
                # (Skipping diagonal jumps for brevity in heuristic)
                if dest != -1 and len(my_path) > 1 and dest == my_path[1]: return 700000
                return 100000
            else:
                return self.history.get(a, 0)
                
        actions.sort(key=score_move, reverse=True)
        
        best_score = -INF * 2
        best_action = actions[0]
        original_alpha = alpha
        
        for i, a in enumerate(actions):
            # Apply action
            new_my_pos, new_wrem, new_h_walls, new_v_walls, new_h_mask, new_v_mask = self._apply_action(a, my_pos, opp_pos, my_wrem, h_walls, v_walls, h_mask, v_mask)
            
            # Path existence check for walls
            if a >= 100:
                if get_dist(new_my_pos, my_goal, new_h_mask, new_v_mask) >= 255: continue
                if get_dist(opp_pos, opp_goal, new_h_mask, new_v_mask) >= 255: continue
            
            new_p0, new_p1, new_w0, new_w1 = (new_my_pos, opp_pos, new_wrem, w1) if turn == 0 else (opp_pos, new_my_pos, w0, new_wrem)
            
            # PVS
            if i == 0:
                score = -self.search(new_p0, new_p1, new_w0, new_w1, new_h_walls, new_v_walls, new_h_mask, new_v_mask, 1 - turn, depth - 1, ply + 1, -beta, -alpha)
            else:
                score = -self.search(new_p0, new_p1, new_w0, new_w1, new_h_walls, new_v_walls, new_h_mask, new_v_mask, 1 - turn, depth - 1, ply + 1, -alpha - 1, -alpha)
                if alpha < score < beta:
                    score = -self.search(new_p0, new_p1, new_w0, new_w1, new_h_walls, new_v_walls, new_h_mask, new_v_mask, 1 - turn, depth - 1, ply + 1, -beta, -alpha)
            
            if score > best_score:
                best_score = score
                best_action = a
                
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    # Beta cutoff
                    if a >= 100: self.history[a] = self.history.get(a, 0) + depth * depth
                    self.killers[ply][1] = self.killers[ply][0]
                    self.killers[ply][0] = a
                    break
        
        # Store in TT
        flag = TT_EXACT
        if best_score <= original_alpha: flag = TT_UPPER
        elif best_score >= beta: flag = TT_LOWER
        
        tt_score = best_score
        if tt_score > WIN_SCORE - 200: tt_score += ply
        elif tt_score < -WIN_SCORE + 200: tt_score -= ply
        
        if len(self.tt) > self.max_tt_size: self.tt.clear()
        self.tt[h] = (depth, tt_score, flag, best_action)
        
        return best_score

    def choose_action(self, state):
        start_time = time.perf_counter()
        
        # Opening book
        turn_count = state.get("turn", 0)
        me = state.get("player_id", state.get("actor", 0))
        if turn_count < 8 and not state.get("walls"):
            forward = "MOVE_UP" if me == 0 else "MOVE_DOWN"
            if forward in state.get("legal_actions", []): return forward
            
        timeout = state.get("decision_timeout", 1.5)
        self.stop_time = start_time + timeout - 0.1
        self.nodes = 0
        
        # Reset turn-based history/killers but keep TT
        self.killers = [[0, 0] for _ in range(MAX_PLY)]
        # Decay history
        for a in self.history: self.history[a] //= 2
        
        p0r, p0c = state["positions"][0]
        p1r, p1c = state["positions"][1]
        p0, p1 = p0r * 9 + p0c, p1r * 9 + p1c
        w0, w1 = state["walls_remaining"]
        turn = me
        
        h_walls, v_walls, h_mask, v_mask = 0, 0, 0, 0
        for w in state.get("walls", []):
            wr, wc = w["row"], w["col"]
            idx8 = wr * 8 + wc
            idx9 = wr * 9 + wc
            if w["dir"] == "H":
                h_walls |= (1 << idx8)
                h_mask |= (1 << idx9) | (1 << (idx9 + 1))
            else:
                v_walls |= (1 << idx8)
                v_mask |= (1 << idx9) | (1 << (idx9 + 9))
        
        best_a = -1
        last_score = 0
        # Iterative Deepening
        try:
            for depth in range(1, 40):
                # Aspiration Windows
                if depth > 3:
                    alpha, beta = last_score - 50, last_score + 50
                    while True:
                        score = self.search(p0, p1, w0, w1, h_walls, v_walls, h_mask, v_mask, turn, depth, 0, alpha, beta)
                        if score <= alpha: alpha -= 200
                        elif score >= beta: beta += 200
                        else: break
                else:
                    score = self.search(p0, p1, w0, w1, h_walls, v_walls, h_mask, v_mask, turn, depth, 0, -WIN_SCORE*2, WIN_SCORE*2)
                
                last_score = score
                h = get_zobrist_hash(p0, p1, w0, w1, h_walls, v_walls, turn)
                best_a = self.tt[h][3]
                
                if score > WIN_SCORE - 200: break
        except TimeoutError:
            pass
            
        if best_a == -1:
            # Fallback
            my_pos = p0 if turn == 0 else p1
            my_goal = 0 if turn == 0 else 8
            path = get_path(my_pos, my_goal, h_mask, v_mask)
            if len(path) > 1:
                # Determine MOVE action
                r, c = my_pos // 9, my_pos % 9
                nr, nc = path[1] // 9, path[1] % 9
                dr, dc = nr - r, nc - c
                if dr == -1: best_a = 0
                elif dr == 1: best_a = 1
                elif dc == -1: best_a = 2
                elif dc == 1: best_a = 3
            else:
                return state["legal_actions"][0]
        
        res = decode_action(best_a)
        if res not in state["legal_actions"]:
            # This should not happen if search is correct, but for safety:
            for act in state["legal_actions"]:
                if act.startswith("MOVE_"): return act
            return state["legal_actions"][0]
        return res

# Global engine instance for persistent TT
_engine = Engine()

def choose_action(state):
    return _engine.choose_action(state)
