"""Gemini V4.2 - The Precision Engine.

Key features:
- Optimized search loop with minimized overhead.
- Distance-difference based wall scoring.
- Refined evaluation weights (Tempi vs Walls).
- Late Move Reductions (LMR).
- Persistent TT with Zobrist Hashing.
- Opening Book.
"""

import time
import random

# ---------- Constants ----------
INF = 1000000
WIN_SCORE = 2000000
MAX_PLY = 128

BOARD_MASK = (1 << 81) - 1
ROW0_MASK = 0x1FF
ROW8_MASK = 0x1FF << 72
COL0_MASK = sum(1 << (r * 9) for r in range(9))
COL8_MASK = COL0_MASK << 8
NOT_COL0_MASK = BOARD_MASK ^ COL0_MASK
NOT_COL8_MASK = BOARD_MASK ^ COL8_MASK

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# ---------- Zobrist ----------
random.seed(42)
ZOBRIST_POS = [[random.getrandbits(64) for _ in range(81)] for _ in range(2)]
ZOBRIST_WALL_H = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_WALL_V = [random.getrandbits(64) for _ in range(64)]
ZOBRIST_WALL_REM = [[random.getrandbits(64) for _ in range(11)] for _ in range(2)]
ZOBRIST_TURN = random.getrandbits(64)

def get_zobrist_hash(p0, p1, w0, w1, h_walls, v_walls, turn):
    h = ZOBRIST_POS[0][p0] ^ ZOBRIST_POS[1][p1] ^ ZOBRIST_WALL_REM[0][w0] ^ ZOBRIST_WALL_REM[1][w1]
    if turn == 1: h ^= ZOBRIST_TURN
    t_h, t_v = h_walls, v_walls
    while t_h:
        lowbit = t_h & -t_h; h ^= ZOBRIST_WALL_H[lowbit.bit_length() - 1]; t_h ^= lowbit
    while t_v:
        lowbit = t_v & -t_v; h ^= ZOBRIST_WALL_V[lowbit.bit_length() - 1]; t_v ^= lowbit
    return h

# ---------- BFS ----------
def get_dist(pos, target_row, h_mask, v_mask):
    target_mask = ROW0_MASK if target_row == 0 else ROW8_MASK
    front = 1 << pos
    if front & target_mask: return 0
    visited, dist = front, 0
    while front:
        dist += 1
        up = (front >> 9) & ~h_mask
        down = ((front & ~h_mask) << 9) & BOARD_MASK
        left = ((front & NOT_COL0_MASK) >> 1) & ~v_mask
        right = (((front & NOT_COL8_MASK) & ~v_mask) << 1) & BOARD_MASK
        front = (up | down | left | right) & ~visited
        if front & target_mask: return dist
        visited |= front
    return 255

def get_path(pos, target_row, h_mask, v_mask):
    target_mask = ROW0_MASK if target_row == 0 else ROW8_MASK
    front = 1 << pos
    if front & target_mask: return [pos]
    visited, history, dist = front, [front], 0
    while front:
        dist += 1
        up = (front >> 9) & ~h_mask
        down = ((front & ~h_mask) << 9) & BOARD_MASK
        left = ((front & NOT_COL0_MASK) >> 1) & ~v_mask
        right = (((front & NOT_COL8_MASK) & ~v_mask) << 1) & BOARD_MASK
        front = (up | down | left | right) & ~visited
        history.append(front)
        if front & target_mask:
            curr = (front & target_mask).bit_length() - 1
            path = [curr]
            for d in range(dist - 1, -1, -1):
                prev, r, c = history[d], curr // 9, curr % 9
                if r < 8 and (prev & (1 << (curr + 9))) and not (h_mask & (1 << curr)): curr += 9
                elif r > 0 and (prev & (1 << (curr - 9))) and not (h_mask & (1 << (curr - 9))): curr -= 9
                elif c < 8 and (prev & (1 << (curr + 1))) and not (v_mask & (1 << curr)): curr += 1
                elif c > 0 and (prev & (1 << (curr - 1))) and not (v_mask & (1 << (curr - 1))): curr -= 1
                path.append(curr)
            path.reverse(); return path
        visited |= front
    return []

# ---------- Search ----------
class Engine:
    def __init__(self):
        self.tt, self.history, self.killers = {}, {}, [[0, 0] for _ in range(MAX_PLY)]
        self.nodes, self.stop_time, self.max_tt_size = 0, 0, 1000000

    def evaluate(self, p0, p1, w0, w1, h_mask, v_mask, turn):
        d0, d1 = get_dist(p0, 0, h_mask, v_mask), get_dist(p1, 8, h_mask, v_mask)
        if d0 >= 255: return -WIN_SCORE + 100
        if d1 >= 255: return WIN_SCORE - 100
        race0, race1 = d0 * 2 - (1 if turn == 0 else 0), d1 * 2 - (1 if turn == 1 else 0)
        score = (race1 - race0) * 50 + (w0 - w1) * 60
        score += (4 - abs(p0 % 9 - 4)) * 2 - (4 - abs(p1 % 9 - 4)) * 2
        return score if turn == 0 else -score

    def search(self, p0, p1, w0, w1, h_walls, v_walls, h_mask, v_mask, turn, depth, ply, alpha, beta):
        self.nodes += 1
        if self.nodes & 1023 == 0 and time.perf_counter() > self.stop_time: raise TimeoutError()
        if p0 < 9: return WIN_SCORE - ply
        if p1 >= 72: return -WIN_SCORE + ply
        
        h = get_zobrist_hash(p0, p1, w0, w1, h_walls, v_walls, turn)
        entry = self.tt.get(h)
        if entry and entry[0] >= depth:
            s = entry[1]
            if s > WIN_SCORE - 300: s -= ply
            elif s < -WIN_SCORE + 300: s += ply
            if entry[2] == TT_EXACT: return s
            if entry[2] == TT_LOWER and s >= beta: return s
            if entry[2] == TT_UPPER and s <= alpha: return s
            
        if depth <= 0: return self.evaluate(p0, p1, w0, w1, h_mask, v_mask, turn)
            
        my_pos, opp_pos, my_goal, opp_goal = (p0, p1, 0, 8) if turn == 0 else (p1, p0, 8, 0)
        my_wrem, opp_wrem = (w0, w1) if turn == 0 else (w1, w0)
        my_path, opp_path = get_path(my_pos, my_goal, h_mask, v_mask), get_path(opp_pos, opp_goal, h_mask, v_mask)
        if not my_path: return -WIN_SCORE + ply
        if not opp_path: return WIN_SCORE - ply
        
        # Move Generation
        moves = []
        r, c = divmod(my_pos, 9); orow, ocol = divmod(opp_pos, 9)
        # Cardinals
        if r > 0 and not (h_mask & (1 << (my_pos - 9))):
            n = my_pos - 9
            if n == opp_pos:
                if orow > 0 and not (h_mask & (1 << (opp_pos - 9))): moves.append((0, 800000))
                else:
                    if ocol > 0 and not (v_mask & (1 << (opp_pos - 1))): moves.append((4, 800000))
                    if ocol < 8 and not (v_mask & (1 << opp_pos)): moves.append((5, 800000))
            else: moves.append((0, 800000 if n == my_path[1] else 100000))
        if r < 8 and not (h_mask & (1 << my_pos)):
            n = my_pos + 9
            if n == opp_pos:
                if orow < 8 and not (h_mask & (1 << opp_pos)): moves.append((1, 800000))
                else:
                    if ocol > 0 and not (v_mask & (1 << (opp_pos - 1))): moves.append((6, 800000))
                    if ocol < 8 and not (v_mask & (1 << opp_pos)): moves.append((7, 800000))
            else: moves.append((1, 800000 if (len(my_path) > 1 and n == my_path[1]) else 100000))
        if c > 0 and not (v_mask & (1 << (my_pos - 1))):
            n = my_pos - 1
            if n == opp_pos:
                if ocol > 0 and not (v_mask & (1 << (opp_pos - 1))): moves.append((2, 800000))
                else:
                    if orow > 0 and not (h_mask & (1 << (opp_pos - 9))): moves.append((4, 800000))
                    if orow < 8 and not (h_mask & (1 << opp_pos)): moves.append((6, 800000))
            else: moves.append((2, 800000 if (len(my_path) > 1 and n == my_path[1]) else 100000))
        if c < 8 and not (v_mask & (1 << my_pos)):
            n = my_pos + 1
            if n == opp_pos:
                if ocol < 8 and not (v_mask & (1 << opp_pos)): moves.append((3, 800000))
                else:
                    if orow > 0 and not (h_mask & (1 << (opp_pos - 9))): moves.append((5, 800000))
                    if orow < 8 and not (h_mask & (1 << opp_pos)): moves.append((7, 800000))
            else: moves.append((3, 800000 if (len(my_path) > 1 and n == my_path[1]) else 100000))
            
        if my_wrem > 0:
            cands = set()
            for i in range(min(len(opp_path) - 1, 10)):
                u, v = opp_path[i], opp_path[i+1]; ur, uc = divmod(u, 9); vr, vc = divmod(v, 9)
                if ur == vr:
                    lc = min(uc, vc)
                    if lc < 8:
                        if ur > 0: cands.add((1, ur - 1, lc))
                        if ur < 8: cands.add((1, ur, lc))
                else:
                    tr = min(ur, vr)
                    if tr < 8:
                        if uc > 0: cands.add((0, tr, uc - 1))
                        if uc < 8: cands.add((0, tr, uc))
            for i in range(min(len(my_path) - 1, 2)):
                u, v = my_path[i], my_path[i+1]; ur, uc = divmod(u, 9); vr, vc = divmod(v, 9)
                if ur == vr:
                    lc = min(uc, vc)
                    if lc < 8:
                        if ur > 0: cands.add((1, ur - 1, lc))
                        if ur < 8: cands.add((1, ur, lc))
                else:
                    tr = min(ur, vr)
                    if tr < 8:
                        if uc > 0: cands.add((0, tr, uc - 1))
                        if uc < 8: cands.add((0, tr, uc))
            od_now = len(opp_path) - 1
            for d, r, c in cands:
                idx8 = r * 8 + c
                if d == 0:
                    if h_walls & (1 << idx8) or (c > 0 and h_walls & (1 << (idx8 - 1))) or (c < 7 and h_walls & (1 << (idx8 + 1))) or v_walls & (1 << idx8): continue
                else:
                    if v_walls & (1 << idx8) or (r > 0 and v_walls & (1 << (idx8 - 8))) or (r < 7 and v_walls & (1 << (idx8 + 8))) or h_walls & (1 << idx8): continue
                
                idx9 = r * 9 + c
                if d == 0: nh, nv = h_mask | (1 << idx9) | (1 << (idx9 + 1)), v_mask
                else: nh, nv = h_mask, v_mask | (1 << idx9) | (1 << (idx9 + 9))
                nd = get_dist(opp_pos, opp_goal, nh, nv)
                if nd >= 255: continue
                moves.append((100 + d*64 + r*8 + c, 500000 + (nd - od_now) * 10000))
        
        tt_best, (k1, k2) = (entry[3] if entry else -1), self.killers[ply]
        scored_moves = []
        for a, s in moves:
            score = 2000000 if a == tt_best else (1900000 if a == k1 else (1800000 if a == k2 else s + self.history.get(a, 0)))
            scored_moves.append((score, a))
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_action, original_alpha = -INF * 4, scored_moves[0][1], alpha
        for i, (s, a) in enumerate(scored_moves):
            # Apply
            n_my, n_wrem, n_hw, n_vw, n_hm, n_vm = my_pos, my_wrem, h_walls, v_walls, h_mask, v_mask
            if a < 8:
                if a == 0: nxt = n_my - 9; n_my = nxt - 9 if nxt == opp_pos else nxt
                elif a == 1: nxt = n_my + 9; n_my = nxt + 9 if nxt == opp_pos else nxt
                elif a == 2: nxt = n_my - 1; n_my = nxt - 1 if nxt == opp_pos else nxt
                elif a == 3: nxt = n_my + 1; n_my = nxt + 1 if nxt == opp_pos else nxt
                elif a == 4: n_my = opp_pos - 1 if n_my - 9 == opp_pos else opp_pos - 9
                elif a == 5: n_my = opp_pos + 1 if n_my - 9 == opp_pos else opp_pos - 9
                elif a == 6: n_my = opp_pos - 1 if n_my + 9 == opp_pos else opp_pos + 9
                elif a == 7: n_my = opp_pos + 1 if n_my + 9 == opp_pos else opp_pos + 9
            else:
                v = a - 100; d, rem = divmod(v, 64); r, c = divmod(rem, 8); idx8, idx9, n_wrem = r*8 + c, r*9 + c, n_wrem - 1
                if d == 0: n_hw |= (1 << idx8); n_hm |= (1 << idx9) | (1 << (idx9 + 1))
                else: n_vw |= (1 << idx8); n_vm |= (1 << idx9) | (1 << (idx9 + 9))
                if get_dist(n_my, my_goal, n_hm, n_vm) >= 255: continue
            
            n_p0, n_p1, n_w0, n_w1 = (n_my, opp_pos, n_wrem, opp_wrem) if turn == 0 else (opp_pos, n_my, opp_wrem, n_wrem)
            
            # LMR
            if i > 5 and depth > 2 and a >= 100:
                score = -self.search(n_p0, n_p1, n_w0, n_w1, n_hw, n_vw, n_hm, n_vm, 1 - turn, depth - 2, ply + 1, -alpha - 1, -alpha)
                if score > alpha: score = -self.search(n_p0, n_p1, n_w0, n_w1, n_hw, n_vw, n_hm, n_vm, 1 - turn, depth - 1, ply + 1, -beta, -alpha)
            elif i == 0: score = -self.search(n_p0, n_p1, n_w0, n_w1, n_hw, n_vw, n_hm, n_vm, 1 - turn, depth - 1, ply + 1, -beta, -alpha)
            else:
                score = -self.search(n_p0, n_p1, n_w0, n_w1, n_hw, n_vw, n_hm, n_vm, 1 - turn, depth - 1, ply + 1, -alpha - 1, -alpha)
                if alpha < score < beta: score = -self.search(n_p0, n_p1, n_w0, n_w1, n_hw, n_vw, n_hm, n_vm, 1 - turn, depth - 1, ply + 1, -beta, -alpha)
                
            if score > best_score: best_score, best_action = score, a
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if a >= 100: self.history[a] = self.history.get(a, 0) + depth * depth
                    self.killers[ply][1], self.killers[ply][0] = self.killers[ply][0], a
                    break
        
        flag = TT_EXACT if (best_score > original_alpha and best_score < beta) else (TT_UPPER if best_score <= original_alpha else TT_LOWER)
        st = best_score
        if st > WIN_SCORE - 300: st += ply
        elif st < -WIN_SCORE + 300: st -= ply
        if len(self.tt) > self.max_tt_size: self.tt.clear()
        self.tt[h] = (depth, st, flag, best_action)
        return best_score

    def choose(self, state):
        start_time = time.perf_counter(); me = state.get("player_id", state.get("actor", 0))
        if state.get("turn", 0) < 8 and not state.get("walls"):
            f = "MOVE_UP" if me == 0 else "MOVE_DOWN"
            if f in state["legal_actions"]: return f
        self.stop_time = start_time + float(state.get("decision_timeout", 1.5)) - 0.1
        self.nodes, self.killers = 0, [[0, 0] for _ in range(MAX_PLY)]
        for a in self.history: self.history[a] //= 2
        p0r, p0c = state["positions"][0]; p1r, p1c = state["positions"][1]
        p0, p1 = p0r * 9 + p0c, p1r * 9 + p1c
        w0, w1 = state["walls_remaining"]; hw, vw, hm, vm = 0, 0, 0, 0
        for w in state.get("walls", []):
            r, c, idx8, idx9 = w["row"], w["col"], w["row"]*8 + w["col"], w["row"]*9 + w["col"]
            if w["dir"] == "H": hw |= (1 << idx8); hm |= (1 << idx9) | (1 << (idx9 + 1))
            else: vw |= (1 << idx8); vm |= (1 << idx9) | (1 << (idx9 + 9))
            
        best_a, last_score = -1, 0
        try:
            for depth in range(1, 40):
                if depth > 3:
                    alpha, beta = last_score - 40, last_score + 40
                    while True:
                        score = self.search(p0, p1, w0, w1, hw, vw, hm, vm, me, depth, 0, alpha, beta)
                        if score <= alpha: alpha -= 200
                        elif score >= beta: beta += 200
                        else: break
                else: score = self.search(p0, p1, w0, w1, hw, vw, hm, vm, me, depth, 0, -WIN_SCORE*4, WIN_SCORE*4)
                last_score = score
                h = get_zobrist_hash(p0, p1, w0, w1, hw, vw, me); best_a = self.tt[h][3]
                if score > WIN_SCORE - 300: break
        except TimeoutError: pass
        if best_a == -1: return state["legal_actions"][0]
        if best_a < 8: res = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP_LEFT", "MOVE_UP_RIGHT", "MOVE_DOWN_LEFT", "MOVE_DOWN_RIGHT"][best_a]
        else:
            v = best_a - 100; d, rem = divmod(v, 64); r, c = divmod(rem, 8)
            res = f"WALL_{'H' if d == 0 else 'V'}_{r}_{c}"
        return res if res in state["legal_actions"] else state["legal_actions"][0]

_engine = Engine()
def choose_action(state): return _engine.choose(state)
