"""Gemini V4.16 - The Bottleneck Hunter.

Key features:
- Core engine from V3.
- Bottleneck Evaluation (penalizes low path redundancy).
- Persistent TT, Zobrist Hashing, Aspiration Windows, PVS.
- Dual-distance wall scoring at root.
"""

import time
import random

INF = 1000000
WIN_SCORE = 2000000
MAX_PLY = 120

BOARD_MASK = 0x1FFFFFFFFFFFFFFFFFFFF
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
ZOBRIST_WALL_H = [random.getrandbits(64) for _ in range(81)]
ZOBRIST_WALL_V = [random.getrandbits(64) for _ in range(81)]
ZOBRIST_WALL_REM = [[random.getrandbits(64) for _ in range(11)] for _ in range(2)]
ZOBRIST_TURN = random.getrandbits(64)

def get_state_hash(p0, p1, w0, w1, hw, vw, turn):
    h = ZOBRIST_POS[0][p0] ^ ZOBRIST_POS[1][p1] ^ ZOBRIST_WALL_REM[0][w0] ^ ZOBRIST_WALL_REM[1][w1]
    if turn == 1: h ^= ZOBRIST_TURN
    th, tv = hw, vw
    while th:
        lb = th & -th; idx = lb.bit_length() - 1
        if idx < 81: h ^= ZOBRIST_WALL_H[idx]
        th ^= lb
    while tv:
        lb = tv & -tv; idx = lb.bit_length() - 1
        if idx < 81: h ^= ZOBRIST_WALL_V[idx]
        tv ^= lb
    return h

# ---------- Fast BFS ----------
def get_dist_and_bottleneck(pos, target_mask, h_mask, v_mask):
    front = 1 << pos
    if front & target_mask: return 0, 10
    visited, dist = front, 0
    while front:
        dist += 1
        up = (front >> 9) & ~h_mask
        down = ((front & ~h_mask) << 9) & BOARD_MASK
        left = ((front & NOT_COL0_MASK) >> 1) & ~v_mask
        right = (((front & NOT_COL8_MASK) & ~v_mask) << 1) & BOARD_MASK
        front = (up | down | left | right) & ~visited
        if front & target_mask:
            red = bin(front & target_mask).count('1')
            return dist, red
        visited |= front
    return 255, 0

def get_path_fast(pos, target_mask, h_mask, v_mask):
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

def get_path_walls(path, limit):
    walls = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]; ur, uc = divmod(u, 9); vr, vc = divmod(v, 9)
        if ur == vr:
            lc = min(uc, vc);
            if lc < 8:
                if ur > 0: walls.append((1, ur - 1, lc))
                if ur < 8: walls.append((1, ur, lc))
        else:
            tr = min(ur, vr);
            if tr < 8:
                if uc > 0: walls.append((0, tr, uc - 1))
                if uc < 8: walls.append((0, tr, uc))
        if len(walls) >= limit: break
    return walls

# ---------- Engine ----------
class Engine:
    def __init__(self):
        self.tt, self.history, self.killers = {}, {}, [[0, 0] for _ in range(MAX_PLY)]
        self.stop_time, self.max_tt_size, self.nodes = 0, 1000000, 0

    def evaluate(self, p0, p1, w0, w1, hm, vm, turn):
        d0, b0 = get_dist_and_bottleneck(p0, ROW0_MASK, hm, vm)
        d1, b1 = get_dist_and_bottleneck(p1, ROW8_MASK, hm, vm)
        if d0 >= 255: return -WIN_SCORE + 100
        if d1 >= 255: return WIN_SCORE - 100
        r0, r1 = d0 * 2 - (1 if turn == 0 else 0), d1 * 2 - (1 if turn == 1 else 0)
        # Weights: Race is priority, Bottleneck and Walls are secondary
        score = (r1 - r0) * 100 + (w0 - w1) * 20 + (b0 - b1) * 5
        return score if turn == 0 else -score

    def search(self, p0, p1, w0, w1, hw, vw, hm, vm, turn, depth, ply, alpha, beta):
        self.nodes += 1
        if (self.nodes & 1023) == 0 and time.perf_counter() > self.stop_time: raise TimeoutError()
        if p0 < 9: return WIN_SCORE - ply
        if p1 >= 72: return -WIN_SCORE + ply
        
        h = get_state_hash(p0, p1, w0, w1, hw, vw, turn)
        entry = self.tt.get(h)
        if entry and entry[0] >= depth:
            s = entry[1]
            if s > WIN_SCORE - 300: s -= ply
            elif s < -WIN_SCORE + 300: s += ply
            if entry[2] == TT_EXACT: return s
            if entry[2] == TT_LOWER and s >= beta: return s
            if entry[2] == TT_UPPER and s <= alpha: return s
            
        if depth <= 0: return self.evaluate(p0, p1, w0, w1, hm, vm, turn)
            
        my_p, opp_p, my_g, opp_g = (p0, p1, ROW0_MASK, ROW8_MASK) if turn == 0 else (p1, p0, ROW8_MASK, ROW0_MASK)
        my_w, opp_w = (w0, w1) if turn == 0 else (w1, w0)
        my_path, opp_path = get_path_fast(my_p, my_g, hm, vm), get_path_fast(opp_p, opp_g, hm, vm)
        if not my_path or not opp_path: return -WIN_SCORE + ply if not my_path else WIN_SCORE - ply
        
        # Moves
        moves = []
        r, c = divmod(my_p, 9); orow, ocol = divmod(opp_p, 9)
        raw_acts = []
        for dr, dc, act in [(-1,0,0),(1,0,1),(0,-1,2),(0,1,3)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr <= 8 and 0 <= nc <= 8:
                m9 = nr*9+nc; b = False
                if dr == -1: b = hm & (1 << (my_p-9))
                elif dr == 1: b = hm & (1 << my_p)
                elif dc == -1: b = vm & (1 << (my_p-1))
                else: b = vm & (1 << my_p)
                if not b:
                    if m9 != opp_p: raw_acts.append(act)
                    else:
                        jr, jc = nr+dr, nc+dc; jd = False
                        if 0 <= jr <= 8 and 0 <= jc <= 8:
                            jb = False
                            if dr == -1: jb = hm & (1 << (opp_p-9))
                            elif dr == 1: jb = hm & (1 << opp_p)
                            elif dc == -1: jb = vm & (1 << (opp_p-1))
                            else: jb = vm & (1 << opp_p)
                            if not jb: raw_acts.append(act); jd = True
                        if not jd:
                            for pr, pc, pact in [(0,-1,4),(0,1,5)] if dr != 0 else [(-1,0,4),(1,0,6)]:
                                sr, sc = nr+pr, nc+pc
                                if 0 <= sr <= 8 and 0 <= sc <= 8:
                                    sb = False
                                    if pr == -1: sb = hm & (1 << (opp_p-9))
                                    elif pr == 1: sb = hm & (1 << opp_p)
                                    elif pc == -1: sb = vm & (1 << (opp_p-1))
                                    else: sb = vm & (1 << opp_p)
                                    if not sb:
                                        fact = pact if dr != 0 else (4 if pr == -1 and dc == -1 else (5 if pr == -1 and dc == 1 else (6 if pr == 1 and dc == -1 else 7)))
                                        raw_acts.append(fact)
        k1, k2 = self.killers[ply]; npos = my_path[1] if len(my_path)>1 else -1
        for a in raw_acts:
            sc, dest = 1000000, -1
            if a==0: dest = my_p-9 if my_p-9!=opp_p else my_p-18
            elif a==1: dest = my_p+9 if my_p+9!=opp_p else my_p+18
            elif a==2: dest = my_p-1 if my_p-1!=opp_p else my_p-2
            elif a==3: dest = my_p+1 if my_p+1!=opp_p else my_p+2
            elif a==4: dest = opp_p-1 if my_p-9 == opp_p else opp_p-9
            elif a==5: dest = opp_p+1 if my_p-9 == opp_p else opp_p-9
            elif a==6: dest = opp_p-1 if my_p+9 == opp_p else opp_p+9
            elif a==7: dest = opp_p+1 if my_p+9 == opp_p else opp_p+9
            if dest == npos: sc = 8000000
            if a == k1: sc = 10000000
            elif a == k2: sc = 9000000
            moves.append((sc, a))
        if my_w > 0:
            owalls = get_path_walls(opp_path, 12); seen = set()
            for d, wr, wc in owalls:
                idx9 = wr*9+wc
                if d==0:
                    if hw&(1<<idx9) or hw&(1<<(idx9+1)) or (wc>0 and hw&(1<<(idx9-1))) or vw&(1<<idx9): continue
                else:
                    if vw&(1<<idx9) or vw&(1<<(idx9+9)) or (wr>0 and vw&(1<<(idx9-9))) or hw&(1<<idx9): continue
                enc = 100 + d*64 + wr*8 + wc; sc = self.history.get(enc, 0)
                if enc==k1: sc=10000000
                elif enc==k2: sc=9000000
                moves.append((sc, enc)); seen.add(enc)
            # Add a few defensive walls
            mwalls = get_path_walls(my_path, 2)
            for d, wr, wc in mwalls:
                idx9 = wr*9+wc
                if d==0:
                    if hw&(1<<idx9) or hw&(1<<(idx9+1)) or (wc>0 and hw&(1<<(idx9-1))) or vw&(1<<idx9): continue
                else:
                    if vw&(1<<idx9) or vw&(1<<(idx9+9)) or (wr>0 and vw&(1<<(idx9-9))) or hw&(1<<idx9): continue
                enc = 100 + d*64 + wr*8 + wc
                if enc not in seen:
                    sc = self.history.get(enc, 0)
                    if enc==k1: sc=10000000
                    elif enc==k2: sc=9000000
                    moves.append((sc, enc))
        
        tt_best = entry[3] if entry else -1
        scored = [(20000000 if a == tt_best else s, a) for s, a in moves]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        best_v, best_a, a_orig = -INF * 4, scored[0][1], alpha
        for i, (sc, a) in enumerate(scored):
            nm, nw, nhw, nvw, nhm, nvm = my_p, my_w, hw, vw, hm, vm
            if a < 8:
                if a == 0: nxt = nm-9; nm = nxt-9 if nxt==opp_p else nxt
                elif a == 1: nxt = nm+9; nm = nxt+9 if nxt==opp_p else nxt
                elif a == 2: nxt = nm-1; nm = nxt-1 if nxt==opp_p else nxt
                elif a == 3: nxt = nm+1; nm = nxt+1 if nxt==opp_p else nxt
                elif a == 4: nm = opp_p-1 if nm-9==opp_p else opp_p-9
                elif a == 5: nm = opp_p+1 if nm-9==opp_p else opp_p-9
                elif a == 6: nm = opp_p-1 if nm+9==opp_p else opp_p+9
                elif a == 7: nm = opp_p+1 if nm+9==opp_p else opp_p+9
            else:
                v = a-100; d, rem = divmod(v, 64); r, c = divmod(rem, 8); idx9, nw = r*9+c, nw-1
                if d == 0: nhw |= (1 << idx9); nhm |= (1 << idx9) | (1 << (idx9 + 1))
                else: nvw |= (1 << idx9); nvm |= (1 << idx9) | (1 << (idx9 + 9))
                # Skip legality check in loop for speed, evaluated at leaf
            
            n_p0, n_p1, n_w0, n_w1 = (nm, opp_p, nw, opp_w) if turn == 0 else (opp_p, nm, opp_w, nw)
            
            # PVS
            if i == 0: v_sc = -self.search(n_p0, n_p1, n_w0, n_w1, nhw, nvw, nhm, nvm, 1-turn, depth-1, ply+1, -beta, -alpha)
            else:
                v_sc = -self.search(n_p0, n_p1, n_w0, n_w1, nhw, nvw, nhm, nvm, 1-turn, depth-1, ply+1, -alpha-1, -alpha)
                if alpha < v_sc < beta: v_sc = -self.search(n_p0, n_p1, n_w0, n_w1, nhw, nvw, nhm, nvm, 1-turn, depth-1, ply+1, -beta, -alpha)
            
            if v_sc > best_v: best_v, best_a = v_sc, a
            if v_sc > alpha:
                alpha = v_sc
                if alpha >= beta:
                    if a >= 100: self.history[a] = self.history.get(a, 0) + depth*depth
                    self.killers[ply][1], self.killers[ply][0] = self.killers[ply][0], a
                    break
        
        flag = TT_EXACT if (best_v > a_orig and best_v < beta) else (TT_UPPER if best_v <= a_orig else TT_LOWER)
        st = best_v
        if st > WIN_SCORE-300: st += ply
        elif st < -WIN_SCORE+300: st -= ply
        if len(self.tt) > self.max_tt_size: self.tt.clear()
        self.tt[h] = (depth, st, flag, best_a)
        return best_v

    def choose(self, state):
        start = time.perf_counter(); me = state.get("player_id", state.get("actor", 0))
        self.stop_time = start + float(state.get("decision_timeout", 1.5)) - 0.1
        self.nodes, self.killers = 0, [[0, 0] for _ in range(MAX_PLY)]
        for a in self.history: self.history[a] //= 2
        p0r, p0c = state["positions"][0]; p1r, p1c = state["positions"][1]
        p0, p1 = p0r*9+p0c, p1r*9+p1c
        w0, w1 = state["walls_remaining"]; hw, vw, hm, vm = 0, 0, 0, 0
        for w in state.get("walls", []):
            idx9 = w["row"]*9 + w["col"]
            if w["dir"] == "H": hw |= (1 << idx9); hm |= (1 << idx9) | (1 << (idx9 + 1))
            else: vw |= (1 << idx9); vm |= (1 << idx9) | (1 << (idx9 + 9))
            
        best_a, last_score = -1, 0
        try:
            for depth in range(1, 40):
                if depth > 3:
                    alpha, beta = last_score-24, last_score+24
                    while True:
                        v_sc = self.search(p0, p1, w0, w1, hw, vw, hm, vm, me, depth, 0, alpha, beta)
                        if v_sc <= alpha: alpha -= 48
                        elif v_sc >= beta: beta += 48
                        else: break
                    last_score = v_sc
                else: last_score = self.search(p0, p1, w0, w1, hw, vw, hm, vm, me, depth, 0, -WIN_SCORE*4, WIN_SCORE*4)
                h = get_state_hash(p0, p1, w0, w1, hw, vw, me); best_a = self.tt[h][3]
                if last_score > WIN_SCORE-300: break
        except TimeoutError: pass
        if best_a == -1: return state["legal_actions"][0]
        if best_a < 8: res = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP_LEFT", "MOVE_UP_RIGHT", "MOVE_DOWN_LEFT", "MOVE_DOWN_RIGHT"][best_a]
        else:
            v = best_a-100; d, rem = divmod(v, 64); r, c = divmod(rem, 8); res = f"WALL_{'H' if d == 0 else 'V'}_{r}_{c}"
        return res if res in state["legal_actions"] else state["legal_actions"][0]

_engine = Engine()
def choose_action(state): return _engine.choose(state)
