import time

# Quoridor Bot v2.1
# Corrected diagonal jump rules and improved search stability.

SAFETY_SECONDS = 0.90
INF = 10**9

# Bitboard constants
ROW_0_MASK = (1 << 9) - 1
ROW_8_MASK = ((1 << 9) - 1) << 72
LEFT_COL_MASK = sum(1 << i for i in range(0, 81, 9))
NOT_LEFT_COL_MASK = ((1 << 81) - 1) ^ LEFT_COL_MASK
RIGHT_COL_MASK = sum(1 << i for i in range(8, 81, 9))
NOT_RIGHT_COL_MASK = ((1 << 81) - 1) ^ RIGHT_COL_MASK
BOARD_MASK = (1 << 81) - 1

def get_shortest_path_dist(start_pos, goal_row, H_edges, V_edges):
    target_mask = ROW_0_MASK if goal_row == 0 else ROW_8_MASK
    front = 1 << start_pos
    visited = front
    dist = 0
    while front:
        if front & target_mask:
            return dist
        
        u = (front & ~(H_edges << 9)) >> 9
        d = (front & ~H_edges) << 9
        l = (front & NOT_LEFT_COL_MASK & ~(V_edges << 1)) >> 1
        r = (front & NOT_RIGHT_COL_MASK & ~V_edges) << 1
        
        front = (u | d | l | r) & BOARD_MASK & ~visited
        visited |= front
        dist += 1
    return 1000

def get_shortest_path(start_pos, goal_row, H_edges, V_edges):
    queue = [start_pos]
    head = 0
    parents_arr = [-1] * 81
    visited = 1 << start_pos
    
    while head < len(queue):
        curr = queue[head]
        head += 1
        
        if (goal_row == 0 and curr < 9) or (goal_row == 8 and curr >= 72):
            path = []
            while curr != -1:
                path.append(curr)
                curr = parents_arr[curr]
            return path[::-1]
            
        # UP
        if curr >= 9 and not (H_edges & (1 << (curr - 9))):
            npos = curr - 9
            if not (visited & (1 << npos)):
                visited |= (1 << npos)
                parents_arr[npos] = curr
                queue.append(npos)
        # DOWN
        if curr < 72 and not (H_edges & (1 << curr)):
            npos = curr + 9
            if not (visited & (1 << npos)):
                visited |= (1 << npos)
                parents_arr[npos] = curr
                queue.append(npos)
        # LEFT
        if (curr % 9) > 0 and not (V_edges & (1 << (curr - 1))):
            npos = curr - 1
            if not (visited & (1 << npos)):
                visited |= (1 << npos)
                parents_arr[npos] = curr
                queue.append(npos)
        # RIGHT
        if (curr % 9) < 8 and not (V_edges & (1 << curr)):
            npos = curr + 1
            if not (visited & (1 << npos)):
                visited |= (1 << npos)
                parents_arr[npos] = curr
                queue.append(npos)
    return []

def get_legal_moves(pos, opp_pos, H_edges, V_edges):
    moves = []
    # Basic directions: UP, DOWN, LEFT, RIGHT
    # UP
    if pos >= 9 and not (H_edges & (1 << (pos - 9))):
        npos = pos - 9
        if npos == opp_pos:
            # Jump UP
            if npos >= 9 and not (H_edges & (1 << (npos - 9))):
                moves.append(npos - 9)
            # Diagonal Jump UP-LEFT
            if (npos % 9) > 0 and not (V_edges & (1 << (npos - 1))):
                moves.append(npos - 1)
            # Diagonal Jump UP-RIGHT
            if (npos % 9) < 8 and not (V_edges & (1 << npos)):
                moves.append(npos + 1)
        else:
            moves.append(npos)
    # DOWN
    if pos < 72 and not (H_edges & (1 << pos)):
        npos = pos + 9
        if npos == opp_pos:
            # Jump DOWN
            if npos < 72 and not (H_edges & (1 << npos)):
                moves.append(npos + 9)
            # Diagonal Jump DOWN-LEFT
            if (npos % 9) > 0 and not (V_edges & (1 << (npos - 1))):
                moves.append(npos - 1)
            # Diagonal Jump DOWN-RIGHT
            if (npos % 9) < 8 and not (V_edges & (1 << npos)):
                moves.append(npos + 1)
        else:
            moves.append(npos)
    # LEFT
    if (pos % 9) > 0 and not (V_edges & (1 << (pos - 1))):
        npos = pos - 1
        if npos == opp_pos:
            # Jump LEFT
            if (npos % 9) > 0 and not (V_edges & (1 << (npos - 1))):
                moves.append(npos - 1)
            # Diagonal Jump LEFT-UP
            if npos >= 9 and not (H_edges & (1 << (npos - 9))):
                moves.append(npos - 9)
            # Diagonal Jump LEFT-DOWN
            if npos < 72 and not (H_edges & (1 << npos)):
                moves.append(npos + 9)
        else:
            moves.append(npos)
    # RIGHT
    if (pos % 9) < 8 and not (V_edges & (1 << pos)):
        npos = pos + 1
        if npos == opp_pos:
            # Jump RIGHT
            if (npos % 9) < 8 and not (V_edges & (1 << npos)):
                moves.append(npos + 1)
            # Diagonal Jump RIGHT-UP
            if npos >= 9 and not (H_edges & (1 << (npos - 9))):
                moves.append(npos - 9)
            # Diagonal Jump RIGHT-DOWN
            if npos < 72 and not (H_edges & (1 << npos)):
                moves.append(npos + 9)
        else:
            moves.append(npos)
    return list(dict.fromkeys(moves))

def walls_blocking_step(a, b):
    walls = []
    if b == a - 9: # UP
        r = (a - 9) // 9
        c = (a - 9) % 9
        if c < 8: walls.append((1, r * 9 + c))
        if c > 0: walls.append((1, r * 9 + c - 1))
    elif b == a + 9: # DOWN
        r = a // 9
        c = a % 9
        if c < 8: walls.append((1, r * 9 + c))
        if c > 0: walls.append((1, r * 9 + c - 1))
    elif b == a - 1: # LEFT
        r = (a - 1) // 9
        c = (a - 1) % 9
        if r < 8: walls.append((2, r * 9 + c))
        if r > 0: walls.append((2, (r - 1) * 9 + c))
    elif b == a + 1: # RIGHT
        r = a // 9
        c = a % 9
        if r < 8: walls.append((2, r * 9 + c))
        if r > 0: walls.append((2, (r - 1) * 9 + c))
    return walls

class SearchTimeout(Exception):
    pass

class Bot:
    name = "gemini_ultimate_v2_1"
    
    def __init__(self):
        self.tt = {}
        self.deadline = 0
        self.me = 0
        self.current_depth = 0
        self.nodes = 0
        self.history = []
        
    def state_to_bitboards(self, state):
        p0_row, p0_col = state["positions"][0]
        p1_row, p1_col = state["positions"][1]
        p0 = p0_row * 9 + p0_col
        p1 = p1_row * 9 + p1_col
        w0 = state["walls_remaining"][0]
        w1 = state["walls_remaining"][1]
        turn = state.get("actor", state.get("player_id", 0))
        
        H_edges = 0
        V_edges = 0
        H_walls = 0
        V_walls = 0
        
        for wall in state.get("walls", []):
            r, c = wall["row"], wall["col"]
            pos = r * 9 + c
            if wall["dir"] == "H":
                H_walls |= (1 << pos)
                H_edges |= ((1 << pos) | (1 << (pos + 1)))
            else:
                V_walls |= (1 << pos)
                V_edges |= ((1 << pos) | (1 << (pos + 9)))
                
        return p0, p1, w0, w1, turn, H_edges, V_edges, H_walls, V_walls

    def generate_focused_walls(self, me, p0, p1, H_edges, V_edges, H_walls, V_walls):
        opp = 1 - me
        opp_pos = p1 if opp == 1 else p0
        opp_goal = 8 if opp == 1 else 0
        opp_path = get_shortest_path(opp_pos, opp_goal, H_edges, V_edges)
        
        candidates = set()
        for i in range(min(len(opp_path) - 1, 10)):
            a = opp_path[i]
            b = opp_path[i+1]
            for is_H, pos in walls_blocking_step(a, b):
                if is_H == 1:
                    if (V_walls & (1 << pos)) or (H_walls & ((1 << pos) | (1 << (pos + 1)) | (1 << (pos - 1)) if pos % 9 > 0 else 0)):
                        continue
                    if (H_edges & ((1 << pos) | (1 << (pos + 1)))):
                        continue
                else:
                    if (H_walls & (1 << pos)) or (V_walls & ((1 << pos) | (1 << (pos + 9)) | (1 << (pos - 9)) if pos >= 9 else 0)):
                        continue
                    if (V_edges & ((1 << pos) | (1 << (pos + 9)))):
                        continue
                candidates.add((is_H, pos))
                
        my_pos = p0 if me == 0 else p1
        my_goal = 0 if me == 0 else 8
        my_path = get_shortest_path(my_pos, my_goal, H_edges, V_edges)
        for i in range(min(len(my_path) - 1, 3)):
            a = my_path[i]
            b = my_path[i+1]
            for is_H, pos in walls_blocking_step(a, b):
                if is_H == 1:
                    if (V_walls & (1 << pos)) or (H_edges & ((1 << pos) | (1 << (pos + 1)))):
                        continue
                else:
                    if (H_walls & (1 << pos)) or (V_edges & ((1 << pos) | (1 << (pos + 9)))):
                        continue
                candidates.add((is_H, pos))

        valid_H = []
        valid_V = []
        for is_H, pos in candidates:
            new_He = H_edges
            new_Ve = V_edges
            if is_H == 1:
                new_He |= ((1 << pos) | (1 << (pos + 1)))
            else:
                new_Ve |= ((1 << pos) | (1 << (pos + 9)))
                
            if get_shortest_path_dist(p0, 0, new_He, new_Ve) < 1000 and \
               get_shortest_path_dist(p1, 8, new_He, new_Ve) < 1000:
                if is_H == 1:
                    valid_H.append(pos)
                else:
                    valid_V.append(pos)
                
        return valid_H, valid_V

    def evaluate(self, p0, p1, w0, w1, H_edges, V_edges, turn):
        d0 = get_shortest_path_dist(p0, 0, H_edges, V_edges)
        d1 = get_shortest_path_dist(p1, 8, H_edges, V_edges)
        
        if d0 == 0: return 1000000 if self.me == 0 else -1000000
        if d1 == 0: return 1000000 if self.me == 1 else -1000000
        
        race0 = 2 * d0 - (1 if turn == 0 else 0)
        race1 = 2 * d1 - (1 if turn == 1 else 0)
        
        score = 24.0 * (race1 - race0) if self.me == 0 else 24.0 * (race0 - race1)
        
        wall_diff = (w0 - w1) if self.me == 0 else (w1 - w0)
        score += 1.5 * wall_diff
        
        if (self.me == 0 and race0 < race1) or (self.me == 1 and race1 < race0):
            score += 8.0
        elif (self.me == 0 and race0 > race1) or (self.me == 1 and race1 > race0):
            score -= 8.0
            
        c0 = abs((p0 % 9) - 4)
        c1 = abs((p1 % 9) - 4)
        if self.me == 0:
            score += 0.2 * (4 - c0) - 0.1 * (4 - c1)
        else:
            score += 0.2 * (4 - c1) - 0.1 * (4 - c0)
            
        return score

    def search(self, depth, alpha, beta, turn, p0, p1, w0, w1, H_edges, V_edges, H_walls, V_walls):
        self.nodes += 1
        if (self.nodes & 255) == 0:
            if time.perf_counter() > self.deadline:
                raise SearchTimeout
            
        if p0 < 9:
            return 1000000 - self.current_depth + depth if self.me == 0 else -1000000 + self.current_depth - depth, None
        if p1 >= 72:
            return 1000000 - self.current_depth + depth if self.me == 1 else -1000000 + self.current_depth - depth, None
            
        if depth <= 0:
            return self.evaluate(p0, p1, w0, w1, H_edges, V_edges, turn), None
            
        state_hash = (turn) | (p0 << 1) | (p1 << 8) | (w0 << 15) | (w1 << 19) | (H_walls << 23) | (V_walls << 104)
        
        tt_entry = self.tt.get(state_hash)
        best_move_tt = None
        if tt_entry:
            tt_depth, tt_score, tt_type, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_type == 0:
                    return tt_score, tt_move
                elif tt_type == 1:
                    alpha = max(alpha, tt_score)
                elif tt_type == 2:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score, tt_move
            best_move_tt = tt_move
            
        moves = []
        curr_pos = p0 if turn == 0 else p1
        opp_pos = p1 if turn == 0 else p0
        
        for npos in get_legal_moves(curr_pos, opp_pos, H_edges, V_edges):
            moves.append((0, npos))
            
        w_curr = w0 if turn == 0 else w1
        if w_curr > 0:
            valid_H, valid_V = self.generate_focused_walls(turn, p0, p1, H_edges, V_edges, H_walls, V_walls)
            for pos in valid_H:
                moves.append((1, pos))
            for pos in valid_V:
                moves.append((2, pos))
                
        my_goal = 0 if turn == 0 else 8
        path = get_shortest_path(curr_pos, my_goal, H_edges, V_edges)
        path_next = path[1] if len(path) > 1 else -1
        
        opp_goal = 8 if turn == 0 else 0
        opp_dist_before = get_shortest_path_dist(opp_pos, opp_goal, H_edges, V_edges)
        
        def move_score(m):
            if m == best_move_tt:
                return 10000
            if m[0] == 0:
                if m[1] == path_next:
                    return 2000
                # Penalty for repeated positions
                if m[1] in self.history[-4:]:
                    return -100
                return 500
            elif m[0] == 1: # Horizontal Wall
                new_He = H_edges | ((1 << m[1]) | (1 << (m[1] + 1)))
                opp_dist_after = get_shortest_path_dist(opp_pos, opp_goal, new_He, V_edges)
                return 100 * (opp_dist_after - opp_dist_before)
            elif m[0] == 2: # Vertical Wall
                new_Ve = V_edges | ((1 << m[1]) | (1 << (m[1] + 9)))
                opp_dist_after = get_shortest_path_dist(opp_pos, opp_goal, H_edges, new_Ve)
                return 100 * (opp_dist_after - opp_dist_before)
            return 0
            
        moves.sort(key=move_score, reverse=True)
        
        best_score = -INF
        best_move = None
        alpha_orig = alpha
        is_me = (turn == self.me)
        
        if is_me:
            best_score = -INF
            for m in moves:
                np0, np1, nw0, nw1 = p0, p1, w0, w1
                nH_edges, nV_edges, nH_walls, nV_walls = H_edges, V_edges, H_walls, V_walls
                
                if m[0] == 0:
                    if turn == 0: np0 = m[1]
                    else: np1 = m[1]
                elif m[0] == 1:
                    if turn == 0: nw0 -= 1
                    else: nw1 -= 1
                    nH_walls |= (1 << m[1])
                    nH_edges |= ((1 << m[1]) | (1 << (m[1] + 1)))
                elif m[0] == 2:
                    if turn == 0: nw0 -= 1
                    else: nw1 -= 1
                    nV_walls |= (1 << m[1])
                    nV_edges |= ((1 << m[1]) | (1 << (m[1] + 9)))
                    
                score, _ = self.search(depth - 1, alpha, beta, 1 - turn, np0, np1, nw0, nw1, nH_edges, nV_edges, nH_walls, nV_walls)
                
                if score > best_score:
                    best_score = score
                    best_move = m
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
        else:
            best_score = INF
            for m in moves:
                np0, np1, nw0, nw1 = p0, p1, w0, w1
                nH_edges, nV_edges, nH_walls, nV_walls = H_edges, V_edges, H_walls, V_walls
                
                if m[0] == 0:
                    if turn == 0: np0 = m[1]
                    else: np1 = m[1]
                elif m[0] == 1:
                    if turn == 0: nw0 -= 1
                    else: nw1 -= 1
                    nH_walls |= (1 << m[1])
                    nH_edges |= ((1 << m[1]) | (1 << (m[1] + 1)))
                elif m[0] == 2:
                    if turn == 0: nw0 -= 1
                    else: nw1 -= 1
                    nV_walls |= (1 << m[1])
                    nV_edges |= ((1 << m[1]) | (1 << (m[1] + 9)))
                    
                score, _ = self.search(depth - 1, alpha, beta, 1 - turn, np0, np1, nw0, nw1, nH_edges, nV_edges, nH_walls, nV_walls)
                
                if score < best_score:
                    best_score = score
                    best_move = m
                beta = min(beta, best_score)
                if alpha >= beta:
                    break
                    
        tt_type = 0
        if best_score <= alpha_orig:
            tt_type = 2
        elif best_score >= beta:
            tt_type = 1
            
        self.tt[state_hash] = (depth, best_score, tt_type, best_move)
        return best_score, best_move

    def move_to_str(self, m, p0, p1, turn):
        if m[0] == 0:
            pos = p0 if turn == 0 else p1
            npos = m[1]
            r, c = pos // 9, pos % 9
            nr, nc = npos // 9, npos % 9
            
            dr = nr - r
            dc = nc - c
            
            if dr == -1 and dc == 0: return "MOVE_UP"
            if dr == 1 and dc == 0: return "MOVE_DOWN"
            if dr == 0 and dc == -1: return "MOVE_LEFT"
            if dr == 0 and dc == 1: return "MOVE_RIGHT"
            
            if dr == -2 and dc == 0: return "MOVE_UP"
            if dr == 2 and dc == 0: return "MOVE_DOWN"
            if dr == 0 and dc == -2: return "MOVE_LEFT"
            if dr == 0 and dc == 2: return "MOVE_RIGHT"
            
            if dr == -1 and dc == -1: return "MOVE_UP_LEFT"
            if dr == -1 and dc == 1: return "MOVE_UP_RIGHT"
            if dr == 1 and dc == -1: return "MOVE_DOWN_LEFT"
            if dr == 1 and dc == 1: return "MOVE_DOWN_RIGHT"
            
            return "MOVE_UP"
        elif m[0] == 1:
            r = m[1] // 9
            c = m[1] % 9
            return f"WALL_H_{r}_{c}"
        elif m[0] == 2:
            r = m[1] // 9
            c = m[1] % 9
            return f"WALL_V_{r}_{c}"

    def choose_action(self, state):
        legal = state.get("legal_actions", [])
        if not legal:
            return ""
            
        self.me = state.get("player_id", state.get("actor", 0))
        p0, p1, w0, w1, turn, H_edges, V_edges, H_walls, V_walls = self.state_to_bitboards(state)
        
        # Track history
        my_pos = p0 if self.me == 0 else p1
        self.history.append(my_pos)
        self.history = self.history[-10:]
        
        my_goal = 0 if self.me == 0 else 8
        if get_shortest_path_dist(my_pos, my_goal, H_edges, V_edges) == 1:
            for m in get_legal_moves(my_pos, p1 if self.me == 0 else p0, H_edges, V_edges):
                if (self.me == 0 and m < 9) or (self.me == 1 and m >= 72):
                    act = self.move_to_str((0, m), p0, p1, turn)
                    if act in legal:
                        return act
                    
        self.deadline = time.perf_counter() + SAFETY_SECONDS
        self.nodes = 0
        best_action_overall = None
        
        if len(self.tt) > 1000000:
            self.tt.clear()
            
        for depth in range(1, 40):
            self.current_depth = depth
            try:
                score, m = self.search(depth, -INF, INF, turn, p0, p1, w0, w1, H_edges, V_edges, H_walls, V_walls)
                if m is not None:
                    best_action_overall = m
                if abs(score) > 900000:
                    break
            except SearchTimeout:
                break
                
        if best_action_overall is not None:
            action_str = self.move_to_str(best_action_overall, p0, p1, turn)
            if action_str in legal:
                return action_str
                
        # Pathfinding fallback
        best_legal_move = None
        min_dist = 1000
        for m in get_legal_moves(my_pos, p1 if self.me == 0 else p0, H_edges, V_edges):
            d = get_shortest_path_dist(m, my_goal, H_edges, V_edges)
            if d < min_dist:
                min_dist = d
                best_legal_move = m
        
        if best_legal_move is not None:
            action_str = self.move_to_str((0, best_legal_move), p0, p1, turn)
            if action_str in legal:
                return action_str
                
        return legal[0]
