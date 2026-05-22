"""Gemini Pro V3 Bot.

A from-scratch highly optimized Alpha-Beta searcher.
Features:
- Bitboard-based fast BFS pathfinding.
- Tactical move generation.
- Transposition Table.
- Novel "Anti-Fork" evaluation: evaluates the vulnerability of paths to single-wall blocks.
"""

import time
import math

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

def choose_action(state):
    PATH_CACHE.clear()
    VULN_CACHE.clear()
    return Bot().choose_action(state)

PATH_CACHE = {}
VULN_CACHE = {}

def bfs_path(pos, target_row, h_mask, v_mask):
    key = (pos, target_row, h_mask, v_mask)
    if key in PATH_CACHE:
        return PATH_CACHE[key]
        
    q = [0] * 81
    q[0] = pos
    visited = 1 << pos
    dist = 0
    parent = [-1] * 81
    
    head = 0
    tail = 1
    
    while head < tail:
        level_tail = tail
        for i in range(head, level_tail):
            curr = q[i]
            r = curr // 9
            if r == target_row:
                path = []
                p = curr
                while p != -1:
                    path.append(p)
                    p = parent[p]
                path.reverse()
                PATH_CACHE[key] = (dist, path)
                return dist, path
                
            # UP
            if r > 0:
                nxt = curr - 9
                if not (h_mask & (1 << nxt)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q[tail] = nxt
                    tail += 1
            # DOWN
            if r < 8:
                nxt = curr + 9
                if not (h_mask & (1 << curr)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q[tail] = nxt
                    tail += 1
            # LEFT
            c = curr % 9
            if c > 0:
                nxt = curr - 1
                if not (v_mask & (1 << nxt)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q[tail] = nxt
                    tail += 1
            # RIGHT
            if c < 8:
                nxt = curr + 1
                if not (v_mask & (1 << curr)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q[tail] = nxt
                    tail += 1
        head = level_tail
        dist += 1
        
    PATH_CACHE[key] = (1000, [])
    return 1000, []

def get_intersecting_walls(path):
    walls = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if abs(u - v) == 9:
            top = min(u, v)
            r, c = top // 9, top % 9
            if c < 8: walls.append(('H', r, c))
            if c > 0: walls.append(('H', r, c - 1))
        elif abs(u - v) == 1:
            left = min(u, v)
            r, c = left // 9, left % 9
            if r < 8: walls.append(('V', r, c))
            if r > 0: walls.append(('V', r - 1, c))
    return walls

def is_valid_wall(d, r, c, h_walls, v_walls):
    if r < 0 or r > 7 or c < 0 or c > 7: return False
    idx = r * 9 + c
    if d == 'H':
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

def calc_vulnerability(pos, target_row, path, h_mask, v_mask, h_walls, v_walls):
    key = (pos, target_row, h_mask, v_mask, h_walls, v_walls)
    if key in VULN_CACHE:
        return VULN_CACHE[key]
        
    walls = get_intersecting_walls(path)
    base_dist = len(path) - 1
    if base_dist < 0: base_dist = 0
    max_dist = base_dist
    
    for w in walls:
        d, r, c = w
        if is_valid_wall(d, r, c, h_walls, v_walls):
            if d == 'H':
                new_h = h_mask | (1 << (r * 9 + c)) | (1 << (r * 9 + c + 1))
                dist, _ = bfs_path(pos, target_row, new_h, v_mask)
            else:
                new_v = v_mask | (1 << (r * 9 + c)) | (1 << ((r + 1) * 9 + c))
                dist, _ = bfs_path(pos, target_row, h_mask, new_v)
                
            if dist < 1000 and dist > max_dist: 
                max_dist = dist
                
    VULN_CACHE[key] = max_dist
    return max_dist

class State:
    __slots__ = ['pos', 'goals', 'walls_rem', 'h_mask', 'v_mask', 'h_walls', 'v_walls', 'actor', 'turn']
    
    def clone(self):
        s = State()
        s.pos = list(self.pos)
        s.goals = self.goals
        s.walls_rem = list(self.walls_rem)
        s.h_mask = self.h_mask
        s.v_mask = self.v_mask
        s.h_walls = self.h_walls
        s.v_walls = self.v_walls
        s.actor = self.actor
        s.turn = self.turn
        return s

def get_legal_actions(state):
    actions = []
    me = state.actor
    opp = 1 - me
    my_pos = state.pos[me]
    opp_pos = state.pos[opp]
    r, c = my_pos // 9, my_pos % 9
    orow, ocol = opp_pos // 9, opp_pos % 9
    
    # 1. Moves
    if r > 0 and not (state.h_mask & (1 << (my_pos - 9))):
        nxt = my_pos - 9
        if nxt == opp_pos:
            if orow > 0 and not (state.h_mask & (1 << (opp_pos - 9))): actions.append("MOVE_UP")
            else:
                if ocol > 0 and not (state.v_mask & (1 << (opp_pos - 1))): actions.append("MOVE_UP_LEFT")
                if ocol < 8 and not (state.v_mask & (1 << opp_pos)): actions.append("MOVE_UP_RIGHT")
        else: actions.append("MOVE_UP")
        
    if r < 8 and not (state.h_mask & (1 << my_pos)):
        nxt = my_pos + 9
        if nxt == opp_pos:
            if orow < 8 and not (state.h_mask & (1 << opp_pos)): actions.append("MOVE_DOWN")
            else:
                if ocol > 0 and not (state.v_mask & (1 << (opp_pos - 1))): actions.append("MOVE_DOWN_LEFT")
                if ocol < 8 and not (state.v_mask & (1 << opp_pos)): actions.append("MOVE_DOWN_RIGHT")
        else: actions.append("MOVE_DOWN")
        
    if c > 0 and not (state.v_mask & (1 << (my_pos - 1))):
        nxt = my_pos - 1
        if nxt == opp_pos:
            if ocol > 0 and not (state.v_mask & (1 << (opp_pos - 1))): actions.append("MOVE_LEFT")
            else:
                if orow > 0 and not (state.h_mask & (1 << (opp_pos - 9))): actions.append("MOVE_UP_LEFT")
                if orow < 8 and not (state.h_mask & (1 << opp_pos)): actions.append("MOVE_DOWN_LEFT")
        else: actions.append("MOVE_LEFT")
        
    if c < 8 and not (state.v_mask & (1 << my_pos)):
        nxt = my_pos + 1
        if nxt == opp_pos:
            if ocol < 8 and not (state.v_mask & (1 << opp_pos)): actions.append("MOVE_RIGHT")
            else:
                if orow > 0 and not (state.h_mask & (1 << (opp_pos - 9))): actions.append("MOVE_UP_RIGHT")
                if orow < 8 and not (state.h_mask & (1 << opp_pos)): actions.append("MOVE_DOWN_RIGHT")
        else: actions.append("MOVE_RIGHT")
            
    # 2. Walls
    if state.walls_rem[me] > 0:
        tactical = set()
        _, opp_path = bfs_path(opp_pos, state.goals[opp], state.h_mask, state.v_mask)
        _, my_path = bfs_path(my_pos, state.goals[me], state.h_mask, state.v_mask)
        
        tactical.update(get_intersecting_walls(opp_path))
        tactical.update(get_intersecting_walls(my_path))
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                wr, wc = orow + dr, ocol + dc
                if 0 <= wr < 8 and 0 <= wc < 8:
                    tactical.add(('H', wr, wc))
                    tactical.add(('V', wr, wc))
                    
        for w in tactical:
            d, wr, wc = w
            if is_valid_wall(d, wr, wc, state.h_walls, state.v_walls):
                new_h = state.h_mask
                new_v = state.v_mask
                if d == 'H':
                    new_h |= (1 << (wr * 9 + wc)) | (1 << (wr * 9 + wc + 1))
                else:
                    new_v |= (1 << (wr * 9 + wc)) | (1 << ((wr + 1) * 9 + wc))
                
                md, _ = bfs_path(my_pos, state.goals[me], new_h, new_v)
                if md < 1000:
                    od, _ = bfs_path(opp_pos, state.goals[opp], new_h, new_v)
                    if od < 1000:
                        actions.append(f"WALL_{d}_{wr}_{wc}")
                        
    return actions

def apply_action(state, action):
    new_state = state.clone()
    me = new_state.actor
    
    if action.startswith("MOVE_"):
        d = MOVE_DELTAS[action]
        if "UP_LEFT" in action or "UP_RIGHT" in action or "DOWN_LEFT" in action or "DOWN_RIGHT" in action:
            new_state.pos[me] += d[0] * 9 + d[1]
        else:
            nxt = new_state.pos[me] + d[0] * 9 + d[1]
            if nxt == new_state.pos[1 - me]:
                new_state.pos[me] += d[0] * 18 + d[1] * 2
            else:
                new_state.pos[me] = nxt
    else:
        _, d, rs, cs = action.split('_')
        r, c = int(rs), int(cs)
        idx = r * 9 + c
        if d == 'H':
            new_state.h_walls |= (1 << idx)
            new_state.h_mask |= (1 << idx) | (1 << (idx + 1))
        else:
            new_state.v_walls |= (1 << idx)
            new_state.v_mask |= (1 << idx) | (1 << (idx + 9))
        new_state.walls_rem[me] -= 1
        
    new_state.actor = 1 - me
    new_state.turn += 1
    return new_state

def evaluate(state, me):
    opp = 1 - me
    my_dist, my_path = bfs_path(state.pos[me], state.goals[me], state.h_mask, state.v_mask)
    opp_dist, opp_path = bfs_path(state.pos[opp], state.goals[opp], state.h_mask, state.v_mask)
    
    if my_dist >= 1000: return -500000
    if opp_dist >= 1000: return 500000
    
    my_turns = my_dist * 2 - (1 if state.actor == me else 0)
    opp_turns = opp_dist * 2 - (1 if state.actor == opp else 0)
    
    score = (opp_turns - my_turns) * 200
    
    if state.walls_rem[opp] > 0:
        my_vuln = calc_vulnerability(state.pos[me], state.goals[me], my_path, state.h_mask, state.v_mask, state.h_walls, state.v_walls)
        score -= (my_vuln - my_dist) * 40
        
    if state.walls_rem[me] > 0:
        opp_vuln = calc_vulnerability(state.pos[opp], state.goals[opp], opp_path, state.h_mask, state.v_mask, state.h_walls, state.v_walls)
        score += (opp_vuln - opp_dist) * 60
        
    score += (state.walls_rem[me] - state.walls_rem[opp]) * 15
    
    my_c = state.pos[me] % 9
    opp_c = state.pos[opp] % 9
    score += (4 - abs(my_c - 4)) * 3
    score -= (4 - abs(opp_c - 4)) * 3
    
    return score

def order_actions(state, actions):
    me = state.actor
    opp = 1 - me
    my_pos = state.pos[me]
    opp_pos = state.pos[opp]
    
    scored = []
    for a in actions:
        if a.startswith("MOVE_"):
            d = MOVE_DELTAS[a]
            nr = my_pos // 9 + d[0]
            if not ("LEFT" in a or "RIGHT" in a):
                nxt = my_pos + d[0]*9 + d[1]
                if nxt == opp_pos:
                    nr += d[0]
            dist = abs(nr - state.goals[me])
            scored.append((1000 - dist * 10, a))
        else:
            _, d, rs, cs = a.split('_')
            r, c = int(rs), int(cs)
            wr = r + 0.5
            wc = c + 0.5
            orow, ocol = opp_pos // 9, opp_pos % 9
            dist_to_opp = abs(wr - orow) + abs(wc - ocol)
            scored.append((500 - dist_to_opp * 5, a))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored]

class Searcher:
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.start_time = time.perf_counter()
        self.nodes = 0
        self.tt = {}
        
    def search(self, state, depth, alpha, beta):
        self.nodes += 1
        if self.nodes % 256 == 0:
            if time.perf_counter() - self.start_time > self.time_limit:
                raise TimeoutError()
                
        if state.pos[0] // 9 == state.goals[0]:
            return (1000000 - state.turn) if state.actor == 0 else (-1000000 + state.turn), None
        if state.pos[1] // 9 == state.goals[1]:
            return (1000000 - state.turn) if state.actor == 1 else (-1000000 + state.turn), None
            
        if depth == 0:
            return evaluate(state, state.actor), None
            
        key = (state.actor, state.pos[0], state.pos[1], state.walls_rem[0], state.walls_rem[1], state.h_walls, state.v_walls)
        
        tt_entry = self.tt.get(key)
        if tt_entry is not None:
            tt_depth, tt_score, tt_action, tt_flag = tt_entry
            if tt_depth >= depth:
                if tt_flag == 'EXACT': return tt_score, tt_action
                elif tt_flag == 'LOWERBOUND': alpha = max(alpha, tt_score)
                elif tt_flag == 'UPPERBOUND': beta = min(beta, tt_score)
                if alpha >= beta: return tt_score, tt_action
                    
        actions = get_legal_actions(state)
        if not actions:
            return -1000000 + state.turn, None
            
        if tt_entry is not None and tt_entry[2] in actions:
            actions.remove(tt_entry[2])
            actions.insert(0, tt_entry[2])
        else:
            actions = order_actions(state, actions)
            
        best_score = -math.inf
        best_action = actions[0]
        original_alpha = alpha
        
        for action in actions:
            child = apply_action(state, action)
            score, _ = self.search(child, depth - 1, -beta, -alpha)
            score = -score
            
            if score > best_score:
                best_score = score
                best_action = action
                
            if score > alpha:
                alpha = score
                
            if alpha >= beta:
                break
                
        if best_score <= original_alpha: flag = 'UPPERBOUND'
        elif best_score >= beta: flag = 'LOWERBOUND'
        else: flag = 'EXACT'
            
        self.tt[key] = (depth, best_score, best_action, flag)
        return best_score, best_action

class Bot:
    name = "gemini_pro_v3"

    def choose_action(self, state_dict):
        legal_actions = state_dict.get("legal_actions", [])
        if not legal_actions:
            return ""
        if len(legal_actions) == 1:
            return legal_actions[0]
            
        me = int(state_dict.get("player_id", state_dict.get("actor", 0)))
        
        for action in legal_actions:
            if action.startswith("MOVE_"):
                d = MOVE_DELTAS[action]
                nr = state_dict["positions"][me][0] + d[0]
                if not ("LEFT" in action or "RIGHT" in action):
                    nxt_r = state_dict["positions"][me][0] + d[0]
                    nxt_c = state_dict["positions"][me][1] + d[1]
                    if [nxt_r, nxt_c] == state_dict["positions"][1-me]:
                        nr += d[0]
                if nr == state_dict.get("goal_rows", [0, 8])[me]:
                    return action
                    
        state = State()
        state.pos = [r * 9 + c for r, c in state_dict["positions"]]
        state.goals = tuple(state_dict.get("goal_rows", [0, 8]))
        state.walls_rem = list(state_dict["walls_remaining"])
        state.actor = int(state_dict.get("actor", me))
        state.turn = int(state_dict.get("turn", 0))
        
        h_mask = 0
        v_mask = 0
        h_walls = 0
        v_walls = 0
        
        for w in state_dict.get("walls", []):
            d, r, c = w["dir"], int(w["row"]), int(w["col"])
            idx = r * 9 + c
            if d == "H":
                h_walls |= (1 << idx)
                h_mask |= (1 << idx) | (1 << (idx + 1))
            else:
                v_walls |= (1 << idx)
                v_mask |= (1 << idx) | (1 << (idx + 9))
                
        state.h_mask = h_mask
        state.v_mask = v_mask
        state.h_walls = h_walls
        state.v_walls = v_walls
        
        # Slightly more time allowance if possible, but 0.45 is safe.
        searcher = Searcher(0.45)
        
        legal_set = set(legal_actions)
        root_actions = [a for a in get_legal_actions(state) if a in legal_set]
        if not root_actions:
            root_actions = [a for a in legal_actions if a.startswith("MOVE")]
            if not root_actions: root_actions = legal_actions
            
        root_actions = order_actions(state, root_actions)
        best_action = root_actions[0]
        
        try:
            for depth in range(1, 20):
                best_score = -math.inf
                alpha = -math.inf
                beta = math.inf
                
                current_best_action = root_actions[0]
                
                for action in root_actions:
                    child = apply_action(state, action)
                    score, _ = searcher.search(child, depth - 1, -beta, -alpha)
                    score = -score
                    
                    if score > best_score:
                        best_score = score
                        current_best_action = action
                        
                    if score > alpha:
                        alpha = score
                        
                best_action = current_best_action
                
                root_actions.remove(best_action)
                root_actions.insert(0, best_action)
                
                if best_score >= 500000:
                    break
        except TimeoutError:
            pass
            
        if best_action not in legal_set:
            return legal_actions[0]
            
        return best_action
