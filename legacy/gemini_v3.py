"""Gemini MCTS Bot V3.

The absolute ceiling of MCTS in Python for Quoridor.
Features:
- MCTS Solver: Propagates exact wins (1) and exact losses (-1) up the tree immediately.
- Unified Race Metric: Calculates exact turn advantage changes for perfectly informed priors.
- Expanded Tactical Vision: Explores walls up to radius 2 around the opponent to set and spot traps.
- "Heuristic PUCT": Replaces random rollouts with a strong evaluation function at leaf nodes.
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


def _time_budget(state, fallback):
    timeout = state.get("decision_timeout") or state.get("time_limit")
    if timeout:
        return max(0.05, float(timeout) - 0.15)
    return fallback

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
                
            if r > 0:
                nxt = curr - 9
                if not (h_mask & (1 << nxt)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q[tail] = nxt
                    tail += 1
            if r < 8:
                nxt = curr + 9
                if not (h_mask & (1 << curr)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q[tail] = nxt
                    tail += 1
            c = curr % 9
            if c > 0:
                nxt = curr - 1
                if not (v_mask & (1 << nxt)) and not (visited & (1 << nxt)):
                    visited |= (1 << nxt)
                    parent[nxt] = curr
                    q[tail] = nxt
                    tail += 1
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

def get_intersecting_walls_list(path):
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

def calc_vulnerability(pos, target_row, path, h_mask, v_mask, h_walls, v_walls):
    key = (pos, target_row, h_mask, v_mask, h_walls, v_walls)
    if key in VULN_CACHE:
        return VULN_CACHE[key]
        
    walls = get_intersecting_walls_list(path)
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

def get_intersecting_walls_set(path):
    walls = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if abs(u - v) == 9:
            top = min(u, v)
            r, c = top // 9, top % 9
            if c < 8: walls.add(('H', r, c))
            if c > 0: walls.add(('H', r, c - 1))
        elif abs(u - v) == 1:
            left = min(u, v)
            r, c = left // 9, left % 9
            if r < 8: walls.add(('V', r, c))
            if r > 0: walls.add(('V', r - 1, c))
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

def get_legal_actions_with_priors(state, my_dist, my_path, opp_dist, opp_path):
    actions = []
    me = state.actor
    opp = 1 - me
    my_pos = state.pos[me]
    opp_pos = state.pos[opp]
    r, c = my_pos // 9, my_pos % 9
    orow, ocol = opp_pos // 9, opp_pos % 9
    
    my_arrival = my_dist * 2 - 2
    opp_arrival = opp_dist * 2 - 1
    old_race = opp_arrival - my_arrival
    
    moves = []
    if r > 0 and not (state.h_mask & (1 << (my_pos - 9))):
        nxt = my_pos - 9
        if nxt == opp_pos:
            if orow > 0 and not (state.h_mask & (1 << (opp_pos - 9))): moves.append(("MOVE_UP", opp_pos - 9))
            else:
                if ocol > 0 and not (state.v_mask & (1 << (opp_pos - 1))): moves.append(("MOVE_UP_LEFT", opp_pos - 1))
                if ocol < 8 and not (state.v_mask & (1 << opp_pos)): moves.append(("MOVE_UP_RIGHT", opp_pos + 1))
        else: moves.append(("MOVE_UP", nxt))
        
    if r < 8 and not (state.h_mask & (1 << my_pos)):
        nxt = my_pos + 9
        if nxt == opp_pos:
            if orow < 8 and not (state.h_mask & (1 << opp_pos)): moves.append(("MOVE_DOWN", opp_pos + 9))
            else:
                if ocol > 0 and not (state.v_mask & (1 << (opp_pos - 1))): moves.append(("MOVE_DOWN_LEFT", opp_pos - 1))
                if ocol < 8 and not (state.v_mask & (1 << opp_pos)): moves.append(("MOVE_DOWN_RIGHT", opp_pos + 1))
        else: moves.append(("MOVE_DOWN", nxt))
        
    if c > 0 and not (state.v_mask & (1 << (my_pos - 1))):
        nxt = my_pos - 1
        if nxt == opp_pos:
            if ocol > 0 and not (state.v_mask & (1 << (opp_pos - 1))): moves.append(("MOVE_LEFT", opp_pos - 1))
            else:
                if orow > 0 and not (state.h_mask & (1 << (opp_pos - 9))): moves.append(("MOVE_UP_LEFT", opp_pos - 9))
                if orow < 8 and not (state.h_mask & (1 << opp_pos)): moves.append(("MOVE_DOWN_LEFT", opp_pos + 9))
        else: moves.append(("MOVE_LEFT", nxt))
        
    if c < 8 and not (state.v_mask & (1 << my_pos)):
        nxt = my_pos + 1
        if nxt == opp_pos:
            if ocol < 8 and not (state.v_mask & (1 << opp_pos)): moves.append(("MOVE_RIGHT", opp_pos + 1))
            else:
                if orow > 0 and not (state.h_mask & (1 << (opp_pos - 9))): moves.append(("MOVE_UP_RIGHT", opp_pos - 9))
                if orow < 8 and not (state.h_mask & (1 << opp_pos)): moves.append(("MOVE_DOWN_RIGHT", opp_pos + 9))
        else: moves.append(("MOVE_RIGHT", nxt))
        
    for move, dest in moves:
        if dest // 9 == state.goals[me]:
            return [(move, 1000.0)]
            
    next_best_pos = my_path[1] if len(my_path) > 1 else -1
    for move, dest in moves:
        if dest == next_best_pos:
            p = 20.0
            if old_race >= 0: p += 30.0
            actions.append((move, p))
        else:
            actions.append((move, 1.0))
            
    if state.walls_rem[me] > 0:
        my_path_walls = get_intersecting_walls_set(my_path)
        opp_path_walls = get_intersecting_walls_set(opp_path)
        
        tactical = set()
        tactical.update(my_path_walls)
        tactical.update(opp_path_walls)
        
        # Radius 2 around opponent for trap setting
        for dr in [-2, -1, 0, 1, 2]:
            for dc in [-2, -1, 0, 1, 2]:
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
                
                md = my_dist
                if w in my_path_walls:
                    md, _ = bfs_path(my_pos, state.goals[me], new_h, new_v)
                if md >= 1000: continue
                
                od = opp_dist
                if w in opp_path_walls or True: # always eval opponent dist for radius 2 walls
                    od, _ = bfs_path(opp_pos, state.goals[opp], new_h, new_v)
                if od >= 1000: continue
                
                new_my_arrival = md * 2
                new_opp_arrival = od * 2 - 1
                new_race = new_opp_arrival - new_my_arrival
                
                if new_race > 0 and old_race <= 0:
                    p = 50.0 + (new_race - old_race) * 5.0
                elif new_race > old_race:
                    p = 15.0 * (new_race - old_race)
                elif od > opp_dist:
                    p = 2.0
                else:
                    p = 0.1
                actions.append((f"WALL_{d}_{wr}_{wc}", p))
                    
    return actions

class Node:
    __slots__ = ['state', 'parent', 'action_taken', 'children', 'N', 'W', 'P', 'is_expanded', 'status']
    def __init__(self, state, parent, action_taken, P):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.N = 0
        self.W = 0.0
        self.P = P
        self.is_expanded = False
        self.status = 0 # 0=unknown, 1=guaranteed win (for node.state.actor), -1=guaranteed loss (for node.state.actor)

class MCTS:
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.c_puct = 2.0
        self.root = None
        
    def expand_and_evaluate(self, node):
        state = node.state
        me = state.actor
        opp = 1 - me
        
        if state.pos[me] // 9 == state.goals[me]:
            node.status = 1
            return 1.0
            
        my_dist, my_path = bfs_path(state.pos[me], state.goals[me], state.h_mask, state.v_mask)
        opp_dist, opp_path = bfs_path(state.pos[opp], state.goals[opp], state.h_mask, state.v_mask)
        
        if my_dist >= 1000:
            node.status = -1
            return -1.0
        if opp_dist >= 1000:
            node.status = 1
            return 1.0
            
        my_turns = my_dist * 2
        opp_turns = opp_dist * 2 - 1 
        
        my_vuln = my_dist
        if state.walls_rem[opp] > 0:
            my_vuln = calc_vulnerability(state.pos[me], state.goals[me], my_path, state.h_mask, state.v_mask, state.h_walls, state.v_walls)
            
        opp_vuln = opp_dist
        if state.walls_rem[me] > 0:
            opp_vuln = calc_vulnerability(state.pos[opp], state.goals[opp], opp_path, state.h_mask, state.v_mask, state.h_walls, state.v_walls)
            
        race_diff = opp_turns - my_turns 
        race_diff -= (my_vuln - my_dist) * 1.5
        race_diff += (opp_vuln - opp_dist) * 1.5
        
        value = math.tanh(race_diff * 0.25)
        
        wall_diff = state.walls_rem[me] - state.walls_rem[opp]
        value += wall_diff * 0.04
        
        my_c = state.pos[me] % 9
        opp_c = state.pos[opp] % 9
        value += (abs(opp_c - 4) - abs(my_c - 4)) * 0.01
        
        value = max(-0.99, min(0.99, value))
        
        node.is_expanded = True
        
        actions = get_legal_actions_with_priors(state, my_dist, my_path, opp_dist, opp_path)
        if not actions:
            node.status = -1
            return -1.0
            
        total_p = sum(p for a, p in actions)
        for a, p in actions:
            node.children.append(Node(apply_action(state, a), node, a, p / total_p))
                
        return value

    def search(self, state):
        current_hash = (state.actor, state.pos[0], state.pos[1], state.walls_rem[0], state.walls_rem[1], state.h_walls, state.v_walls)
        
        if self.root is not None:
            found = False
            q = [self.root]
            while q:
                node = q.pop(0)
                if (node.state.actor, node.state.pos[0], node.state.pos[1], node.state.walls_rem[0], node.state.walls_rem[1], node.state.h_walls, node.state.v_walls) == current_hash:
                    self.root = node
                    self.root.parent = None
                    found = True
                    break
                q.extend(node.children)
            if not found:
                self.root = Node(state, None, None, 1.0)
        else:
            self.root = Node(state, None, None, 1.0)
            
        start_time = time.perf_counter()
        nodes_searched = 0
        
        if not self.root.is_expanded and self.root.status == 0:
            self.expand_and_evaluate(self.root)
            
        while True:
            nodes_searched += 1
            if nodes_searched % 16 == 0:
                if time.perf_counter() - start_time > self.time_limit:
                    break
                    
            node = self.root
            path = [node]
            
            while node.is_expanded and node.children and node.status == 0:
                best_child = None
                best_uct = -math.inf
                sqrt_N = math.sqrt(node.N)
                
                for child in node.children:
                    if child.status == -1: # Found winning move!
                        node.status = 1
                        break
                    
                    if child.status == 1: # Proven loss, avoid
                        continue
                        
                    q_node = -(child.W / child.N) if child.N > 0 else 0.0
                    uct = q_node + self.c_puct * child.P * sqrt_N / (1 + child.N)
                    if uct > best_uct:
                        best_uct = uct
                        best_child = child
                        
                if node.status == 1:
                    break
                    
                if best_child is None:
                    node.status = -1
                    break
                    
                node = best_child
                path.append(node)
                
            if node.status != 0:
                value = node.status
            elif not node.is_expanded:
                value = self.expand_and_evaluate(node)
                if node.status != 0:
                    value = node.status
            else:
                value = 0
                
            current_val = value
            for n in reversed(path):
                n.N += 1
                n.W += current_val
                
                if n.status == 0 and n.is_expanded and n.children:
                    has_win = False
                    all_loss = True
                    for c in n.children:
                        if c.status == -1:
                            has_win = True
                            break
                        if c.status != 1:
                            all_loss = False
                    if has_win:
                        n.status = 1
                    elif all_loss:
                        n.status = -1
                        
                current_val = -current_val

class Bot:
    name = "gemini_MCTS_v3"

    def __init__(self):
        self.mcts = MCTS(0.85)

    def choose_action(self, state_dict):
        legal_actions = state_dict.get("legal_actions", [])
        if not legal_actions:
            return ""
        if len(legal_actions) == 1:
            return legal_actions[0]
        self.mcts.time_limit = _time_budget(state_dict, 0.85)
            
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
        
        self.mcts.search(state)
        
        legal_set = set(legal_actions)
        
        for child in self.mcts.root.children:
            if child.status == -1 and child.action_taken in legal_set:
                return child.action_taken
                
        valid_children = [c for c in self.mcts.root.children if c.action_taken in legal_set and c.status != 1]
        
        if valid_children:
            best_child = max(valid_children, key=lambda c: c.N)
            return best_child.action_taken
            
        valid_children_any = [c for c in self.mcts.root.children if c.action_taken in legal_set]
        if valid_children_any:
            best_child = max(valid_children_any, key=lambda c: c.N)
            return best_child.action_taken
            
        return legal_actions[0]
