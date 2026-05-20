import time
import collections

# Board constants
BOARD_SIZE = 9
TOTAL_SQUARES = BOARD_SIZE * BOARD_SIZE

class Bot:
    name = "gemini-hard"

    def __init__(self):
        self.transposition_table = {}
        self.start_time = 0
        self.time_limit = 0.94
        self.max_depth = 0

    def get_state_key(self, state):
        pos = (state["positions"][0][0], state["positions"][0][1], 
               state["positions"][1][0], state["positions"][1][1])
        walls = tuple((w["dir"], w["row"], w["col"]) for w in state["walls"])
        walls_rem = (state["walls_remaining"][0], state["walls_remaining"][1])
        return (pos, walls, walls_rem, state["actor"])

    def get_distances_and_path(self, r, c, goal_row, h_walls, v_walls, return_path=False):
        dist = [-1] * TOTAL_SQUARES
        parent = {}
        start_idx = r * BOARD_SIZE + c
        dist[start_idx] = 0
        q = collections.deque([start_idx])
        goal_idx = -1
        while q:
            curr = q.popleft()
            curr_r, curr_c = curr // BOARD_SIZE, curr % BOARD_SIZE
            if curr_r == goal_row:
                if goal_idx == -1: goal_idx = curr
                # We can't break if we want all paths, but for one path it's fine
                break
            for dr, dc, move_name in [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    blocked = False
                    if move_name == "UP": blocked = (curr_r-1, curr_c) in h_walls or (curr_c > 0 and (curr_r-1, curr_c-1) in h_walls)
                    elif move_name == "DOWN": blocked = (curr_r, curr_c) in h_walls or (curr_c > 0 and (curr_r, curr_c-1) in h_walls)
                    elif move_name == "LEFT": blocked = (curr_r, curr_c-1) in v_walls or (curr_r > 0 and (curr_r-1, curr_c-1) in v_walls)
                    elif move_name == "RIGHT": blocked = (curr_r, curr_c) in v_walls or (curr_r > 0 and (curr_r-1, curr_c) in v_walls)
                    if not blocked:
                        idx = nr * BOARD_SIZE + nc
                        if dist[idx] == -1:
                            dist[idx] = dist[curr] + 1
                            if return_path: parent[idx] = curr
                            q.append(idx)
        if goal_idx == -1: return 100, []
        path = []
        if return_path:
            curr = goal_idx
            while curr != start_idx:
                path.append(curr)
                curr = parent[curr]
            path.append(start_idx)
            path.reverse()
        return dist[goal_idx], path

    def evaluate(self, state):
        actor, opp = state["actor"], 1 - state["actor"]
        if state.get("winner") == actor: return 10000
        if state.get("winner") == opp: return -10000
        h_walls, v_walls = self.get_wall_sets(state["walls"])
        my_pos, opp_pos = state["positions"][actor], state["positions"][opp]
        my_dist, _ = self.get_distances_and_path(my_pos[0], my_pos[1], state["goal_rows"][actor], h_walls, v_walls)
        opp_dist, _ = self.get_distances_and_path(opp_pos[0], opp_pos[1], state["goal_rows"][opp], h_walls, v_walls)
        score = (opp_dist - my_dist) * 40 + (state["walls_remaining"][actor] - state["walls_remaining"][opp]) * 15
        score += (4 - abs(my_pos[1] - 4)) * 5 - (4 - abs(opp_pos[1] - 4)) * 3
        # Favor being further along the board
        score += (8 - abs(my_pos[0] - state["goal_rows"][actor])) * 2
        return score

    def get_wall_sets(self, walls):
        h, v = set(), set()
        for w in walls:
            if w["dir"] == "H": h.add((w["row"], w["col"]))
            else: v.add((w["row"], w["col"]))
        return h, v

    def choose_action(self, state):
        self.start_time = time.time()
        self.transposition_table = {}
        best_action = state["legal_actions"][0]
        try:
            for depth in range(1, 10):
                self.max_depth = depth
                score, action = self.search(state, depth)
                if action: best_action = action
                if score > 9000: break
                if time.time() - self.start_time > self.time_limit: break
        except Exception: pass
        return best_action

    def search(self, state, depth):
        alpha, beta = -20000, 20000
        best_val, best_move = -20000, None
        actions = self.order_actions(state, state["legal_actions"], depth)
        for action in actions:
            if time.time() - self.start_time > self.time_limit: break
            next_state = self.apply_action_fast(state, action)
            val = -self.negamax(next_state, depth - 1, -beta, -alpha)
            if val > best_val: best_val, best_move = val, action
            alpha = max(alpha, val)
            if alpha >= beta: break
        return best_val, best_move

    def negamax(self, state, depth, alpha, beta):
        if time.time() - self.start_time > self.time_limit: return 0
        state_key = self.get_state_key(state)
        if state_key in self.transposition_table:
            entry = self.transposition_table[state_key]
            if entry['depth'] >= depth: return entry['score']
        if state.get("winner") is not None: return self.evaluate(state)
        if depth == 0: return self.evaluate(state)
        actions = self.order_actions(state, state.get("legal_actions", []), depth)
        if not actions: return -10000
        best_val = -20000
        limit = 12 if depth > 2 else (25 if depth > 1 else 100)
        for action in actions[:limit]:
            next_state = self.apply_action_fast(state, action)
            val = -self.negamax(next_state, depth - 1, -beta, -alpha)
            best_val = max(best_val, val)
            alpha = max(alpha, val)
            if alpha >= beta: break
        self.transposition_table[state_key] = {'depth': depth, 'score': best_val}
        return best_val

    def order_actions(self, state, actions, depth):
        scored = []
        actor, opp = state["actor"], 1 - state["actor"]
        my_pos, opp_pos = state["positions"][actor], state["positions"][opp]
        goal, opp_goal = state["goal_rows"][actor], state["goal_rows"][opp]
        h_walls, v_walls = self.get_wall_sets(state["walls"])
        base_my_dist, _ = self.get_distances_and_path(my_pos[0], my_pos[1], goal, h_walls, v_walls)
        base_opp_dist, opp_path = self.get_distances_and_path(opp_pos[0], opp_pos[1], opp_goal, h_walls, v_walls, True)
        for a in actions:
            score = 0
            if a.startswith("MOVE_"):
                dest = self.get_move_destination(state, actor, a)
                if dest:
                    if dest[0] == goal: score = 1200
                    else: score = (base_my_dist - abs(dest[0] - goal)) * 25 + (4 - abs(dest[1] - 4)) * 2
                else: score = -500
            elif a.startswith("WALL_"):
                parts = a.split("_")
                d, r, c = parts[1], int(parts[2]), int(parts[3])
                on_path = False
                for i in range(len(opp_path)-1):
                    p1, p2 = opp_path[i], opp_path[i+1]
                    r1, c1, r2, c2 = p1//9, p1%9, p2//9, p2%9
                    if d == "H":
                        if (r1 == r and r2 == r+1 and (c1 == c or c1 == c+1)) or (r1 == r+1 and r2 == r and (c1 == c or c1 == c+1)):
                            on_path = True; break
                    else:
                        if (c1 == c and c2 == c+1 and (r1 == r or r1 == r+1)) or (c1 == c+1 and c2 == c and (r1 == r or r1 == r+1)):
                            on_path = True; break
                if on_path:
                    nh, nv = h_walls.copy(), v_walls.copy()
                    if d == "H": nh.add((r, c))
                    else: nv.add((r, c))
                    nd_opp, _ = self.get_distances_and_path(opp_pos[0], opp_pos[1], opp_goal, nh, nv)
                    nd_my, _ = self.get_distances_and_path(my_pos[0], my_pos[1], goal, nh, nv)
                    if nd_opp == 100: score = -1000
                    else: score = (nd_opp - base_opp_dist) * 60 - (nd_my - base_my_dist) * 30
                else: score = -150
            scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored]

    def apply_action_fast(self, state, action):
        actor = state["actor"]
        new_s = {"player_id": state["player_id"], "actor": 1-actor, "positions": [list(p) for p in state["positions"]],
                 "walls_remaining": list(state["walls_remaining"]), "walls": list(state["walls"]),
                 "goal_rows": state["goal_rows"], "turn": state["turn"]+1, "board_size": 9}
        if action.startswith("MOVE_"):
            dest = self.get_move_destination(state, actor, action)
            if dest:
                new_s["positions"][actor] = list(dest)
                if dest[0] == state["goal_rows"][actor]: new_s["winner"] = actor
        elif action.startswith("WALL_"):
            parts = action.split("_")
            d, r, c = parts[1], int(parts[2]), int(parts[3])
            w = {"dir": d, "row": r, "col": c}
            inserted = False
            for i, existing in enumerate(new_s["walls"]):
                if (d, r, c) < (existing["dir"], existing["row"], existing["col"]):
                    new_s["walls"].insert(i, w); inserted = True; break
            if not inserted: new_s["walls"].append(w)
            new_s["walls_remaining"][actor] -= 1
        new_s["legal_actions"] = self.get_legal_actions_simplified(new_s)
        return new_s

    def get_move_destination(self, state, actor, action):
        p, o = state["positions"][actor], state["positions"][1-actor]
        hw, vw = self.get_wall_sets(state["walls"])
        deltas = {"MOVE_UP": (-1,0), "MOVE_DOWN": (1,0), "MOVE_LEFT": (0,-1), "MOVE_RIGHT": (0,1),
                  "MOVE_UP_LEFT": (-1,-1), "MOVE_UP_RIGHT": (-1,1), "MOVE_DOWN_LEFT": (1,-1), "MOVE_DOWN_RIGHT": (1,1)}
        dr, dc = deltas.get(action, (0,0))
        def blocked(r1, c1, r2, c2):
            if r1 == r2: return (r1, min(c1, c2)) in vw or (r1 > 0 and (r1-1, min(c1, c2)) in vw)
            if c1 == c2: return (min(r1, r2), c1) in hw or (c1 > 0 and (min(r1, r2), c1-1) in hw)
            return True
        if action in ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]:
            nr, nc = p[0]+dr, p[1]+dc
            if not (0<=nr<9 and 0<=nc<9) or blocked(p[0], p[1], nr, nc): return None
            if nr == o[0] and nc == o[1]:
                jr, jc = nr+dr, nc+dc
                if 0<=jr<9 and 0<=jc<9 and not blocked(nr, nc, jr, jc): return jr, jc
                return None
            return nr, nc
        return p[0]+dr, p[1]+dc if action.startswith("MOVE_") else None

    def get_legal_actions_simplified(self, state):
        actor, p, o = state["actor"], state["positions"][state["actor"]], state["positions"][1-state["actor"]]
        hw, vw = self.get_wall_sets(state["walls"])
        actions = []
        def is_blocked(r1, c1, r2, c2):
            if r1 == r2: return (r1, min(c1, c2)) in vw or (r1 > 0 and (r1-1, min(c1, c2)) in vw)
            if c1 == c2: return (min(r1, r2), c1) in hw or (c1 > 0 and (min(r1, r2), c1-1) in hw)
            return True
        for name, (dr, dc) in [("MOVE_UP", (-1,0)), ("MOVE_DOWN", (1,0)), ("MOVE_LEFT", (0,-1)), ("MOVE_RIGHT", (0,1))]:
            nr, nc = p[0]+dr, p[1]+dc
            if 0<=nr<9 and 0<=nc<9 and not is_blocked(p[0], p[1], nr, nc):
                if nr == o[0] and nc == o[1]:
                    jr, jc = nr+dr, nc+dc
                    if 0<=jr<9 and 0<=jc<9 and not is_blocked(nr, nc, jr, jc): actions.append(name)
                    else:
                        for dnr, dnc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                            sr, sc = p[0]+dnr, p[1]+dnc
                            if 0<=sr<9 and 0<=sc<9 and not is_blocked(nr, nc, sr, sc):
                                if dnr == -1 and dnc == -1: actions.append("MOVE_UP_LEFT")
                                elif dnr == -1 and dnc == 1: actions.append("MOVE_UP_RIGHT")
                                elif dnr == 1 and dnc == -1: actions.append("MOVE_DOWN_LEFT")
                                elif dnr == 1 and dnc == 1: actions.append("MOVE_DOWN_RIGHT")
                else: actions.append(name)
        return actions

def choose_action(state):
    return Bot().choose_action(state)
