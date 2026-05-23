"""Strong LQQ bot with iterative deepening alpha-beta and path redundancy evaluation.
"""

import time
import collections

SAFETY_MARGIN_SECONDS = 0.15

BOARD_SIZE = 9
INF = 1000
WIN_SCORE = 1000000

class Bot:
    name = "gemini_pro"

    def __init__(self):
        self.transposition_table = {}
        self.killer_moves = {} # (depth, actor) -> [action1, action2]
        self.max_depth = 1
        self.start_time = 0
        self.time_limit = 0.7
        self.nodes_searched = 0

    def choose_action(self, state):
        self.start_time = time.perf_counter()
        timeout = state.get("decision_timeout") or state.get("time_limit")
        if timeout:
            self.time_limit = max(0.05, float(timeout) - SAFETY_MARGIN_SECONDS)
        self.transposition_table = {}
        self.killer_moves = {}
        self.nodes_searched = 0
        
        player_id = state["player_id"]
        legal_actions = state["legal_actions"]
        
        if not legal_actions:
            return ""
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Check for immediate winning move
        for action in legal_actions:
            if action.startswith("MOVE_"):
                res = self._simulate_move(state, action)
                if res["positions"][player_id][0] == state["goal_rows"][player_id]:
                    return action

        # Iterative Deepening
        best_action = legal_actions[0]
        try:
            for depth in range(1, 12):
                self.max_depth = depth
                score, action = self._alpha_beta(state, depth, -float('inf'), float('inf'), True)
                if action:
                    best_action = action
                if score >= WIN_SCORE - 500:
                    break
                # If we have very little time left, don't start next depth
                if time.perf_counter() - self.start_time > self.time_limit * 0.5:
                    break
        except TimeoutError:
            pass

        return best_action

    def _alpha_beta(self, state, depth, alpha, beta, is_maximizing):
        self.nodes_searched += 1
        if self.nodes_searched % 100 == 0:
            if time.perf_counter() - self.start_time > self.time_limit:
                raise TimeoutError()

        player_id = state["player_id"]
        actor = state["actor"]
        
        # Check game over
        winner = state.get("winner")
        if winner is not None:
            if winner == player_id:
                return WIN_SCORE - state["turn"], None
            else:
                return -WIN_SCORE + state["turn"], None

        if depth == 0:
            return self._evaluate(state, player_id), None

        # State hashing for TT
        state_key = self._get_state_key(state)
        if state_key in self.transposition_table:
            cached_depth, cached_score, cached_action = self.transposition_table[state_key]
            if cached_depth >= depth:
                return cached_score, cached_action

        legal = state["legal_actions"]
        if not legal:
            return (-WIN_SCORE if is_maximizing else WIN_SCORE), None

        # Move ordering
        ordered_actions = self._order_moves(state, legal, depth, is_maximizing)
        
        best_action = None
        if is_maximizing:
            max_eval = -float('inf')
            for action in ordered_actions:
                child_state = self._apply_action(state, action)
                eval_score, _ = self._alpha_beta(child_state, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    # Killer move heuristic
                    self._add_killer_move(depth, actor, action)
                    break
            self.transposition_table[state_key] = (depth, max_eval, best_action)
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in ordered_actions:
                child_state = self._apply_action(state, action)
                eval_score, _ = self._alpha_beta(child_state, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    # Killer move heuristic
                    self._add_killer_move(depth, actor, action)
                    break
            self.transposition_table[state_key] = (depth, min_eval, best_action)
            return min_eval, best_action

    def _add_killer_move(self, depth, actor, action):
        key = (depth, actor)
        if key not in self.killer_moves:
            self.killer_moves[key] = []
        if action not in self.killer_moves[key]:
            self.killer_moves[key].insert(0, action)
            self.killer_moves[key] = self.killer_moves[key][:2]

    def _get_state_key(self, state):
        pos = (tuple(state["positions"][0]), tuple(state["positions"][1]))
        walls = tuple(sorted((w["dir"], w["row"], w["col"]) for w in state["walls"]))
        return (pos, walls, state["actor"])

    def _order_moves(self, state, actions, depth, is_maximizing):
        scored_actions = []
        actor = state["actor"]
        opp_id = 1 - actor
        
        my_pos = state["positions"][actor]
        my_goal = state["goal_rows"][actor]
        
        # Get killers
        killers = self.killer_moves.get((depth, actor), [])
        
        # Pre-calculate distances for sorting
        my_dist = self._bfs(state, actor)

        for action in actions:
            score = 0
            if action in killers:
                score = 1000
            elif action.startswith("MOVE_"):
                new_pos = self._get_move_dest(state, actor, action)
                if new_pos:
                    # Bonus for moving towards goal
                    dist_to_goal = abs(new_pos[0] - my_goal)
                    score = 500 - dist_to_goal * 10
            elif action.startswith("WALL_"):
                # Heuristic: walls that are on opponent's shortest path
                # To keep it fast, we don't do full BFS for every wall here
                score = 100
            scored_actions.append((score, action))
            
        scored_actions.sort(key=lambda x: x[0], reverse=True)
        return [a[1] for a in scored_actions]

    def _evaluate(self, state, player_id):
        opp_id = 1 - player_id
        
        my_dist, my_path = self._bfs_with_path(state, player_id)
        opp_dist, opp_path = self._bfs_with_path(state, opp_id)
        
        if my_dist >= INF: return -WIN_SCORE // 2
        if opp_dist >= INF: return WIN_SCORE // 2

        # 1. Race score
        # Using arrival turns (considering whose turn it is)
        my_arrival = my_dist * 2 - (1 if state["actor"] == player_id else 0)
        opp_arrival = opp_dist * 2 - (1 if state["actor"] == opp_id else 0)
        score = (opp_arrival - my_arrival) * 1000
        
        # 2. Wall advantage
        score += (state["walls_remaining"][player_id] - state["walls_remaining"][opp_id]) * 150
        
        # 3. Robustness / Bottleneck score
        # Resilience: how much does the shortest path increase if we block its first step?
        my_resilience = self._get_resilience(state, player_id, my_dist, my_path)
        opp_resilience = self._get_resilience(state, opp_id, opp_dist, opp_path)
        
        score -= my_resilience * 50 # We want low resilience (alternative path is close)
        score += opp_resilience * 50 # We want high resilience for opponent (no alternative path)
        
        # 4. Progress and Centering
        my_row, my_col = state["positions"][player_id]
        opp_row, opp_col = state["positions"][opp_id]
        score += (4 - abs(my_col - 4)) * 30
        score -= (4 - abs(opp_col - 4)) * 30
        
        my_progress = abs(my_row - (8 if state["goal_rows"][player_id] == 0 else 0))
        opp_progress = abs(opp_row - (0 if state["goal_rows"][opp_id] == 8 else 8))
        score += my_progress * 40
        score -= opp_progress * 40

        return score

    def _get_resilience(self, state, player_id, dist, path):
        if not path or len(path) < 2:
            return 0
        
        # Temporarily block the first edge of the shortest path
        p1, p2 = path[0], path[1]
        blocked_edges = self._get_blocked_edges(state)
        
        # Add temporary block
        if p1[0] == p2[0]: # Same row, vertical wall between
            edge = ("V", p1[0], min(p1[1], p2[1]))
        else: # Same col, horizontal wall between
            edge = ("H", min(p1[0], p2[0]), p1[1])
            
        blocked_edges.add(edge)
        alt_dist = self._bfs_pos(state, tuple(p1), state["goal_rows"][player_id], blocked_edges)
        
        if alt_dist >= INF:
            return 20 # Very high penalty for having only one path
        return alt_dist - dist

    def _bfs(self, state, player_id):
        start_pos = tuple(state["positions"][player_id])
        goal_row = state["goal_rows"][player_id]
        return self._bfs_pos(state, start_pos, goal_row, self._get_blocked_edges(state))

    def _bfs_with_path(self, state, player_id):
        start_pos = tuple(state["positions"][player_id])
        goal_row = state["goal_rows"][player_id]
        queue = collections.deque([(start_pos, [start_pos])])
        visited = {start_pos}
        blocked_edges = self._get_blocked_edges(state)
        
        while queue:
            (r, c), path = queue.popleft()
            if r == goal_row:
                return len(path) - 1, path
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if not self._is_blocked(r, c, nr, nc, blocked_edges):
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append(((nr, nc), path + [(nr, nc)]))
        return INF, []

    def _bfs_pos(self, state, start_pos, goal_row, blocked_edges):
        queue = collections.deque([(start_pos, 0)])
        visited = {start_pos}
        while queue:
            (r, c), dist = queue.popleft()
            if r == goal_row:
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if not self._is_blocked(r, c, nr, nc, blocked_edges):
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append(((nr, nc), dist + 1))
        return INF

    def _is_blocked(self, r1, c1, r2, c2, blocked_edges):
        if r1 == r2:
            return ("V", r1, min(c1, c2)) in blocked_edges
        else:
            return ("H", min(r1, r2), c1) in blocked_edges

    def _get_blocked_edges(self, state):
        edges = set()
        for wall in state["walls"]:
            r, c, d = wall["row"], wall["col"], wall["dir"]
            if d == "H":
                edges.add(("H", r, c))
                edges.add(("H", r, c + 1))
            else:
                edges.add(("V", r, c))
                edges.add(("V", r + 1, c))
        return edges

    def _apply_action(self, state, action):
        # Deep copy state and apply action
        new_state = {
            "player_id": state["player_id"],
            "actor": 1 - state["actor"],
            "positions": [list(p) for p in state["positions"]],
            "walls_remaining": list(state["walls_remaining"]),
            "walls": [dict(w) for w in state["walls"]],
            "goal_rows": state["goal_rows"],
            "turn": state["turn"] + 1,
            "board_size": state["board_size"]
        }
        
        actor = state["actor"]
        if action.startswith("MOVE_"):
            dest = self._get_move_dest(state, actor, action)
            if dest:
                new_state["positions"][actor] = list(dest)
                if dest[0] == new_state["goal_rows"][actor]:
                    new_state["winner"] = actor
        elif action.startswith("WALL_"):
            parts = action.split("_")
            new_state["walls"].append({"dir": parts[1], "row": int(parts[2]), "col": int(parts[3])})
            new_state["walls_remaining"][actor] -= 1
            
        # Update legal actions for the next actor
        new_state["legal_actions"] = self._get_legal_actions(new_state, new_state["actor"])
        return new_state

    def _simulate_move(self, state, action):
        # Light version of apply_action for immediate check
        actor = state["actor"]
        dest = self._get_move_dest(state, actor, action)
        new_positions = [list(p) for p in state["positions"]]
        if dest:
            new_positions[actor] = list(dest)
        return {"positions": new_positions}

    def _get_move_dest(self, state, player_id, action):
        # Simplified move logic for internal search
        # In the actual game, MOVE_UP might be a jump. 
        # But legal_actions already contains the valid MOVE_UP etc.
        # We need to correctly identify where the player ends up.
        
        # Since state["legal_actions"] are pre-calculated by the referee,
        # we can't easily know the destination without re-implementing the jump logic.
        # However, we can use a heuristic: most moves are 1 step. Jumps are 2 steps.
        
        # Re-implementing jump logic from lqq_env.py
        dr, dc = self._get_delta(action)
        if dr == 0 and dc == 0: return None
        
        p = state["positions"][player_id]
        opp = state["positions"][1 - player_id]
        blocked_edges = self._get_blocked_edges(state)
        
        # Standard move
        nr, nc = p[0] + dr, p[1] + dc
        if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE): return None
        if self._is_blocked(p[0], p[1], nr, nc, blocked_edges): return None
        
        if nr != opp[0] or nc != opp[1]:
            return (nr, nc)
        
        # Jump logic
        if action in ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]:
            jr, jc = nr + dr, nc + dc
            if 0 <= jr < BOARD_SIZE and 0 <= jc < BOARD_SIZE and not self._is_blocked(nr, nc, jr, jc, blocked_edges):
                return (jr, jc)
            # Cannot jump forward, maybe side jump?
            # Side jumps are MOVE_UP_LEFT etc. handled below
            return None
        else:
            # Diagonal jumps
            # The action string like MOVE_UP_LEFT already implies the end direction
            # If we are here, it means we are jumping over opponent
            return (nr, nc)

    def _get_delta(self, action):
        deltas = {
            "MOVE_UP": (-1, 0), "MOVE_DOWN": (1, 0), "MOVE_LEFT": (0, -1), "MOVE_RIGHT": (0, 1),
            "MOVE_UP_LEFT": (-1, -1), "MOVE_UP_RIGHT": (-1, 1), "MOVE_DOWN_LEFT": (1, -1), "MOVE_DOWN_RIGHT": (1, 1),
        }
        return deltas.get(action, (0, 0))

    def _get_legal_actions(self, state, player_id):
        actions = []
        p = state["positions"][player_id]
        opp = state["positions"][1 - player_id]
        blocked_edges = self._get_blocked_edges(state)
        
        # Moves
        for move, (dr, dc) in {
            "MOVE_UP": (-1, 0), "MOVE_DOWN": (1, 0), "MOVE_LEFT": (0, -1), "MOVE_RIGHT": (0, 1)
        }.items():
            nr, nc = p[0] + dr, p[1] + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and not self._is_blocked(p[0], p[1], nr, nc, blocked_edges):
                if nr != opp[0] or nc != opp[1]:
                    actions.append(move)
                else:
                    # Jump
                    jr, jc = nr + dr, nc + dc
                    if 0 <= jr < BOARD_SIZE and 0 <= jc < BOARD_SIZE and not self._is_blocked(nr, nc, jr, jc, blocked_edges):
                        actions.append(move)
                    else:
                        # Side jumps
                        for sdr, sdc in ([(-1, 0), (1, 0)] if dr == 0 else [(0, -1), (0, 1)]):
                            sr, sc = nr + sdr, nc + sdc
                            if 0 <= sr < BOARD_SIZE and 0 <= sc < BOARD_SIZE and not self._is_blocked(nr, nc, sr, sc, blocked_edges):
                                # Map to MOVE_...
                                if dr == -1: actions.append("MOVE_UP_LEFT" if sdc == -1 else "MOVE_UP_RIGHT")
                                elif dr == 1: actions.append("MOVE_DOWN_LEFT" if sdc == -1 else "MOVE_DOWN_RIGHT")
                                elif dc == -1: actions.append("MOVE_UP_LEFT" if sdr == -1 else "MOVE_DOWN_LEFT")
                                elif dc == 1: actions.append("MOVE_UP_RIGHT" if sdr == -1 else "MOVE_DOWN_RIGHT")

        # Walls
        if state["walls_remaining"][player_id] > 0:
            # To keep it fast, we only consider a subset of walls:
            # 1. Walls that intercept shortest paths
            # 2. Walls near players
            potential_walls = self._get_potential_walls(state)
            for wall_action in potential_walls:
                if self._is_wall_legal(state, wall_action, blocked_edges):
                    actions.append(wall_action)
            
        return actions

    def _get_potential_walls(self, state):
        potential = set()
        # Only consider walls near the players and on the shortest paths
        for pid in [0, 1]:
            pr, pc = state["positions"][pid]
            for r in range(max(0, pr-2), min(8, pr+1)):
                for c in range(max(0, pc-2), min(8, pc+1)):
                    potential.add(f"WALL_H_{r}_{c}")
                    potential.add(f"WALL_V_{r}_{c}")
                    
        # Add walls that block the opponent's path
        opp_id = 1 - state["actor"]
        path = self._get_shortest_path_coords(state, opp_id)
        for r, c in path[:6]: 
            for dr in [-1, 0]:
                for dc in [-1, 0]:
                    wr, wc = r + dr, c + dc
                    if 0 <= wr < 8 and 0 <= wc < 8:
                        potential.add(f"WALL_H_{wr}_{wc}")
                        potential.add(f"WALL_V_{wr}_{wc}")
        return potential

    def _get_shortest_path_coords(self, state, player_id):
        start_pos = tuple(state["positions"][player_id])
        goal_row = state["goal_rows"][player_id]
        queue = collections.deque([(start_pos, [])])
        visited = {start_pos}
        blocked_edges = self._get_blocked_edges(state)
        while queue:
            (r, c), path = queue.popleft()
            if r == goal_row:
                return path + [(r, c)]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and not self._is_blocked(r, c, nr, nc, blocked_edges):
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(r, c)]))
        return []

    def _is_wall_legal(self, state, wall_action, current_blocked_edges):
        parts = wall_action.split("_")
        w_dir, w_row, w_col = parts[1], int(parts[2]), int(parts[3])
        
        # Fast overlap check
        for w in state["walls"]:
            if w["row"] == w_row and w["col"] == w_col: return False
            if w["dir"] == w_dir:
                if w_dir == "H" and w["row"] == w_row and abs(w["col"] - w_col) <= 1: return False
                if w_dir == "V" and w["col"] == w_col and abs(w["row"] - w_row) <= 1: return False
            elif w["row"] == w_row and w["col"] == w_col: return False
        
        # Path check
        new_blocked = set(current_blocked_edges)
        if w_dir == "H":
            new_blocked.add(("H", w_row, w_col))
            new_blocked.add(("H", w_row, w_col + 1))
        else:
            new_blocked.add(("V", w_row, w_col))
            new_blocked.add(("V", w_row + 1, w_col))
            
        return self._has_path(state, 0, new_blocked) and self._has_path(state, 1, new_blocked)

    def _has_path(self, state, player_id, blocked_edges):
        start_pos = tuple(state["positions"][player_id])
        goal_row = state["goal_rows"][player_id]
        queue = collections.deque([start_pos])
        visited = {start_pos}
        while queue:
            r, c = queue.popleft()
            if r == goal_row: return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and not self._is_blocked(r, c, nr, nc, blocked_edges):
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

def choose_action(state):
    return Bot().choose_action(state)
