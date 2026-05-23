from __future__ import annotations

from collections import deque
import time


BOARD_SIZE = 9
INF = 10**9
SAFETY_SECONDS = 1.35
SAFETY_MARGIN_SECONDS = 0.15

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

CARDINAL_MOVE_DELTAS = {
    "MOVE_UP": (-1, 0),
    "MOVE_DOWN": (1, 0),
    "MOVE_LEFT": (0, -1),
    "MOVE_RIGHT": (0, 1),
}


class SearchTimeout(Exception):
    pass


def wall_edges(direction, row, col):
    if direction == "H":
        return (("H", row, col), ("H", row, col + 1))
    return (("V", row, col), ("V", row + 1, col))


def opposite_wall_direction(direction):
    return "V" if direction == "H" else "H"


def parse_wall_action(action):
    parts = action.split("_")
    if len(parts) != 4 or parts[0] != "WALL" or parts[1] not in ("H", "V"):
        return None
    try:
        return parts[1], int(parts[2]), int(parts[3])
    except ValueError:
        return None


def in_bounds(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def clamp_wall_origin(direction, row, col):
    if direction == "H" and 0 <= row < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1:
        return True
    if direction == "V" and 0 <= row < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1:
        return True
    return False


class MiniState:
    __slots__ = (
        "current",
        "positions",
        "goals",
        "walls_remaining",
        "walls",
        "blocked_edges",
        "turn",
        "winner",
        "_dist_cache",
        "_path_cache",
    )

    def __init__(
        self,
        current,
        positions,
        goals,
        walls_remaining,
        walls,
        blocked_edges,
        turn,
        winner=None,
    ):
        self.current = current
        self.positions = positions
        self.goals = goals
        self.walls_remaining = walls_remaining
        self.walls = walls
        self.blocked_edges = blocked_edges
        self.turn = turn
        self.winner = winner
        self._dist_cache = [None, None]
        self._path_cache = [None, None]

    @classmethod
    def from_bot_state(cls, state):
        walls = set()
        blocked = set()
        for item in state.get("walls", ()):
            direction = item["dir"]
            row = item["row"]
            col = item["col"]
            walls.add((direction, row, col))
            blocked.update(wall_edges(direction, row, col))
        current = int(state.get("actor", state.get("player_id", 0)))
        return cls(
            current=current,
            positions=tuple((int(row), int(col)) for row, col in state["positions"]),
            goals=tuple(int(goal) for goal in state["goal_rows"]),
            walls_remaining=tuple(int(count) for count in state["walls_remaining"]),
            walls=frozenset(walls),
            blocked_edges=frozenset(blocked),
            turn=int(state.get("turn", 0)),
            winner=state.get("winner"),
        )

    def apply(self, action):
        return self.apply_as(self.current, action)

    def apply_as(self, actor, action):
        positions = [list(pos) for pos in self.positions]
        walls_remaining = list(self.walls_remaining)
        walls = set(self.walls)
        blocked = set(self.blocked_edges)
        winner = None

        if action in MOVE_DELTAS:
            destination = self.move_destination(actor, action)
            if destination is None:
                return self
            positions[actor][0], positions[actor][1] = destination
            if positions[actor][0] == self.goals[actor]:
                winner = actor
        else:
            parsed = parse_wall_action(action)
            if parsed is None:
                return self
            direction, row, col = parsed
            walls.add((direction, row, col))
            blocked.update(wall_edges(direction, row, col))
            walls_remaining[actor] -= 1

        return MiniState(
            current=1 - actor,
            positions=tuple((row, col) for row, col in positions),
            goals=self.goals,
            walls_remaining=tuple(walls_remaining),
            walls=frozenset(walls),
            blocked_edges=frozenset(blocked),
            turn=self.turn + 1,
            winner=winner,
        )

    def legal_moves(self, player=None):
        if player is None:
            player = self.current
        result = []
        for action in MOVE_DELTAS:
            if self.move_destination(player, action) is not None:
                result.append(action)
        return result

    def legal_actions_full(self):
        if self.winner is not None:
            return []
        player = self.current
        actions = self.legal_moves(player)
        if self.walls_remaining[player] <= 0:
            return actions

        for row in range(BOARD_SIZE - 1):
            for col in range(BOARD_SIZE - 1):
                if self.is_wall_legal("H", row, col):
                    actions.append(f"WALL_H_{row}_{col}")
                if self.is_wall_legal("V", row, col):
                    actions.append(f"WALL_V_{row}_{col}")
        return actions

    def has_wall_between(self, from_row, from_col, to_row, to_col):
        if from_row == to_row:
            return ("V", from_row, min(from_col, to_col)) in self.blocked_edges
        if from_col == to_col:
            return ("H", min(from_row, to_row), from_col) in self.blocked_edges
        return True

    def move_destination(self, player, action):
        row, col = self.positions[player]
        opp_row, opp_col = self.positions[1 - player]
        dr, dc = MOVE_DELTAS[action]

        if dr != 0 and dc != 0:
            return self.side_jump_destination(row, col, opp_row, opp_col, dr, dc)

        nr = row + dr
        nc = col + dc

        if not in_bounds(nr, nc):
            return None
        if self.has_wall_between(row, col, nr, nc):
            return None
        if nr != opp_row or nc != opp_col:
            return nr, nc

        jump_row = opp_row + dr
        jump_col = opp_col + dc
        if not in_bounds(jump_row, jump_col):
            return None
        if self.has_wall_between(opp_row, opp_col, jump_row, jump_col):
            return None
        return jump_row, jump_col

    def side_jump_destination(self, row, col, opp_row, opp_col, dr, dc):
        if abs(row - opp_row) + abs(col - opp_col) != 1:
            return None

        toward_row = opp_row - row
        toward_col = opp_col - col
        if dr != toward_row and dc != toward_col:
            return None
        if self.has_wall_between(row, col, opp_row, opp_col):
            return None

        nr = row + dr
        nc = col + dc
        if not in_bounds(nr, nc):
            return None
        if self.has_wall_between(opp_row, opp_col, nr, nc):
            return None
        return nr, nc

    def is_wall_legal(self, direction, row, col):
        if not clamp_wall_origin(direction, row, col):
            return False
        if (direction, row, col) in self.walls:
            return False
        if (opposite_wall_direction(direction), row, col) in self.walls:
            return False
        edges = wall_edges(direction, row, col)
        if any(edge in self.blocked_edges for edge in edges):
            return False

        blocked = set(self.blocked_edges)
        blocked.update(edges)
        return self.has_path_to_goal(0, blocked) and self.has_path_to_goal(1, blocked)

    def has_path_to_goal(self, player, blocked_edges=None):
        if blocked_edges is None:
            blocked_edges = self.blocked_edges
        start = self.positions[player]
        goal = self.goals[player]
        queue = deque([start])
        seen = {start}

        while queue:
            row, col = queue.popleft()
            if row == goal:
                return True
            for dr, dc in MOVE_DELTAS.values():
                nr = row + dr
                nc = col + dc
                if not in_bounds(nr, nc):
                    continue
                if self.wall_between_with_edges(row, col, nr, nc, blocked_edges):
                    continue
                key = (nr, nc)
                if key not in seen:
                    seen.add(key)
                    queue.append(key)
        return False

    def shortest_distance(self, player):
        cached = self._dist_cache[player]
        if cached is not None:
            return cached
        distance, path = self.shortest_path(player)
        self._dist_cache[player] = distance
        self._path_cache[player] = path
        return distance

    def shortest_path(self, player):
        cached = self._path_cache[player]
        if cached is not None:
            return self._dist_cache[player], cached

        start = self.positions[player]
        goal = self.goals[player]
        queue = deque([start])
        parent = {start: None}

        while queue:
            row, col = queue.popleft()
            if row == goal:
                path = []
                cur = (row, col)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                distance = len(path) - 1
                self._dist_cache[player] = distance
                self._path_cache[player] = path
                return distance, path

            for nr, nc in self.path_neighbors(player, row, col):
                key = (nr, nc)
                if key in parent:
                    continue
                parent[key] = (row, col)
                queue.append(key)

        self._dist_cache[player] = INF
        self._path_cache[player] = []
        return INF, []

    def path_neighbors(self, player, row, col):
        result = []
        opp_row, opp_col = self.positions[1 - player]
        for dr, dc in CARDINAL_MOVE_DELTAS.values():
            nr = row + dr
            nc = col + dc
            if not in_bounds(nr, nc):
                continue
            if self.has_wall_between(row, col, nr, nc):
                continue
            if nr != opp_row or nc != opp_col:
                result.append((nr, nc))
                continue

            jump_row = opp_row + dr
            jump_col = opp_col + dc
            if not in_bounds(jump_row, jump_col):
                continue
            if self.has_wall_between(opp_row, opp_col, jump_row, jump_col):
                continue
            result.append((jump_row, jump_col))
        return result

    @staticmethod
    def wall_between_with_edges(from_row, from_col, to_row, to_col, blocked_edges):
        if from_row == to_row:
            return ("V", from_row, min(from_col, to_col)) in blocked_edges
        if from_col == to_col:
            return ("H", min(from_row, to_row), from_col) in blocked_edges
        return True

    def focused_wall_actions(self, actor):
        if self.walls_remaining[actor] <= 0:
            return []

        result = set()
        targets = [1 - actor, actor]
        for target in targets:
            _, path = self.shortest_path(target)
            for index in range(min(len(path) - 1, 10)):
                for action in walls_blocking_step(path[index], path[index + 1]):
                    parsed = parse_wall_action(action)
                    if parsed is None:
                        continue
                    direction, row, col = parsed
                    if self.is_wall_legal(direction, row, col):
                        result.add(action)
        return sorted(result)


def walls_blocking_step(a, b):
    result = []
    for edge in edges_for_step(a, b):
        result.extend(walls_blocking_edge(edge))
    return result


def walls_blocking_edge(edge):
    direction, row, col = edge
    result = []
    if direction == "H":
        for wall_col in (col - 1, col):
            if 0 <= row < BOARD_SIZE - 1 and 0 <= wall_col < BOARD_SIZE - 1:
                result.append(f"WALL_H_{row}_{wall_col}")
    else:
        for wall_row in (row - 1, row):
            if 0 <= wall_row < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1:
                result.append(f"WALL_V_{wall_row}_{col}")
    return result


def edges_for_step(a, b):
    ar, ac = a
    br, bc = b
    result = []
    if ac == bc and ar != br:
        step = 1 if br > ar else -1
        for row in range(ar, br, step):
            result.append(("H", min(row, row + step), ac))
    elif ar == br and ac != bc:
        step = 1 if bc > ac else -1
        for col in range(ac, bc, step):
            result.append(("V", ar, min(col, col + step)))
    return result


class Bot:
    name = "gpt5_xhigh_lqq_v2"

    def __init__(self):
        self.recent_positions = []
        self.root_actor = None

    def choose_action(self, state):
        legal = state.get("legal_actions", [])
        if not legal:
            return ""
        try:
            action = self._choose_action(state, legal)
        except Exception:
            action = self._fallback(state, legal)
        self.remember_action(state, action)
        return action

    def remember_action(self, state, action):
        if action not in MOVE_DELTAS:
            return
        root = MiniState.from_bot_state(state)
        child = root.apply(action)
        self.recent_positions.append(child.positions[root.current])
        self.recent_positions = self.recent_positions[-8:]

    def _choose_action(self, state, legal):
        quick = self.quick_opening_action(state, legal)
        if quick is not None:
            return quick

        self.deadline = time.perf_counter() + _time_budget(state, SAFETY_SECONDS)
        root = MiniState.from_bot_state(state)
        me = root.current
        self.root_actor = me

        for action in legal:
            if action in MOVE_DELTAS:
                child = root.apply(action)
                if child.winner == me:
                    return action

        urgent = self.urgent_defense_action(root, legal)
        if urgent is not None:
            return urgent

        anti_jump = self.anti_jump_action(root, legal)
        if anti_jump is not None:
            return anti_jump

        progress = self.progress_action(root, legal)
        if progress is not None:
            return progress

        closing = self.closing_move(root, legal)
        if closing is not None:
            return closing

        ranked = self.rank_actions(root, legal_actions=legal, root=True)
        if not ranked:
            return self._fallback(state, legal)
        best_action = ranked[0]

        for depth in (1, 2, 3):
            try:
                value, action = self.search_root(root, me, legal, depth)
                if action is not None:
                    best_action = action
                if abs(value) > 900000:
                    break
            except SearchTimeout:
                break
        return best_action if best_action in legal else self._fallback(state, legal)

    def urgent_defense_action(self, state, legal):
        opponent = 1 - state.current
        if not self.has_immediate_win(state, opponent):
            return None

        candidates = []
        for action in legal:
            child = state.apply(action)
            if child.winner == state.current:
                return action
            if self.has_immediate_win(child, opponent):
                continue
            candidates.append((self.score_action(state, action, state.current), action))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def anti_jump_action(self, state, legal):
        actor = state.current
        opponent = 1 - actor
        path_action = self.shortest_path_move(state, actor, legal)
        if path_action is None:
            return None
        path_child = state.apply(path_action)
        if not self.has_jump_gain(path_child, opponent):
            return None

        candidates = []
        for action in legal:
            child = state.apply(action)
            if child.winner == actor:
                return action
            if self.has_immediate_win(child, opponent) or self.has_jump_gain(child, opponent):
                continue
            candidates.append((self.score_action(state, action, actor), action))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def progress_action(self, state, legal):
        if state.turn < 20:
            return None

        actor = state.current
        opponent = 1 - actor
        before = state.shortest_distance(actor)
        candidates = []
        for action in legal:
            if action not in MOVE_DELTAS:
                continue
            child = state.apply(action)
            after = child.shortest_distance(actor)
            if after >= before:
                continue
            if self.has_immediate_win(child, opponent) or self.has_jump_gain(child, opponent):
                continue
            candidates.append((self.score_action(state, action, actor), action))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def closing_move(self, state, legal):
        actor = state.current
        opponent = 1 - actor
        my_dist = state.shortest_distance(actor)
        opp_dist = state.shortest_distance(opponent)
        my_race = 2 * my_dist - 1
        opp_race = 2 * opp_dist

        if my_dist > 5 and my_race > opp_race - 2:
            return None

        action = self.shortest_path_move(state, actor, legal)
        if action is None:
            return None

        child = state.apply(action)
        if child.shortest_distance(actor) >= my_dist:
            return None
        if my_race <= opp_race or my_dist <= 3:
            if not self.has_immediate_win(child, opponent) and not self.has_jump_gain(child, opponent):
                return action
        return None

    @staticmethod
    def has_immediate_win(state, player):
        for action in state.legal_moves(player):
            child = state.apply_as(player, action)
            if child.winner == player:
                return True
        return False

    @staticmethod
    def has_jump_gain(state, player):
        row, col = state.positions[player]
        for action in state.legal_moves(player):
            destination = state.move_destination(player, action)
            if destination is None:
                continue
            dest_row, dest_col = destination
            if abs(dest_row - row) + abs(dest_col - col) >= 2:
                return True
        return False

    @staticmethod
    def shortest_path_move(state, actor, legal):
        _, path = state.shortest_path(actor)
        if len(path) < 2:
            return None
        target = path[1]
        for action in MOVE_DELTAS:
            if action in legal and state.move_destination(actor, action) == target:
                return action
        return None

    def search_root(self, state, me, legal, depth):
        self.check_time()
        actions = self.rank_actions(state, legal_actions=legal, root=True)
        best_value = -INF
        best_action = None
        alpha = -INF
        beta = INF

        for action in actions:
            self.check_time()
            value = self.minimax(state.apply(action), depth - 1, alpha, beta, me)
            if value > best_value:
                best_value = value
                best_action = action
            if value > alpha:
                alpha = value
        return best_value, best_action

    def minimax(self, state, depth, alpha, beta, me):
        self.check_time()
        if state.winner is not None or depth <= 0:
            return self.evaluate(state, me)

        actions = self.rank_actions(state, root=False)
        if not actions:
            return 900000 if state.current != me else -900000

        if state.current == me:
            value = -INF
            for action in actions:
                value = max(value, self.minimax(state.apply(action), depth - 1, alpha, beta, me))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = INF
        for action in actions:
            value = min(value, self.minimax(state.apply(action), depth - 1, alpha, beta, me))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def rank_actions(self, state, legal_actions=None, root=False):
        actor = state.current
        if legal_actions is None:
            move_actions = state.legal_moves(actor)
            wall_actions = state.focused_wall_actions(actor)
            legal_actions = move_actions + wall_actions

        moves = []
        walls = []
        for action in legal_actions:
            if action in MOVE_DELTAS:
                moves.append(action)
            elif action.startswith("WALL_"):
                walls.append(action)

        scored_moves = [(self.score_action(state, action, actor), action) for action in moves]
        scored_walls = [(self.score_action(state, action, actor), action) for action in walls]
        scored_moves.sort(reverse=True)
        scored_walls.sort(reverse=True)

        wall_limit = 20 if root else 8
        actor_dist = state.shortest_distance(actor)
        opponent_dist = state.shortest_distance(1 - actor)
        if opponent_dist <= 2:
            wall_limit += 8
        if actor_dist <= 2 and actor_dist <= opponent_dist:
            wall_limit = max(6, wall_limit // 2)
        if actor_dist <= 5 and actor_dist <= opponent_dist:
            wall_limit = min(wall_limit, 4 if root else 2)

        ordered = [action for _, action in scored_moves]
        ordered.extend(action for _, action in scored_walls[:wall_limit])

        limit = 24 if root else 12
        return ordered[:limit]

    def score_action(self, state, action, actor):
        child = state.apply(action)
        if child.winner == actor:
            return 1000000

        score = self.evaluate(child, actor)
        if action in MOVE_DELTAS:
            before = state.shortest_distance(actor)
            after = child.shortest_distance(actor)
            opponent = 1 - actor
            opp_dist = state.shortest_distance(opponent)
            score += 42.0 * (before - after)
            row, col = child.positions[actor]
            score += 0.15 * (4 - abs(col - 4))
            if self.has_jump_gain(child, opponent):
                score -= 120.0
            if actor == self.root_actor and child.positions[actor] in self.recent_positions[-4:]:
                score -= 85.0
            if actor == self.root_actor and len(self.recent_positions) >= 2:
                if child.positions[actor] == self.recent_positions[-2]:
                    score -= 180.0
            if after >= before:
                score -= 28.0
            if before <= 5 and before <= opp_dist and after >= before:
                score -= 90.0
            path_action = self.shortest_path_move(state, actor, [action])
            if path_action == action:
                score += 14.0
            return score

        opponent = 1 - actor
        opp_gain = child.shortest_distance(opponent) - state.shortest_distance(opponent)
        self_cost = child.shortest_distance(actor) - state.shortest_distance(actor)
        score += 32.0 * opp_gain - 18.0 * self_cost - 2.2
        if state.shortest_distance(actor) <= 5 and state.shortest_distance(actor) <= state.shortest_distance(opponent):
            score -= 34.0
        if opp_gain <= 0:
            score -= 20.0
        if state.shortest_distance(opponent) <= state.shortest_distance(actor):
            score += 8.0 * max(0, opp_gain)
        score += self.wall_path_tiebreak(state, action, actor)
        return score

    def wall_path_tiebreak(self, state, action, actor):
        parsed = parse_wall_action(action)
        if parsed is None:
            return 0.0
        edges = set(wall_edges(*parsed))
        opponent = 1 - actor
        bonus = 0.0

        _, opp_path = state.shortest_path(opponent)
        for index in range(len(opp_path) - 1):
            if edges.intersection(edges_for_step(opp_path[index], opp_path[index + 1])):
                bonus += max(0.0, 5.0 - 0.45 * index)
                break

        _, own_path = state.shortest_path(actor)
        for index in range(len(own_path) - 1):
            if edges.intersection(edges_for_step(own_path[index], own_path[index + 1])):
                bonus -= max(0.0, 2.0 - 0.25 * index)
                break

        return bonus

    def evaluate(self, state, player):
        if state.winner is not None:
            if state.winner == player:
                return 1000000 - state.turn
            return -1000000 + state.turn

        opponent = 1 - player
        my_dist = state.shortest_distance(player)
        opp_dist = state.shortest_distance(opponent)
        my_race = 2 * my_dist - (1 if state.current == player else 0)
        opp_race = 2 * opp_dist - (1 if state.current == opponent else 0)

        score = 24.0 * (opp_race - my_race)
        score += 1.2 * (state.walls_remaining[player] - state.walls_remaining[opponent])

        if my_race < opp_race:
            score += 6.0
        elif my_race > opp_race:
            score -= 6.0

        if my_dist <= 2:
            score += 18.0 * (3 - my_dist)
        if opp_dist <= 2:
            score -= 18.0 * (3 - opp_dist)

        my_col = state.positions[player][1]
        opp_col = state.positions[opponent][1]
        score += 0.2 * (4 - abs(my_col - 4))
        score -= 0.1 * (4 - abs(opp_col - 4))
        return score

    def check_time(self):
        if time.perf_counter() >= self.deadline:
            raise SearchTimeout

    @staticmethod
    def quick_opening_action(state, legal):
        if state.get("walls"):
            return None
        player_id = state.get("actor", state.get("player_id", 0))
        opponent = 1 - player_id
        row = state["positions"][player_id][0]
        opp_row = state["positions"][opponent][0]
        if abs(row - opp_row) <= 2:
            return None
        goal = state["goal_rows"][player_id]
        forward = "MOVE_UP" if goal < row else "MOVE_DOWN"
        if forward in legal:
            return forward
        return None

    @staticmethod
    def _fallback(state, legal):
        player_id = state.get("player_id", state.get("actor", 0))
        row = state["positions"][player_id][0]
        goal = state["goal_rows"][player_id]
        forward = "MOVE_UP" if goal < row else "MOVE_DOWN"
        if forward in legal:
            return forward
        for action in legal:
            if action.startswith("MOVE_"):
                return action
        return legal[0]


def _time_budget(state, fallback):
    timeout = state.get("decision_timeout") or state.get("time_limit")
    if timeout:
        return max(0.05, float(timeout) - SAFETY_MARGIN_SECONDS)
    return fallback
