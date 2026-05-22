"""Threat-space Luqiangqi bot.

This bot is intentionally not a wrapper around the existing search bots.  It
uses a deterministic one-reply maximin model: every root action is judged by
the position it leaves after the opponent's strongest local reply.  The static
score values path length, route width, escape moves, and the damage of the next
available wall, so it is less eager to walk into human-style left/right fork
positions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time


BOARD_SIZE = 9
INF = 10_000
WIN_SCORE = 1_000_000
TIME_BUDGET = 0.82

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

CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MOVE_NAMES = tuple(MOVE_DELTAS)


@dataclass(frozen=True)
class Board:
    current: int
    positions: tuple[tuple[int, int], tuple[int, int]]
    goals: tuple[int, int]
    remaining: tuple[int, int]
    walls: tuple[tuple[str, int, int], ...]
    turn: int


class TimeUp(Exception):
    pass


def choose_action(state):
    return Bot().choose_action(state)


class Bot:
    name = "bot_v4_threat_space"

    def __init__(self):
        self.deadline = 0.0
        self.me = 0
        self._blocked_cache = {}
        self._path_cache = {}
        self._field_cache = {}
        self._legal_cache = {}
        self._pressure_cache = {}

    def choose_action(self, state):
        legal = list(state.get("legal_actions", ()))
        if not legal:
            return ""
        if len(legal) == 1:
            return legal[0]

        self.deadline = time.perf_counter() + TIME_BUDGET
        self.me = int(state.get("actor", state.get("player_id", 0)))
        self._pressure_cache.clear()
        node = self._from_state(state)

        win = self._winning_move(node, legal)
        if win is not None:
            return win

        urgent = self._urgent_reply(node, legal)
        if urgent is not None:
            return urgent

        try:
            action = self._choose_maximin(node, legal)
        except TimeUp:
            action = None
        if action in legal:
            return action
        return self._fallback(node, legal)

    def _choose_maximin(self, node, legal):
        candidates = self._root_candidates(node, legal)
        if not candidates:
            return self._fallback(node, legal)

        best_action = candidates[0]
        best_value = -10**18
        for action in candidates:
            self._check_time()
            child = self._apply(node, action)
            if self._winner(child) == node.current:
                return action

            static = self._evaluate(child, self.me)
            reply = self._opponent_reply_value(child, self.me)
            if reply is None:
                value = static
            else:
                value = min(static, reply + 120)
            value += self._action_bias(node, action)

            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def _root_candidates(self, node, legal):
        legal_set = set(legal)
        moves = [action for action in legal if action in MOVE_DELTAS]
        safe_moves = []
        for action in moves:
            child = self._apply(node, action)
            if self._winner(child) == node.current or not self._has_forward_jump(child, 1 - node.current):
                safe_moves.append(action)
        if safe_moves:
            moves = safe_moves
        scored = []
        for action in moves:
            scored.append((self._action_bias(node, action) + self._evaluate(self._apply(node, action), self.me) * 0.08, action))

        if node.remaining[node.current] > 0:
            legal_set = set(legal)
            focused = [action for action in self._focused_walls(node) if action in legal_set]
            for action in focused[:42]:
                scored.append((self._cheap_wall_score(node, action), action))

        scored.sort(reverse=True)
        result = []
        seen = set()
        for _, action in scored:
            if action in legal_set and action not in seen:
                seen.add(action)
                result.append(action)
            if len(result) >= 18:
                break
        return result

    def _opponent_reply_value(self, node, root_player):
        if self._winner(node) is not None:
            return None
        replies = self._reply_candidates(node)
        if not replies:
            return self._evaluate(node, root_player)

        worst = None
        for action in replies:
            self._check_time()
            child = self._apply(node, action)
            value = self._evaluate(child, root_player)
            value -= self._action_bias(node, action) * 0.35
            if worst is None or value < worst:
                worst = value
        return worst

    def _reply_candidates(self, node):
        actions = list(self._legal_moves(node, node.current))
        if node.remaining[node.current] > 0:
            wall_scores = []
            for action in self._focused_walls(node):
                wall = parse_wall(action)
                if wall is None or not self._valid_wall(node, wall):
                    continue
                wall_scores.append((self._cheap_wall_score(node, action), action))
            wall_scores.sort(reverse=True)
            actions.extend(action for _, action in wall_scores[:8])

        seen = set()
        result = []
        reverse = node.current == self.me
        actions.sort(key=lambda action: self._evaluate(self._apply(node, action), self.me), reverse=reverse)
        for action in actions:
            if action not in seen:
                seen.add(action)
                result.append(action)
            if len(result) >= 12:
                break
        return result

    def _evaluate(self, node, player):
        winner = self._winner(node)
        if winner is not None:
            return WIN_SCORE - node.turn if winner == player else -WIN_SCORE + node.turn

        opponent = 1 - player
        my_dist, _ = self._shortest_path(node, player)
        opp_dist, _ = self._shortest_path(node, opponent)
        my_race = 2 * my_dist - (1 if node.current == player else 0)
        opp_race = 2 * opp_dist - (1 if node.current == opponent else 0)

        my_width = self._route_width(node, player)
        opp_width = self._route_width(node, opponent)
        my_escape = self._escape_quality(node, player)
        opp_escape = self._escape_quality(node, opponent)
        my_pressure = self._wall_pressure(node, by=opponent, target=player)
        opp_pressure = self._wall_pressure(node, by=player, target=opponent)

        my_row, my_col = node.positions[player]
        opp_row, opp_col = node.positions[opponent]
        score = (opp_race - my_race) * 780
        score += (node.remaining[player] - node.remaining[opponent]) * 30
        score += (my_width - opp_width) * 115
        score += (my_escape - opp_escape) * 95
        score -= my_pressure * 150
        score += opp_pressure * 105
        score += (4 - abs(my_col - 4)) * 9
        score -= (4 - abs(opp_col - 4)) * 6

        if my_dist <= 3:
            score += (4 - my_dist) * 260
        if opp_dist <= 3:
            score -= (4 - opp_dist) * 320
        if abs(my_row - opp_row) + abs(my_col - opp_col) == 1:
            if self._has_forward_jump(node, player):
                score += 1450
            if self._has_forward_jump(node, opponent):
                score -= 2200
        if my_dist >= opp_dist + 3:
            score -= (my_dist - opp_dist - 2) * 240
        return score

    def _action_bias(self, node, action):
        actor = node.current
        sign = 1 if actor == self.me else -1
        child = self._apply(node, action)
        if self._winner(child) == actor:
            return sign * WIN_SCORE

        before, _ = self._shortest_path(node, actor)
        after, _ = self._shortest_path(child, actor)
        bias = sign * (before - after) * 210
        if action.startswith("WALL_"):
            bias += sign * self._wall_action_score(node, action) * 0.22
        else:
            if self._has_forward_jump(child, 1 - actor):
                bias -= sign * 1900
            if self._has_forward_jump(child, actor):
                bias += sign * 900
            bias -= sign * self._wall_pressure(child, by=1 - actor, target=actor) * 32
        return bias

    def _wall_action_score(self, node, action):
        wall = parse_wall(action)
        if wall is None:
            return -INF
        actor = node.current
        opponent = 1 - actor
        my_dist, my_path = self._shortest_path(node, actor)
        opp_dist, opp_path = self._shortest_path(node, opponent)
        child = self._apply(node, action)
        after_my, _ = self._shortest_path(child, actor)
        after_opp, _ = self._shortest_path(child, opponent)
        gain = after_opp - opp_dist
        pain = after_my - my_dist

        score = gain * 430 - pain * 380
        if gain <= 0:
            score -= 180
        if pain > gain and my_dist >= opp_dist:
            score -= (pain - gain) * 230
        if after_my >= after_opp + 5:
            score -= (after_my - after_opp - 4) * 130

        edges = set(wall_edges(wall))
        if self._wall_hits_path(edges, opp_path):
            score += 80
        if self._wall_hits_path(edges, my_path):
            score -= 95

        if opp_dist <= 3:
            score += max(0, gain) * 190
        return score if actor == self.me else -score

    def _cheap_wall_score(self, node, action):
        wall = parse_wall(action)
        if wall is None:
            return -INF
        actor = node.current
        opponent = 1 - actor
        my_dist, my_path = self._shortest_path(node, actor)
        opp_dist, opp_path = self._shortest_path(node, opponent)
        child = self._apply(node, action)
        after_my, _ = self._shortest_path(child, actor)
        after_opp, _ = self._shortest_path(child, opponent)
        gain = after_opp - opp_dist
        pain = after_my - my_dist
        score = gain * 360 - pain * 300
        edges = set(wall_edges(wall))
        if self._wall_hits_path(edges, opp_path):
            score += 70
        if self._wall_hits_path(edges, my_path):
            score -= 70
        return score if actor == self.me else -score

    def _route_width(self, node, player):
        dist, _ = self._shortest_path(node, player)
        if dist >= INF:
            return -20
        field = self._distance_field(node.walls, node.goals[player])
        row, col = node.positions[player]
        options = 0
        for nr, nc in self._path_neighbors(node.walls, row, col):
            nd = field.get((nr, nc), INF)
            if nd <= dist:
                options += 2
            elif nd == dist + 1:
                options += 1
        return options

    def _escape_quality(self, node, player):
        move_node = Board(
            current=player,
            positions=node.positions,
            goals=node.goals,
            remaining=node.remaining,
            walls=node.walls,
            turn=node.turn,
        )
        before, _ = self._shortest_path(move_node, player)
        values = []
        for action in self._legal_moves(move_node, player):
            child = self._apply(move_node, action)
            after, _ = self._shortest_path(child, player)
            pressure = self._wall_pressure(child, by=1 - player, target=player)
            values.append((before - after) * 3 - pressure)
        if not values:
            return -12
        values.sort(reverse=True)
        return values[0] + (values[1] * 0.4 if len(values) > 1 else 0)

    def _wall_pressure(self, node, by, target):
        key = ("pressure", by, target, node.positions, node.remaining, node.walls)
        cached = self._pressure_cache.get(key)
        if cached is not None:
            return cached
        if node.remaining[by] <= 0:
            self._pressure_cache[key] = 0
            return 0

        attack = Board(
            current=by,
            positions=node.positions,
            goals=node.goals,
            remaining=node.remaining,
            walls=node.walls,
            turn=node.turn,
        )
        base, _ = self._shortest_path(attack, target)
        gains = []
        for action in sorted(self._focused_walls(attack))[:14]:
            wall = parse_wall(action)
            if wall is None or not self._valid_wall(attack, wall):
                continue
            child = self._apply(attack, action)
            after, _ = self._shortest_path(child, target)
            gain = after - base
            if gain > 0:
                gains.append(gain)
        if not gains:
            value = 0
        else:
            gains.sort(reverse=True)
            value = gains[0] * 2 + (gains[1] if len(gains) > 1 else 0)
        self._pressure_cache[key] = value
        return value

    def _urgent_reply(self, node, legal):
        opponent = 1 - node.current
        if not self._has_immediate_win(node, opponent):
            return None
        safe = []
        for action in legal:
            child = self._apply(node, action)
            if self._winner(child) == node.current:
                return action
            if self._has_immediate_win(child, opponent):
                continue
            safe.append((self._evaluate(child, self.me), action))
        if not safe:
            return None
        safe.sort(reverse=True)
        return safe[0][1]

    def _focused_walls(self, node):
        result = set()
        for player in (node.current, 1 - node.current):
            _, path = self._shortest_path(node, player)
            for a, b in zip(path[:11], path[1:12]):
                result.update(self._walls_blocking_edge(a, b))
            row, col = node.positions[player]
            for nr, nc in self._path_neighbors(node.walls, row, col):
                result.update(self._walls_blocking_edge((row, col), (nr, nc)))
            goal = node.goals[player]
            fence = 2 if goal == 0 else BOARD_SIZE - 3
            for wr in (fence - 1, fence, fence + 1, row - 1, row):
                for wc in (col - 2, col - 1, col, col + 1, 1, 3, 5, 7):
                    if 0 <= wr < BOARD_SIZE - 1 and 0 <= wc < BOARD_SIZE - 1:
                        result.add(f"WALL_H_{wr}_{wc}")
        return result

    def _from_state(self, state):
        walls = tuple(
            sorted((item["dir"], int(item["row"]), int(item["col"])) for item in state.get("walls", ()))
        )
        return Board(
            current=int(state.get("actor", state.get("player_id", 0))),
            positions=tuple((int(row), int(col)) for row, col in state["positions"]),
            goals=tuple(int(goal) for goal in state.get("goal_rows", (0, BOARD_SIZE - 1))),
            remaining=tuple(int(count) for count in state.get("walls_remaining", (10, 10))),
            walls=walls,
            turn=int(state.get("turn", 0)),
        )

    def _apply(self, node, action):
        actor = node.current
        positions = [list(node.positions[0]), list(node.positions[1])]
        remaining = list(node.remaining)
        walls = list(node.walls)

        if action in MOVE_DELTAS:
            dest = self._move_destination(node, actor, action)
            if dest is not None:
                positions[actor] = [dest[0], dest[1]]
        else:
            wall = parse_wall(action)
            if wall is not None:
                walls.append(wall)
                walls.sort()
                remaining[actor] -= 1

        return Board(
            current=1 - actor,
            positions=(tuple(positions[0]), tuple(positions[1])),
            goals=node.goals,
            remaining=(remaining[0], remaining[1]),
            walls=tuple(walls),
            turn=node.turn + 1,
        )

    def _legal_actions(self, node):
        key = node
        cached = self._legal_cache.get(key)
        if cached is not None:
            return cached
        actions = list(self._legal_moves(node, node.current))
        if node.remaining[node.current] > 0:
            for row in range(BOARD_SIZE - 1):
                for col in range(BOARD_SIZE - 1):
                    for direction in ("H", "V"):
                        wall = (direction, row, col)
                        if self._valid_wall(node, wall):
                            actions.append(f"WALL_{direction}_{row}_{col}")
        self._legal_cache[key] = actions
        return actions

    def _legal_moves(self, node, player):
        return [action for action in MOVE_NAMES if self._move_destination(node, player, action) is not None]

    def _move_destination(self, node, player_id, action):
        if action not in MOVE_DELTAS:
            return None
        player = node.positions[player_id]
        opponent = node.positions[1 - player_id]
        dr, dc = MOVE_DELTAS[action]
        if dr != 0 and dc != 0:
            return self._side_jump_destination(node, player, opponent, dr, dc)

        dest = (player[0] + dr, player[1] + dc)
        if not in_bounds(*dest) or self._has_wall_between(node.walls, player, dest):
            return None
        if dest != opponent:
            return dest

        jump = (opponent[0] + dr, opponent[1] + dc)
        if not in_bounds(*jump) or self._has_wall_between(node.walls, opponent, jump):
            return None
        return jump

    def _side_jump_destination(self, node, player, opponent, dr, dc):
        if abs(player[0] - opponent[0]) + abs(player[1] - opponent[1]) != 1:
            return None
        toward = (opponent[0] - player[0], opponent[1] - player[1])
        if dr != toward[0] and dc != toward[1]:
            return None
        if self._has_wall_between(node.walls, player, opponent):
            return None
        dest = (player[0] + dr, player[1] + dc)
        if not in_bounds(*dest) or self._has_wall_between(node.walls, opponent, dest):
            return None
        return dest

    def _valid_wall(self, node, wall):
        direction, row, col = wall
        if not (0 <= row < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1):
            return False
        walls = set(node.walls)
        if wall in walls or (opposite(direction), row, col) in walls:
            return False
        blocked = self._blocked_edges(node.walls)
        if any(edge in blocked for edge in wall_edges(wall)):
            return False
        new_walls = tuple(sorted(node.walls + (wall,)))
        return (
            self._path_exists(new_walls, node.positions[0], node.goals[0])
            and self._path_exists(new_walls, node.positions[1], node.goals[1])
        )

    def _shortest_path(self, node, player):
        return self._shortest_path_raw(node.walls, node.positions[player], node.goals[player])

    def _shortest_path_raw(self, walls, start, goal):
        key = (walls, start, goal)
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached
        queue = deque([start])
        parent = {start: None}
        while queue:
            cell = queue.popleft()
            if cell[0] == goal:
                path = []
                cur = cell
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                result = (len(path) - 1, tuple(path))
                self._path_cache[key] = result
                return result
            for nxt in self._path_neighbors(walls, cell[0], cell[1]):
                if nxt not in parent:
                    parent[nxt] = cell
                    queue.append(nxt)
        result = (INF, ())
        self._path_cache[key] = result
        return result

    def _distance_field(self, walls, goal):
        key = (walls, goal)
        cached = self._field_cache.get(key)
        if cached is not None:
            return cached
        queue = deque((goal, col) for col in range(BOARD_SIZE))
        dist = {(goal, col): 0 for col in range(BOARD_SIZE)}
        while queue:
            row, col = queue.popleft()
            for nxt in self._path_neighbors(walls, row, col):
                if nxt not in dist:
                    dist[nxt] = dist[(row, col)] + 1
                    queue.append(nxt)
        self._field_cache[key] = dist
        return dist

    def _path_exists(self, walls, start, goal):
        dist, _ = self._shortest_path_raw(walls, start, goal)
        return dist < INF

    def _path_neighbors(self, walls, row, col):
        result = []
        for dr, dc in CARDINALS:
            nxt = (row + dr, col + dc)
            if in_bounds(*nxt) and not self._has_wall_between(walls, (row, col), nxt):
                result.append(nxt)
        return result

    def _has_wall_between(self, walls, a, b):
        ar, ac = a
        br, bc = b
        if ar == br:
            edge = ("V", ar, min(ac, bc))
        elif ac == bc:
            edge = ("H", min(ar, br), ac)
        else:
            return True
        return edge in self._blocked_edges(walls)

    def _blocked_edges(self, walls):
        cached = self._blocked_cache.get(walls)
        if cached is not None:
            return cached
        blocked = set()
        for wall in walls:
            blocked.update(wall_edges(wall))
        result = frozenset(blocked)
        self._blocked_cache[walls] = result
        return result

    def _walls_blocking_edge(self, a, b):
        ar, ac = a
        br, bc = b
        result = []
        if ar == br:
            col = min(ac, bc)
            for wr in (ar - 1, ar):
                if 0 <= wr < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1:
                    result.append(f"WALL_V_{wr}_{col}")
        elif ac == bc:
            row = min(ar, br)
            for wc in (ac - 1, ac):
                if 0 <= row < BOARD_SIZE - 1 and 0 <= wc < BOARD_SIZE - 1:
                    result.append(f"WALL_H_{row}_{wc}")
        return result

    def _has_immediate_win(self, node, player):
        goal = node.goals[player]
        for action in MOVE_NAMES:
            dest = self._move_destination(node, player, action)
            if dest is not None and dest[0] == goal:
                return True
        return False

    def _winning_move(self, node, legal):
        goal = node.goals[node.current]
        for action in legal:
            if action not in MOVE_DELTAS:
                continue
            dest = self._move_destination(node, node.current, action)
            if dest is not None and dest[0] == goal:
                return action
        return None

    def _winner(self, node):
        for player in (0, 1):
            if node.positions[player][0] == node.goals[player]:
                return player
        return None

    def _has_forward_jump(self, node, player):
        row, _ = node.positions[player]
        direction = "MOVE_UP" if node.goals[player] < row else "MOVE_DOWN"
        dest = self._move_destination(node, player, direction)
        return dest is not None and abs(dest[0] - row) == 2

    @staticmethod
    def _wall_hits_path(edges, path):
        for a, b in zip(path, path[1:]):
            ar, ac = a
            br, bc = b
            if ar == br:
                edge = ("V", ar, min(ac, bc))
            elif ac == bc:
                edge = ("H", min(ar, br), ac)
            else:
                continue
            if edge in edges:
                return True
        return False

    def _fallback(self, node, legal):
        _, path = self._shortest_path(node, node.current)
        if len(path) >= 2:
            target = path[1]
            for action in MOVE_NAMES:
                if action in legal and self._move_destination(node, node.current, action) == target:
                    child = self._apply(node, action)
                    if not self._has_forward_jump(child, 1 - node.current):
                        return action
        safe = []
        for action in legal:
            if action in MOVE_DELTAS:
                child = self._apply(node, action)
                if not self._has_forward_jump(child, 1 - node.current):
                    safe.append(action)
        if safe:
            before, _ = self._shortest_path(node, node.current)
            return min(safe, key=lambda action: self._shortest_path(self._apply(node, action), 1 - self._apply(node, action).current)[0] - before)
        for action in legal:
            if action in MOVE_DELTAS:
                return action
        return legal[0]

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise TimeUp


def parse_wall(action):
    parts = action.split("_")
    if len(parts) != 4 or parts[0] != "WALL" or parts[1] not in {"H", "V"}:
        return None
    try:
        return parts[1], int(parts[2]), int(parts[3])
    except ValueError:
        return None


def wall_edges(wall):
    direction, row, col = wall
    if direction == "H":
        return (("H", row, col), ("H", row, col + 1))
    return (("V", row, col), ("V", row + 1, col))


def opposite(direction):
    return "V" if direction == "H" else "H"


def in_bounds(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
