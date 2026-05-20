"""Search-based Luqiangqi bot.

This bot is self contained: it rebuilds the public rules from the state shape,
then uses shortest-path pressure and a shallow alpha-beta search with a hard
time budget.  The branch factor in wall games is large, so wall moves are
filtered to the ones that interact with the current shortest paths before
searching.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time


BOARD_SIZE = 9
INF = 10_000
WIN_SCORE = 1_000_000
TIME_BUDGET = 0.55

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

CARDINALS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

MOVE_NAMES = tuple(MOVE_DELTAS)


@dataclass(frozen=True)
class Node:
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
    name = "gpt_hard"

    def __init__(self):
        self.me = 0
        self.deadline = 0.0
        self._blocked_cache = {}
        self._path_cache = {}
        self._legal_cache = {}
        self._eval_cache = {}
        self._seen = {}

    def choose_action(self, state):
        legal = list(state["legal_actions"])
        if not legal:
            return ""
        if len(legal) == 1:
            return legal[0]

        self.me = int(state.get("player_id", state.get("actor", 0)))
        self.deadline = time.perf_counter() + TIME_BUDGET
        self._eval_cache.clear()

        node = self._node_from_state(state)
        self._seen[self._seen_key(node)] = self._seen.get(self._seen_key(node), 0) + 1

        instant = self._winning_move(node, legal)
        if instant is not None:
            return instant

        book = self._book_action(node, legal)
        if book is not None:
            return book

        if node.remaining[node.current] == 0:
            move = self._best_path_move(node, legal)
            if move is not None:
                return move

        candidates = self._candidate_actions(node, legal, root=True)
        if not candidates:
            return legal[0]

        best_action = max(candidates, key=lambda a: self._static_after(node, a))
        best_value = -math.inf

        try:
            for depth in (1, 2, 3, 4):
                self._check_time()
                depth_best = best_action
                depth_value = -math.inf
                alpha = -math.inf
                beta = math.inf
                ordered = sorted(
                    candidates,
                    key=lambda a: self._static_after(node, a),
                    reverse=True,
                )
                for action in ordered:
                    self._check_time()
                    child = self._apply(node, action)
                    value = self._search(child, depth - 1, alpha, beta)
                    if value > depth_value:
                        depth_value = value
                        depth_best = action
                    alpha = max(alpha, depth_value)
                best_action = depth_best
                best_value = depth_value
                if abs(best_value) > WIN_SCORE // 2:
                    break
        except TimeUp:
            pass

        if best_action in legal:
            return best_action
        return max(legal, key=lambda a: self._static_after(node, a))

    def _node_from_state(self, state):
        positions = tuple(tuple(p) for p in state["positions"])
        goals = tuple(state.get("goal_rows", (0, BOARD_SIZE - 1)))
        remaining = tuple(state["walls_remaining"])
        walls = tuple(
            sorted((w["dir"], int(w["row"]), int(w["col"])) for w in state["walls"])
        )
        return Node(
            current=int(state.get("actor", state.get("player_id", 0))),
            positions=positions,
            goals=goals,
            remaining=remaining,
            walls=walls,
            turn=int(state.get("turn", 0)),
        )

    def _search(self, node, depth, alpha, beta):
        self._check_time()
        winner = self._winner(node)
        if winner is not None:
            return WIN_SCORE - node.turn if winner == self.me else -WIN_SCORE + node.turn
        if depth <= 0:
            return self._evaluate(node)

        legal = self._legal_actions(node)
        if not legal:
            return -WIN_SCORE + node.turn if node.current == self.me else WIN_SCORE - node.turn

        maximizing = node.current == self.me
        actions = self._candidate_actions(node, legal, root=False)
        if maximizing:
            value = -math.inf
            ordered = sorted(actions, key=lambda a: self._static_after(node, a), reverse=True)
            for action in ordered:
                value = max(value, self._search(self._apply(node, action), depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = math.inf
            ordered = sorted(actions, key=lambda a: self._static_after(node, a))
            for action in ordered:
                value = min(value, self._search(self._apply(node, action), depth - 1, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break

        return value

    def _evaluate(self, node):
        winner = self._winner(node)
        if winner is not None:
            return WIN_SCORE - node.turn if winner == self.me else -WIN_SCORE + node.turn

        opp = 1 - self.me
        my_dist, my_path = self._shortest_path(node, self.me)
        opp_dist, opp_path = self._shortest_path(node, opp)
        if my_dist >= INF:
            return -WIN_SCORE // 2
        if opp_dist >= INF:
            return WIN_SCORE // 2

        my_arrival = my_dist * 2 - (1 if node.current == self.me else 0)
        opp_arrival = opp_dist * 2 - (1 if node.current == opp else 0)
        race = opp_arrival - my_arrival

        my_row, my_col = node.positions[self.me]
        opp_row, opp_col = node.positions[opp]
        my_forward = -1 if node.goals[self.me] < my_row else 1
        opp_forward = -1 if node.goals[opp] < opp_row else 1

        score = race * 950
        score += (node.remaining[self.me] - node.remaining[opp]) * 38
        score += (4 - abs(my_col - 4)) * 9
        score -= (4 - abs(opp_col - 4)) * 7

        if my_path:
            score -= self._path_crowding(node, my_path) * 22
        if opp_path:
            score += self._path_crowding(node, opp_path) * 14

        if abs(my_row - opp_row) + abs(my_col - opp_col) == 1:
            if (opp_row - my_row) == my_forward:
                score += 120
            if (my_row - opp_row) == opp_forward:
                score -= 120

        if self._has_forward_jump(node, self.me):
            score += 720 if node.current == self.me else 260
        if self._has_forward_jump(node, opp):
            score -= 720 if node.current == opp else 260

        if race > 0:
            score += 70
        elif race < 0:
            score -= 70

        return score

    def _path_crowding(self, node, path):
        walls = set(node.walls)
        total = 0
        for row, col in path[:5]:
            for wall in self._walls_touching_cell(row, col):
                if wall in walls:
                    total += 1
        return total

    def _static_after(self, node, action):
        child = self._apply(node, action)
        value = self._evaluate(child)
        repeat = self._seen.get(self._seen_key(child), 0)
        if repeat:
            value -= repeat * (900 if node.current == self.me else -450)
        if action.startswith("WALL_"):
            value += self._wall_bonus(node, action)
        else:
            value += self._move_bonus(node, action)
        return value

    def _move_bonus(self, node, action):
        actor = node.current
        sign = 1 if actor == self.me else -1
        before, _ = self._shortest_path(node, actor)
        child = self._apply(node, action)
        after, _ = self._shortest_path(child, actor)
        bonus = sign * (before - after) * 160
        if action.startswith("MOVE_"):
            opponent = 1 - actor
            if self._has_forward_jump(child, opponent):
                bonus -= sign * 520
            if self._has_forward_jump(child, actor):
                bonus += sign * 180
        return bonus

    def _wall_bonus(self, node, action):
        actor = node.current
        target = 1 - actor
        before_target, _ = self._shortest_path(node, target)
        before_actor, _ = self._shortest_path(node, actor)
        child = self._apply(node, action)
        after_target, _ = self._shortest_path(child, target)
        after_actor, _ = self._shortest_path(child, actor)
        delta = (after_target - before_target) * 260 - (after_actor - before_actor) * 190
        return delta if actor == self.me else -delta

    def _candidate_actions(self, node, legal, root):
        legal = list(legal)
        legal_set = set(legal)
        actions = []
        for action in legal:
            if action.startswith("MOVE_"):
                actions.append(action)

        win = self._winning_move(node, legal)
        if win is not None and win not in actions:
            actions.insert(0, win)

        wall_actions = [a for a in legal if a.startswith("WALL_")]
        if wall_actions:
            _, target_path = self._shortest_path(node, 1 - node.current)
            target_dist = len(target_path) - 1 if target_path else INF
            tactical = self._tactical_wall_set(node)
            if root:
                tactical.update(self._fence_wall_set(node, 1 - node.current))
                tactical.update(self._fence_wall_set(node, node.current))
                scored = [
                    (self._raw_wall_score(node, a), a)
                    for a in tactical
                    if a in legal_set
                ]
                if len(scored) < 10:
                    scored.extend(
                        (self._raw_wall_score(node, a), a)
                        for a in wall_actions[:24]
                    )
                scored.sort(reverse=True)
                wall_limit = 20
                if node.remaining[node.current] <= 2 and target_dist > 3:
                    actions.extend(a for score, a in scored[:wall_limit] if score > 260)
                else:
                    actions.extend(a for score, a in scored[:wall_limit] if score > -900)
            else:
                tactical.update(self._fence_wall_set(node, 1 - node.current))
                scored = [
                    (self._raw_wall_score(node, a), a)
                    for a in tactical
                    if a in legal_set
                ]
                if len(scored) < 8:
                    extra = [(self._raw_wall_score(node, a), a) for a in wall_actions[:40]]
                    scored.extend(extra)
                scored.sort(reverse=True)
                wall_limit = 12 if node.current == self.me else 16
                if node.remaining[node.current] <= 2 and target_dist > 3:
                    actions.extend(a for score, a in scored[:wall_limit] if score > 260)
                else:
                    actions.extend(a for score, a in scored[:wall_limit] if score > -700)

        seen = set()
        unique = []
        for action in actions:
            if action in legal_set and action not in seen:
                seen.add(action)
                unique.append(action)

        limit = 30 if root else 20
        return unique[:limit] if len(unique) > limit else unique

    def _book_action(self, node, legal):
        if node.remaining[node.current] <= 0 or node.turn < 6 or node.turn > 30:
            return None

        actor = node.current
        if actor == 1:
            return None
        if actor == 0 and node.remaining[actor] <= 2:
            return None

        target = 1 - actor
        actor_row, _ = node.positions[actor]
        if actor == 0 and actor_row > 5:
            return None
        if actor == 1 and actor_row < 3:
            return None

        my_dist, _ = self._shortest_path(node, actor)
        target_dist, _ = self._shortest_path(node, target)
        if my_dist <= 2:
            return None

        if actor == 0:
            used = 10 - node.remaining[actor]
            if used >= 1 and ("H", 1, 3) not in set(node.walls):
                return None
            sequence = [
                "WALL_H_6_3",
                "WALL_H_6_5",
                "WALL_H_6_1",
                "WALL_H_5_4",
                "WALL_H_6_7",
                "WALL_H_5_2",
                "WALL_H_5_6",
                "WALL_V_5_0",
            ]
        else:
            sequence = [
                "WALL_H_1_3",
                "WALL_H_1_5",
                "WALL_H_1_1",
                "WALL_H_1_7",
                "WALL_V_4_3",
                "WALL_V_2_3",
                "WALL_V_4_5",
                "WALL_V_3_4",
                "WALL_H_2_5",
                "WALL_H_2_7",
            ]

        legal_set = set(legal)
        for action in sequence:
            if action in legal_set:
                score = self._raw_wall_score(node, action)
                if target_dist <= my_dist + 2 or score > -180:
                    return action
        return None

    def _best_path_move(self, node, legal):
        moves = [action for action in legal if action.startswith("MOVE_")]
        if not moves:
            return None
        actor = node.current
        target = 1 - actor
        before_dist, _ = self._shortest_path(node, actor)
        goal = node.goals[actor]
        row, col = node.positions[actor]

        def score(action):
            child = self._apply(node, action)
            after_dist, _ = self._shortest_path(child, actor)
            robust_dist = self._worst_distance_after_current_wall(child, actor)
            target_dist, _ = self._shortest_path(child, target)
            new_row, new_col = child.positions[actor]
            progress = abs(goal - row) - abs(goal - new_row)
            value = (before_dist - after_dist) * 420
            value -= (robust_dist - after_dist) * 520
            if self._opponent_can_make_corridor_trap(child, actor):
                value -= 900
            value += progress * 760
            value += target_dist * 18
            value += (4 - abs(new_col - 4)) * 4
            if self._has_forward_jump(child, target):
                value -= 700
            if self._winner(child) == actor:
                value += WIN_SCORE
            value -= self._seen.get(self._seen_key(child), 0) * 1200
            return value

        return max(moves, key=score)

    @staticmethod
    def _seen_key(node):
        return (node.current, node.positions, node.remaining, node.walls)

    def _worst_distance_after_current_wall(self, node, target):
        base, _ = self._shortest_path(node, target)
        if node.remaining[node.current] <= 0:
            return base
        legal_set = set(self._legal_actions(node))
        wall_actions = self._tactical_wall_set(node)
        wall_actions.update(self._fence_wall_set(node, target))
        worst = base
        scored = []
        for action in wall_actions:
            if action not in legal_set:
                continue
            child = self._apply(node, action)
            dist, _ = self._shortest_path(child, target)
            scored.append((dist, action))
        scored.sort(reverse=True)
        for dist, _ in scored[:10]:
            if dist > worst:
                worst = dist
        return worst

    def _opponent_can_make_corridor_trap(self, node, target):
        if node.remaining[node.current] <= 0:
            return False
        legal_set = set(self._legal_actions(node))
        wall_actions = self._tactical_wall_set(node)
        wall_actions.update(self._fence_wall_set(node, target))
        for action in wall_actions:
            if action not in legal_set:
                continue
            child = self._apply(node, action)
            if self._winner(child) is not None:
                continue
            moves = [a for a in self._legal_actions(child) if a.startswith("MOVE_")]
            dist, _ = self._shortest_path(child, target)
            if len(moves) <= 1 and dist > 2:
                return True
        return False

    def _raw_wall_score(self, node, action):
        if not action.startswith("WALL_"):
            return -INF
        actor = node.current
        target = 1 - actor
        before_target, target_path = self._shortest_path(node, target)
        before_actor, actor_path = self._shortest_path(node, actor)
        child = self._apply(node, action)
        after_target, _ = self._shortest_path(child, target)
        after_actor, _ = self._shortest_path(child, actor)

        gain = after_target - before_target
        pain = after_actor - before_actor
        score = gain * 145 - pain * 78
        if before_target <= 3:
            score += gain * 80
        if before_actor <= 3:
            score -= pain * 120

        wall = parse_wall(action)
        if wall is not None:
            touched_target = self._wall_hits_path(wall, target_path)
            touched_actor = self._wall_hits_path(wall, actor_path)
            score += 30 if touched_target else 0
            score -= 18 if touched_actor else 0
            tr, tc = node.positions[target]
            direction, wr, wc = wall
            score += max(0, 5 - abs(wr - tr) - abs(wc - tc)) * 4
            target_fence = 2 if node.goals[target] == 0 else BOARD_SIZE - 3
            actor_fence = 2 if node.goals[actor] == 0 else BOARD_SIZE - 3
            if direction == "H" and abs(wr - target_fence) <= 1:
                score += max(0, 5 - abs((wc + 1) - tc)) * 18
            if direction == "H" and abs(wr - actor_fence) <= 1 and touched_actor:
                score -= 35

        if node.current != self.me:
            score = -score
        return score

    def _has_forward_jump(self, node, player):
        row, _ = node.positions[player]
        direction = "MOVE_UP" if node.goals[player] < row else "MOVE_DOWN"
        dest = self._move_destination(node, player, direction)
        if dest is None:
            return False
        return abs(dest[0] - row) == 2 and abs(node.goals[player] - dest[0]) < abs(node.goals[player] - row)

    def _tactical_wall_set(self, node):
        result = set()
        for player in (node.current, 1 - node.current):
            _, path = self._shortest_path(node, player)
            for a, b in zip(path, path[1:]):
                result.update(self._walls_blocking_edge(a, b))
            row, col = node.positions[player]
            for nr, nc in self._neighbors_for_path(node.walls, row, col):
                result.update(self._walls_blocking_edge((row, col), (nr, nc)))
        return result

    def _fence_wall_set(self, node, target):
        row, col = node.positions[target]
        goal = node.goals[target]
        fence_row = 2 if goal == 0 else BOARD_SIZE - 3
        rows = (fence_row, fence_row - 1, fence_row + 1, row - 1, row)
        cols = (col - 2, col - 1, col, col + 1, 1, 3, 5, 7, 0, 2, 4, 6)
        result = set()
        for wr in rows:
            for wc in cols:
                if 0 <= wr < BOARD_SIZE - 1 and 0 <= wc < BOARD_SIZE - 1:
                    result.add(f"WALL_H_{wr}_{wc}")
        return result

    def _winning_move(self, node, legal):
        goal = node.goals[node.current]
        for action in legal:
            if not action.startswith("MOVE_"):
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

    def _legal_actions(self, node):
        key = node
        cached = self._legal_cache.get(key)
        if cached is not None:
            return cached

        actions = []
        for action in MOVE_NAMES:
            if self._move_destination(node, node.current, action) is not None:
                actions.append(action)

        if node.remaining[node.current] > 0:
            for row in range(BOARD_SIZE - 1):
                for col in range(BOARD_SIZE - 1):
                    for direction in ("H", "V"):
                        wall = (direction, row, col)
                        if self._valid_wall(node, wall):
                            actions.append(f"WALL_{direction}_{row}_{col}")

        self._legal_cache[key] = actions
        return actions

    def _move_destination(self, node, player_id, action):
        if action not in MOVE_DELTAS:
            return None
        player = node.positions[player_id]
        opponent = node.positions[1 - player_id]
        dr, dc = MOVE_DELTAS[action]

        if dr != 0 and dc != 0:
            return self._side_jump_destination(node, player, opponent, dr, dc)

        row = player[0] + dr
        col = player[1] + dc
        if not in_bounds(row, col):
            return None
        if self._has_wall_between(node.walls, player, (row, col)):
            return None
        if (row, col) != opponent:
            return row, col

        jump = (opponent[0] + dr, opponent[1] + dc)
        if not in_bounds(*jump):
            return None
        if self._has_wall_between(node.walls, opponent, jump):
            return None
        return jump

    def _side_jump_destination(self, node, player, opponent, dr, dc):
        if abs(player[0] - opponent[0]) + abs(player[1] - opponent[1]) != 1:
            return None
        toward_row = opponent[0] - player[0]
        toward_col = opponent[1] - player[1]
        if dr != toward_row and dc != toward_col:
            return None
        if self._has_wall_between(node.walls, player, opponent):
            return None
        dest = (player[0] + dr, player[1] + dc)
        if not in_bounds(*dest):
            return None
        if self._has_wall_between(node.walls, opponent, dest):
            return None
        return dest

    def _apply(self, node, action):
        current = node.current
        positions = [list(node.positions[0]), list(node.positions[1])]
        remaining = list(node.remaining)
        walls = list(node.walls)

        if action.startswith("MOVE_"):
            dest = self._move_destination(node, current, action)
            if dest is None:
                return node
            positions[current] = [dest[0], dest[1]]
        else:
            wall = parse_wall(action)
            if wall is not None:
                walls.append(wall)
                walls.sort()
                remaining[current] -= 1

        return Node(
            current=1 - current,
            positions=(tuple(positions[0]), tuple(positions[1])),
            goals=node.goals,
            remaining=(remaining[0], remaining[1]),
            walls=tuple(walls),
            turn=node.turn + 1,
        )

    def _valid_wall(self, node, wall):
        direction, row, col = wall
        if not (0 <= row < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1):
            return False
        walls = set(node.walls)
        if wall in walls:
            return False
        if (opposite(direction), row, col) in walls:
            return False

        blocked = self._blocked_edges(node.walls)
        for edge in wall_edges(wall):
            if edge in blocked:
                return False

        new_walls = tuple(sorted(node.walls + (wall,)))
        return (
            self._path_exists(new_walls, node.positions[0], node.goals[0])
            and self._path_exists(new_walls, node.positions[1], node.goals[1])
        )

    def _shortest_path(self, node, player):
        return self._shortest_path_raw(node.walls, node.positions[player], node.goals[player])

    def _shortest_path_raw(self, walls, start, goal_row):
        key = (walls, start, goal_row)
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached

        queue = deque([start])
        parents = {start: None}
        while queue:
            cell = queue.popleft()
            if cell[0] == goal_row:
                path = []
                cur = cell
                while cur is not None:
                    path.append(cur)
                    cur = parents[cur]
                path.reverse()
                result = (len(path) - 1, tuple(path))
                self._path_cache[key] = result
                return result
            for nxt in self._neighbors_for_path(walls, cell[0], cell[1]):
                if nxt not in parents:
                    parents[nxt] = cell
                    queue.append(nxt)

        result = (INF, ())
        self._path_cache[key] = result
        return result

    def _path_exists(self, walls, start, goal_row):
        dist, _ = self._shortest_path_raw(walls, start, goal_row)
        return dist < INF

    def _neighbors_for_path(self, walls, row, col):
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
        frozen = frozenset(blocked)
        self._blocked_cache[walls] = frozen
        return frozen

    def _wall_hits_path(self, wall, path):
        if len(path) < 2:
            return False
        edges = set(wall_edges(wall))
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

    def _walls_blocking_edge(self, a, b):
        ar, ac = a
        br, bc = b
        result = []
        if ar == br:
            row = ar
            col = min(ac, bc)
            for wr in (row - 1, row):
                wall = ("V", wr, col)
                if 0 <= wr < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1:
                    result.append(f"WALL_V_{wr}_{col}")
        elif ac == bc:
            row = min(ar, br)
            col = ac
            for wc in (col - 1, col):
                wall = ("H", row, wc)
                if 0 <= row < BOARD_SIZE - 1 and 0 <= wc < BOARD_SIZE - 1:
                    result.append(f"WALL_H_{row}_{wc}")
        return result

    def _walls_touching_cell(self, row, col):
        result = []
        for wr in (row - 1, row):
            for wc in (col - 1, col):
                if 0 <= wr < BOARD_SIZE - 1 and 0 <= wc < BOARD_SIZE - 1:
                    result.append(("H", wr, wc))
                    result.append(("V", wr, wc))
        return result

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise TimeUp()


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
