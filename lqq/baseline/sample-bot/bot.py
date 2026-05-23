from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
import time


BOARD_SIZE = 9
INF = 10**9
WIN_SCORE = 1_000_000

# Keep a real margin under a 1s referee timeout.  The environment runs the bot
# in a worker thread, so returning at ~0.84s leaves room for scheduling jitter.
TIME_BUDGET = 0.62
SAFETY_MARGIN_SECONDS = 0.15
EXPLORATION = 1.15
ROLLOUT_PLIES = 5

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


def _time_budget(state, fallback):
    timeout = state.get("decision_timeout") or state.get("time_limit")
    if timeout:
        return max(0.05, float(timeout) - SAFETY_MARGIN_SECONDS)
    return fallback


class TimeUp(Exception):
    pass


@dataclass(frozen=True)
class Board:
    current: int
    positions: tuple[tuple[int, int], tuple[int, int]]
    goals: tuple[int, int]
    remaining: tuple[int, int]
    walls: tuple[tuple[str, int, int], ...]
    turn: int


class TreeNode:
    __slots__ = (
        "state",
        "parent",
        "action",
        "children",
        "untried",
        "visits",
        "total",
        "prior",
    )

    def __init__(self, state, parent=None, action=None, untried=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.untried = list(untried or ())
        self.visits = 0
        self.total = 0.0
        self.prior = prior


def choose_action(state):
    return Bot().choose_action(state)


class Bot:
    name = "mcts_lqq_1s"

    def __init__(self):
        self.me = 0
        self.deadline = 0.0
        self.rng = random.Random(20260520)
        self._blocked_cache = {}
        self._path_cache = {}
        self._eval_cache = {}
        self._state_seen = {}
        self._recent_own_positions = []

    def choose_action(self, state):
        legal = list(state.get("legal_actions", ()))
        if not legal:
            return ""
        if len(legal) == 1:
            return legal[0]

        self.deadline = time.perf_counter() + _time_budget(state, TIME_BUDGET)
        self.me = int(state.get("player_id", state.get("actor", 0)))
        self._eval_cache.clear()
        if len(self._path_cache) > 50000:
            self._path_cache.clear()
            self._blocked_cache.clear()

        root = self._board_from_state(state)
        self._remember_seen(root)

        try:
            action = self._choose_action(root, legal, state)
        except Exception:
            action = self._fallback(state, legal)

        self._remember_move(root, action)
        return action if action in legal else self._fallback(state, legal)

    def _choose_action(self, root, legal, raw_state):
        instant = self._winning_move(root, legal)
        if instant is not None:
            return instant

        urgent = self._urgent_defense(root, legal)
        if urgent is not None:
            return urgent

        opening = self._opening_action(raw_state, legal)
        if opening is not None:
            return opening

        closing = self._closing_move(root, legal)
        if closing is not None:
            return closing

        progress = self._progress_action(root, legal)
        if progress is not None:
            return progress

        candidates = self._candidate_actions(root, legal=legal, root=True)
        if not candidates:
            return legal[0]

        static_best = candidates[0]
        if root.remaining[root.current] == 0:
            path_move = self._best_path_move(root, legal)
            if path_move is not None:
                static_best = path_move

        try:
            best = self._mcts(root, candidates, static_best)
        except TimeUp:
            best = static_best

        return best if best in legal else static_best

    def _mcts(self, root, root_actions, fallback):
        root_node = TreeNode(root, untried=root_actions)
        best = fallback
        iterations = 0

        try:
            while True:
                self._check_time()
                node = root_node

                while not node.untried and node.children:
                    node = self._select_child(node)

                if node.untried:
                    action = node.untried.pop(0)
                    child_state = self._apply(node.state, action)
                    child_actions = self._candidate_actions(child_state, root=False)
                    prior = self._prior_value(node.state, action)
                    child = TreeNode(
                        child_state,
                        parent=node,
                        action=action,
                        untried=child_actions,
                        prior=prior,
                    )
                    node.children.append(child)
                    node = child

                value = self._rollout_value(node.state)
                self._backup(node, value)
                iterations += 1

                if root_node.children:
                    best = self._best_root_action(root_node, fallback)

                if iterations >= 40 and time.perf_counter() + 0.006 >= self.deadline:
                    break
        except TimeUp:
            pass

        return best

    def _select_child(self, node):
        log_parent = math.log(node.visits + 1.0)
        actor_is_me = node.state.current == self.me
        best_child = node.children[0]
        best_score = -INF

        for child in node.children:
            if child.visits <= 0:
                score = INF
            else:
                mean = child.total / child.visits
                exploit = mean if actor_is_me else -mean
                explore = EXPLORATION * math.sqrt(log_parent / child.visits)
                prior = 0.04 * child.prior if actor_is_me else -0.04 * child.prior
                score = exploit + explore + prior
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _backup(self, node, value):
        while node is not None:
            node.visits += 1
            node.total += value
            node = node.parent

    def _best_root_action(self, root_node, fallback):
        best_action = fallback
        best_score = -INF
        total_visits = max(1, root_node.visits)

        for child in root_node.children:
            if child.visits <= 0:
                continue
            mean = child.total / child.visits
            confidence = math.sqrt(child.visits / total_visits) * 0.08
            score = mean + confidence + 0.025 * child.prior
            if score > best_score:
                best_score = score
                best_action = child.action
        return best_action

    def _rollout_value(self, state):
        node = state
        for _ in range(ROLLOUT_PLIES):
            winner = self._winner(node)
            if winner is not None:
                return 1.0 if winner == self.me else -1.0

            self._soft_check_time()
            actor = node.current
            actions = self._candidate_actions(node, root=False, rollout=True)
            if not actions:
                return -1.0 if actor == self.me else 1.0

            win = self._winning_move(node, actions)
            if win is not None:
                action = win
            else:
                action = self._rollout_pick(actions)
            node = self._apply(node, action)

        return math.tanh(self._evaluate(node) / 3600.0)

    def _rollout_pick(self, actions):
        if len(actions) == 1:
            return actions[0]
        roll = self.rng.random()
        if roll < 0.64:
            index = 0
        elif roll < 0.84:
            index = 1
        elif roll < 0.94:
            index = 2
        else:
            index = self.rng.randrange(min(len(actions), 5))
        return actions[min(index, len(actions) - 1)]

    def _board_from_state(self, state):
        walls = tuple(
            sorted(
                (item["dir"], int(item["row"]), int(item["col"]))
                for item in state.get("walls", ())
            )
        )
        return Board(
            current=int(state.get("actor", state.get("player_id", 0))),
            positions=tuple((int(row), int(col)) for row, col in state["positions"]),
            goals=tuple(int(goal) for goal in state.get("goal_rows", (0, BOARD_SIZE - 1))),
            remaining=tuple(int(count) for count in state.get("walls_remaining", (10, 10))),
            walls=walls,
            turn=int(state.get("turn", 0)),
        )

    def _candidate_actions(self, node, legal=None, root=False, rollout=False):
        actor = node.current
        legal_set = set(legal) if legal is not None else None

        if legal is None:
            moves = self._legal_moves(node, actor)
        else:
            moves = [action for action in legal if action in MOVE_DELTAS]

        actions = list(moves)

        if node.remaining[actor] > 0:
            wall_pool = self._focused_wall_set(node)
            if legal is not None:
                wall_pool = [action for action in wall_pool if action in legal_set]
                if root:
                    legal_walls = [action for action in legal if action.startswith("WALL_")]
                    if len(wall_pool) < 8:
                        wall_pool.extend(legal_walls[:36])
            else:
                wall_pool = [
                    action
                    for action in wall_pool
                    if self._valid_wall_action(node, action)
                ]

            scored_walls = [(self._raw_wall_score(node, action), action) for action in wall_pool]
            scored_walls.sort(reverse=(actor == self.me))

            actor_dist, _ = self._shortest_path(node, actor)
            opponent_dist, _ = self._shortest_path(node, 1 - actor)
            if rollout:
                wall_limit = 4
            elif root:
                wall_limit = 18
            else:
                wall_limit = 9
            if opponent_dist <= 3:
                wall_limit += 5
            if actor_dist <= 4 and actor_dist <= opponent_dist:
                wall_limit = min(wall_limit, 3 if root else 2)
            elif actor_dist <= 6 and actor_dist <= opponent_dist + 1:
                wall_limit = min(wall_limit, 6 if root else 3)
            if node.remaining[actor] <= 2 and actor_dist <= opponent_dist + 2:
                wall_limit = min(wall_limit, 2)

            for score, action in scored_walls[:wall_limit]:
                if rollout and ((actor == self.me and score < -450) or (actor != self.me and score > 450)):
                    continue
                actions.append(action)

        seen = set()
        unique = []
        for action in actions:
            if action in seen:
                continue
            if legal_set is not None and action not in legal_set:
                continue
            seen.add(action)
            unique.append(action)

        reverse = actor == self.me
        unique.sort(key=lambda action: self._static_after(node, action), reverse=reverse)

        if rollout:
            limit = 8
        elif root:
            limit = 24
        else:
            limit = 13
        return unique[:limit]

    def _focused_wall_set(self, node):
        result = set()
        for player in (1 - node.current, node.current):
            _, path = self._shortest_path(node, player)
            for a, b in zip(path[:10], path[1:11]):
                result.update(self._walls_blocking_step(a, b))

            row, col = node.positions[player]
            for nr, nc in self._neighbors_for_path(node.walls, row, col):
                result.update(self._walls_blocking_step((row, col), (nr, nc)))

            goal = node.goals[player]
            fence = 2 if goal == 0 else BOARD_SIZE - 3
            for wr in (fence - 1, fence, fence + 1, row - 1, row):
                for wc in (col - 2, col - 1, col, col + 1, 1, 3, 5, 7):
                    if 0 <= wr < BOARD_SIZE - 1 and 0 <= wc < BOARD_SIZE - 1:
                        result.add(f"WALL_H_{wr}_{wc}")
        return list(result)

    def _static_after(self, node, action):
        child = self._apply(node, action)
        value = self._evaluate(child)

        repeat = self._state_seen.get(self._seen_key(child), 0)
        if repeat:
            value -= repeat * (0.035 * WIN_SCORE if node.current == self.me else -0.018 * WIN_SCORE)

        if action.startswith("WALL_"):
            value += self._wall_bonus(node, action)
        else:
            value += self._move_bonus(node, action)
        return value

    def _prior_value(self, node, action):
        return math.tanh(self._static_after(node, action) / 4200.0)

    def _move_bonus(self, node, action):
        actor = node.current
        sign = 1 if actor == self.me else -1
        before, _ = self._shortest_path(node, actor)
        child = self._apply(node, action)
        after, _ = self._shortest_path(child, actor)
        bonus = sign * (before - after) * 210

        opponent = 1 - actor
        if self._has_forward_jump(child, opponent):
            bonus -= sign * 540
        if self._has_forward_jump(child, actor):
            bonus += sign * 260
        if self._winner(child) == actor:
            bonus += sign * WIN_SCORE
        return bonus

    def _wall_bonus(self, node, action):
        actor = node.current
        target = 1 - actor
        before_target, _ = self._shortest_path(node, target)
        before_actor, _ = self._shortest_path(node, actor)
        child = self._apply(node, action)
        after_target, _ = self._shortest_path(child, target)
        after_actor, _ = self._shortest_path(child, actor)
        delta = (after_target - before_target) * 210 - (after_actor - before_actor) * 190
        if before_target <= 3:
            delta += (after_target - before_target) * 120
        if before_actor <= 3:
            delta -= (after_actor - before_actor) * 260
        return delta if actor == self.me else -delta

    def _raw_wall_score(self, node, action):
        parsed = parse_wall(action)
        if parsed is None:
            return -INF if node.current == self.me else INF

        actor = node.current
        target = 1 - actor
        before_target, target_path = self._shortest_path(node, target)
        before_actor, actor_path = self._shortest_path(node, actor)
        child = self._apply(node, action)
        after_target, _ = self._shortest_path(child, target)
        after_actor, _ = self._shortest_path(child, actor)

        gain = after_target - before_target
        pain = after_actor - before_actor
        score = gain * 175 - pain * 96

        wall_edges_set = set(wall_edges(parsed))
        if self._wall_hits_path(wall_edges_set, target_path):
            score += 45
        if self._wall_hits_path(wall_edges_set, actor_path):
            score -= 36

        tr, tc = node.positions[target]
        direction, wr, wc = parsed
        score += max(0, 5 - abs(wr - tr) - abs((wc + 1) - tc)) * 8

        if direction == "H":
            target_fence = 2 if node.goals[target] == 0 else BOARD_SIZE - 3
            if abs(wr - target_fence) <= 1:
                score += max(0, 6 - abs((wc + 1) - tc)) * 13

        if before_actor <= 4 and before_actor <= before_target:
            score -= 28
        if before_target <= before_actor + 1:
            score += max(0, gain) * 60

        return score if actor == self.me else -score

    def _evaluate(self, node):
        cached = self._eval_cache.get(node)
        if cached is not None:
            return cached

        winner = self._winner(node)
        if winner is not None:
            value = WIN_SCORE - node.turn if winner == self.me else -WIN_SCORE + node.turn
            self._eval_cache[node] = value
            return value

        opponent = 1 - self.me
        my_dist, my_path = self._shortest_path(node, self.me)
        opp_dist, opp_path = self._shortest_path(node, opponent)

        my_race = 2 * my_dist - (1 if node.current == self.me else 0)
        opp_race = 2 * opp_dist - (1 if node.current == opponent else 0)
        race = opp_race - my_race

        my_row, my_col = node.positions[self.me]
        opp_row, opp_col = node.positions[opponent]

        value = race * 880
        value += (node.remaining[self.me] - node.remaining[opponent]) * 42
        value += (4 - abs(my_col - 4)) * 12
        value -= (4 - abs(opp_col - 4)) * 8

        if my_dist <= 2:
            value += 360 * (3 - my_dist)
        if opp_dist <= 2:
            value -= 420 * (3 - opp_dist)

        if my_path:
            value -= self._path_crowding(node, my_path[:7]) * 22
        if opp_path:
            value += self._path_crowding(node, opp_path[:7]) * 17

        if self._has_forward_jump(node, self.me):
            value += 680 if node.current == self.me else 260
        if self._has_forward_jump(node, opponent):
            value -= 760 if node.current == opponent else 280

        if abs(my_row - opp_row) + abs(my_col - opp_col) == 1:
            my_forward = -1 if node.goals[self.me] < my_row else 1
            opp_forward = -1 if node.goals[opponent] < opp_row else 1
            if opp_row - my_row == my_forward:
                value += 150
            if my_row - opp_row == opp_forward:
                value -= 150

        if node.positions[self.me] in self._recent_own_positions[-4:]:
            value -= 160

        self._eval_cache[node] = value
        return value

    def _path_crowding(self, node, path):
        walls = set(node.walls)
        total = 0
        for row, col in path:
            for action in self._walls_touching_cell(row, col):
                parsed = parse_wall(action)
                if parsed in walls:
                    total += 1
        return total

    def _urgent_defense(self, node, legal):
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
            safe.append((self._static_after(node, action), action))

        if not safe:
            return None
        safe.sort(reverse=(node.current == self.me))
        return safe[0][1]

    def _winning_move(self, node, legal):
        goal = node.goals[node.current]
        for action in legal:
            if action not in MOVE_DELTAS:
                continue
            dest = self._move_destination(node, node.current, action)
            if dest is not None and dest[0] == goal:
                return action
        return None

    def _has_immediate_win(self, node, player):
        goal = node.goals[player]
        for action in MOVE_NAMES:
            dest = self._move_destination(node, player, action)
            if dest is not None and dest[0] == goal:
                return True
        return False

    def _best_path_move(self, node, legal):
        moves = [action for action in legal if action in MOVE_DELTAS]
        if not moves:
            return None
        return max(moves, key=lambda action: self._static_after(node, action))

    def _closing_move(self, node, legal):
        actor = node.current
        opponent = 1 - actor
        my_dist, _ = self._shortest_path(node, actor)
        opp_dist, _ = self._shortest_path(node, opponent)
        my_race = 2 * my_dist - 1
        opp_race = 2 * opp_dist

        if my_dist > 5:
            return None
        if my_race > opp_race + 1:
            return None

        action = self._shortest_path_move(node, actor, legal)
        if action is None:
            return None

        child = self._apply(node, action)
        after, _ = self._shortest_path(child, actor)
        if after >= my_dist:
            return None
        if self._has_immediate_win(child, opponent):
            return None
        if self._has_jump_gain(child, opponent) and my_dist > 2:
            return None
        return action

    def _progress_action(self, node, legal):
        if node.turn < 18:
            return None

        actor = node.current
        opponent = 1 - actor
        my_dist, _ = self._shortest_path(node, actor)
        opp_dist, _ = self._shortest_path(node, opponent)
        if my_dist > opp_dist + 1 and node.remaining[actor] > 2:
            return None

        action = self._shortest_path_move(node, actor, legal)
        if action is None:
            return None

        child = self._apply(node, action)
        after, _ = self._shortest_path(child, actor)
        if after >= my_dist:
            return None
        if self._has_immediate_win(child, opponent):
            return None
        if self._has_jump_gain(child, opponent) and my_dist > 3:
            return None
        return action

    def _shortest_path_move(self, node, actor, legal):
        _, path = self._shortest_path(node, actor)
        if len(path) < 2:
            return None
        target = path[1]
        for action in MOVE_NAMES:
            if action in legal and self._move_destination(node, actor, action) == target:
                return action
        return None

    def _has_jump_gain(self, node, player):
        row, col = node.positions[player]
        goal = node.goals[player]
        for action in MOVE_NAMES:
            dest = self._move_destination(node, player, action)
            if dest is None:
                continue
            distance = abs(dest[0] - row) + abs(dest[1] - col)
            if distance >= 2 and abs(goal - dest[0]) < abs(goal - row):
                return True
        return False

    @staticmethod
    def _opening_action(state, legal):
        if state.get("walls"):
            return None
        actor = int(state.get("actor", state.get("player_id", 0)))
        opponent = 1 - actor
        row = state["positions"][actor][0]
        opp_row = state["positions"][opponent][0]
        if abs(row - opp_row) <= 2:
            return None
        goal = state["goal_rows"][actor]
        forward = "MOVE_UP" if goal < row else "MOVE_DOWN"
        return forward if forward in legal else None

    def _legal_moves(self, node, player):
        result = []
        for action in MOVE_NAMES:
            if self._move_destination(node, player, action) is not None:
                result.append(action)
        return result

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

    def _valid_wall_action(self, node, action):
        wall = parse_wall(action)
        if wall is None:
            return False
        return self._valid_wall(node, wall)

    def _valid_wall(self, node, wall):
        direction, row, col = wall
        if direction not in ("H", "V"):
            return False
        if not (0 <= row < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1):
            return False

        walls = set(node.walls)
        if wall in walls:
            return False
        if (opposite(direction), row, col) in walls:
            return False

        blocked = self._blocked_edges(node.walls)
        if any(edge in blocked for edge in wall_edges(wall)):
            return False

        new_walls = tuple(sorted(node.walls + (wall,)))
        return (
            self._path_exists(new_walls, node.positions[0], node.goals[0])
            and self._path_exists(new_walls, node.positions[1], node.goals[1])
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

    def _winner(self, node):
        for player in (0, 1):
            if node.positions[player][0] == node.goals[player]:
                return player
        return None

    def _shortest_path(self, node, player):
        return self._shortest_path_raw(node.walls, node.positions[player], node.goals[player])

    def _shortest_path_raw(self, walls, start, goal_row):
        key = (walls, start, goal_row)
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached

        queue = deque([start])
        parent = {start: None}
        while queue:
            cell = queue.popleft()
            if cell[0] == goal_row:
                path = []
                cur = cell
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                result = (len(path) - 1, tuple(path))
                self._path_cache[key] = result
                return result

            for nxt in self._neighbors_for_path(walls, cell[0], cell[1]):
                if nxt not in parent:
                    parent[nxt] = cell
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

    def _has_forward_jump(self, node, player):
        row, _ = node.positions[player]
        direction = "MOVE_UP" if node.goals[player] < row else "MOVE_DOWN"
        dest = self._move_destination(node, player, direction)
        if dest is None:
            return False
        return abs(dest[0] - row) == 2 and abs(node.goals[player] - dest[0]) < abs(node.goals[player] - row)

    def _walls_blocking_step(self, a, b):
        result = []
        for edge in edges_for_step(a, b):
            result.extend(walls_blocking_edge(edge))
        return result

    @staticmethod
    def _walls_touching_cell(row, col):
        result = []
        for wall_row in (row - 1, row):
            for wall_col in (col - 1, col):
                if 0 <= wall_row < BOARD_SIZE - 1 and 0 <= wall_col < BOARD_SIZE - 1:
                    result.append(f"WALL_H_{wall_row}_{wall_col}")
                    result.append(f"WALL_V_{wall_row}_{wall_col}")
        return result

    @staticmethod
    def _wall_hits_path(wall_edges_set, path):
        for a, b in zip(path, path[1:]):
            for edge in edges_for_step(a, b):
                if edge in wall_edges_set:
                    return True
        return False

    @staticmethod
    def _seen_key(node):
        return (node.current, node.positions, node.remaining, node.walls)

    def _remember_seen(self, node):
        key = self._seen_key(node)
        self._state_seen[key] = self._state_seen.get(key, 0) + 1
        if len(self._state_seen) > 512:
            self._state_seen.clear()

    def _remember_move(self, node, action):
        if action not in MOVE_DELTAS:
            return
        child = self._apply(node, action)
        self._recent_own_positions.append(child.positions[node.current])
        self._recent_own_positions = self._recent_own_positions[-10:]

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise TimeUp

    def _soft_check_time(self):
        if time.perf_counter() + 0.004 >= self.deadline:
            raise TimeUp

    @staticmethod
    def _fallback(state, legal):
        player_id = int(state.get("player_id", state.get("actor", 0)))
        row = state["positions"][player_id][0]
        goal = state["goal_rows"][player_id]
        forward = "MOVE_UP" if goal < row else "MOVE_DOWN"
        if forward in legal:
            return forward
        for action in legal:
            if action.startswith("MOVE_"):
                return action
        return legal[0]


def in_bounds(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def parse_wall(action):
    parts = action.split("_")
    if len(parts) != 4 or parts[0] != "WALL" or parts[1] not in ("H", "V"):
        return None
    try:
        return parts[1], int(parts[2]), int(parts[3])
    except ValueError:
        return None


def opposite(direction):
    return "V" if direction == "H" else "H"


def wall_edges(wall):
    direction, row, col = wall
    if direction == "H":
        return (("H", row, col), ("H", row, col + 1))
    return (("V", row, col), ("V", row + 1, col))


def edges_for_step(a, b):
    ar, ac = a
    br, bc = b
    if ac == bc and ar != br:
        step = 1 if br > ar else -1
        return tuple(("H", min(row, row + step), ac) for row in range(ar, br, step))
    if ar == br and ac != bc:
        step = 1 if bc > ac else -1
        return tuple(("V", ar, min(col, col + step)) for col in range(ac, bc, step))
    return ()


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
