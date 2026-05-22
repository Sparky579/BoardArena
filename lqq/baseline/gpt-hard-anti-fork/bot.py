"""gpt-hard with extra anti-fork wall-safety evaluation.

The base gpt-hard bot is strong at shortest-path races, but it can over-enter
positions where the opponent has several equivalent wall threats and can wait
for a left/right commitment.  This wrapper keeps the base search and adds a
cheap pressure model: positions are penalized when the opponent has one or more
legal walls that can sharply extend our path, even before that extension appears
as an immediate shortest-path loss.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time


_BASE_PATH = Path(__file__).resolve().parents[1] / "gpt-hard" / "bot.py"
_SPEC = importlib.util.spec_from_file_location("_lqq_gpt_hard_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot import base bot: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)


BOARD_SIZE = _BASE.BOARD_SIZE
INF = _BASE.INF
MOVE_DELTAS = _BASE.MOVE_DELTAS
Node = _BASE.Node


def choose_action(state):
    return Bot().choose_action(state)


class Bot(_BASE.Bot):
    name = "gpt_hard_anti_fork"

    def __init__(self):
        super().__init__()
        self._pressure_cache = {}
        self._escape_cache = {}
        self._root_turn = 0
        self._security_eval = False

    def choose_action(self, state):
        self._pressure_cache.clear()
        self._escape_cache.clear()
        self._root_turn = int(state.get("turn", 0))
        self._security_eval = False
        action = super().choose_action(state)
        if self._root_turn < 28:
            return action
        try:
            node = self._node_from_state(state)
            return self._safer_root_action(node, list(state["legal_actions"]), action)
        except Exception:
            return action

    def _evaluate(self, node):
        score = super()._evaluate(node)
        if self._security_eval and self._root_turn >= 28 and node.turn <= self._root_turn + 2:
            score += self._fork_security_score(node)
        return score

    def _candidate_actions(self, node, legal, root):
        actions = list(super()._candidate_actions(node, legal, root))
        if not self._security_eval or not root or node.turn < 28 or node.remaining[node.current] <= 0:
            return actions

        legal_set = set(legal)
        extras = []
        for action in self._defensive_wall_set(node):
            if action not in legal_set or action in actions:
                continue
            child = self._apply(node, action)
            score = self._fork_security_score(child) + self._raw_wall_score(node, action)
            extras.append((score, action))
        extras.sort(reverse=True)

        for _, action in extras[:8]:
            actions.append(action)

        seen = set()
        unique = []
        for action in actions:
            if action in legal_set and action not in seen:
                seen.add(action)
                unique.append(action)
        return unique[:34]

    def _book_action(self, node, legal):
        return super()._book_action(node, legal)

    def _safer_root_action(self, node, legal, base_action):
        stop = time.perf_counter() + 0.28
        legal_set = set(legal)
        candidates = [base_action]
        candidates.extend(action for action in legal if action in MOVE_DELTAS)

        if node.remaining[node.current] > 0:
            walls = []
            for action in self._defensive_wall_set(node):
                if time.perf_counter() >= stop:
                    break
                if action in legal_set:
                    walls.append((self._raw_wall_score(node, action), action))
            walls.sort(reverse=True)
            candidates.extend(action for _, action in walls[:8])

        seen = set()
        best_action = base_action
        best_score = -10**18
        base_score = None
        for action in candidates:
            if time.perf_counter() >= stop:
                break
            if action not in legal_set or action in seen:
                continue
            seen.add(action)
            child = self._apply(node, action)
            score = super()._static_after(node, action)
            score += self._fork_security_score(child)
            if action == base_action:
                base_score = score
            if score > best_score:
                best_score = score
                best_action = action

        if base_score is None:
            return base_action
        return best_action if best_score > base_score + 180 else base_action

    def _fork_security_score(self, node):
        winner = self._winner(node)
        if winner is not None:
            return 0

        me = self.me
        opp = 1 - me
        my_pressure = self._wall_pressure(node, by=opp, target=me)
        opp_pressure = self._wall_pressure(node, by=me, target=opp)

        score = (opp_pressure - my_pressure) * 145
        if node.current == opp:
            score -= my_pressure * 115
        else:
            score += opp_pressure * 70

        if node.turn <= self._root_turn + 1:
            my_escape = self._escape_value(node, me)
            opp_escape = self._escape_value(node, opp)
            score += (my_escape - opp_escape) * 55

        my_dist, _ = self._shortest_path(node, me)
        opp_dist, _ = self._shortest_path(node, opp)
        if my_dist <= 5:
            score -= my_pressure * (52 + (6 - my_dist) * 14)
        if opp_dist <= 5:
            score += opp_pressure * (42 + (6 - opp_dist) * 12)
        return score

    def _wall_pressure(self, node, by, target):
        key = ("pressure", by, target, node.positions, node.remaining, node.walls)
        cached = self._pressure_cache.get(key)
        if cached is not None:
            return cached

        if node.remaining[by] <= 0:
            self._pressure_cache[key] = 0
            return 0

        attack_node = Node(
            current=by,
            positions=node.positions,
            goals=node.goals,
            remaining=node.remaining,
            walls=node.walls,
            turn=node.turn,
        )
        base_dist, _ = self._shortest_path(attack_node, target)
        if base_dist >= INF:
            self._pressure_cache[key] = 0
            return 0

        candidates = set()
        candidates.update(self._tactical_wall_set(attack_node))
        candidates.update(self._fence_wall_set(attack_node, target))
        candidates.update(self._near_path_wall_set(attack_node, target))

        gains = []
        for action in sorted(candidates)[:14]:
            wall = _BASE.parse_wall(action)
            if wall is None or not self._valid_wall(attack_node, wall):
                continue
            child = self._apply(attack_node, action)
            after_dist, _ = self._shortest_path(child, target)
            gain = after_dist - base_dist
            if gain > 0:
                gains.append(gain)

        if not gains:
            value = 0
        else:
            gains.sort(reverse=True)
            value = gains[0] * 2
            if len(gains) > 1:
                value += gains[1]
            if len(gains) > 2:
                value += min(gains[2], 2)

        self._pressure_cache[key] = value
        return value

    def _escape_value(self, node, player):
        key = ("escape", player, node.positions, node.remaining, node.walls, node.current)
        cached = self._escape_cache.get(key)
        if cached is not None:
            return cached

        move_node = Node(
            current=player,
            positions=node.positions,
            goals=node.goals,
            remaining=node.remaining,
            walls=node.walls,
            turn=node.turn,
        )
        before, _ = self._shortest_path(move_node, player)
        if before >= INF:
            self._escape_cache[key] = -20
            return -20

        opponent = 1 - player
        values = []
        for action in self._legal_actions(move_node):
            if action not in MOVE_DELTAS:
                continue
            child = self._apply(move_node, action)
            after, _ = self._shortest_path(child, player)
            pressure = self._wall_pressure(child, by=opponent, target=player)
            progress = before - after
            values.append(progress * 3 - pressure)

        if not values:
            value = -20
        else:
            values.sort(reverse=True)
            value = values[0]
            if len(values) > 1:
                value += max(-4, values[1]) * 0.45

        self._escape_cache[key] = value
        return value

    def _near_path_wall_set(self, node, target):
        result = set()
        _, path = self._shortest_path(node, target)
        for row, col in path[:12]:
            for wall in self._walls_touching_cell(row, col):
                direction, wr, wc = wall
                result.add(f"WALL_{direction}_{wr}_{wc}")
            for dc in (-2, -1, 0, 1):
                wr = row - 1
                wc = col + dc
                if 0 <= wr < BOARD_SIZE - 1 and 0 <= wc < BOARD_SIZE - 1:
                    result.add(f"WALL_H_{wr}_{wc}")
                    result.add(f"WALL_V_{wr}_{wc}")
            for dr in (-2, -1, 0, 1):
                wr = row + dr
                wc = col - 1
                if 0 <= wr < BOARD_SIZE - 1 and 0 <= wc < BOARD_SIZE - 1:
                    result.add(f"WALL_H_{wr}_{wc}")
                    result.add(f"WALL_V_{wr}_{wc}")
        return result

    def _defensive_wall_set(self, node):
        actor = node.current
        opponent = 1 - actor
        result = set()
        result.update(self._fence_wall_set(node, actor))
        result.update(self._near_path_wall_set(node, actor))
        result.update(self._near_path_wall_set(node, opponent))
        return result
