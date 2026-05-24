"""Synthetic 'human-like fork builder' for testing anti-fork bots.

Strategy:
1. Turns 0-12: never place walls. Always move toward goal (or center if blocked).
2. Turns 13+: alternate moves and wall placements. Wall placement priority:
   a. If opp_dist <= 5 and a 2-wall fork is available, build the first wall of it.
   b. Else place a wall that extends opp's path by the most (greedy +N).
   c. Else just move.
3. Never place a wall in our own path.

This simulates a human who saves walls early then forks late.
"""

import importlib.util
from pathlib import Path
import sys
from collections import deque


_CORE_PATH = Path(__file__).resolve().parent.parent / "claude-v2" / "bot.py"
_SPEC = importlib.util.spec_from_file_location("_anti_fork_core_h", _CORE_PATH)
_CORE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CORE
_SPEC.loader.exec_module(_CORE)


def choose_action(state):
    return Bot().choose_action(state)


class Bot:
    name = "human_synth"

    def choose_action(self, state_dict):
        legal = list(state_dict.get("legal_actions", ()))
        if not legal:
            return ""
        if len(legal) == 1:
            return legal[0]

        me = int(state_dict.get("player_id", state_dict.get("actor", 0)))
        opp = 1 - me
        my_pos_rc = state_dict["positions"][me]
        opp_pos_rc = state_dict["positions"][opp]
        goal = state_dict.get("goal_rows", [0, 8])[me]
        turn = int(state_dict.get("turn", 0))
        walls_rem = state_dict["walls_remaining"]

        # Build bitmasks like claude-v2 does
        h_mask = v_mask = h_walls = v_walls = 0
        for w in state_dict.get("walls", []):
            d, r, c = w["dir"], int(w["row"]), int(w["col"])
            idx = r * 9 + c
            if d == "H":
                h_walls |= (1 << idx)
                h_mask |= (1 << idx) | (1 << (idx + 1))
            else:
                v_walls |= (1 << idx)
                v_mask |= (1 << idx) | (1 << (idx + 9))

        my_pos = my_pos_rc[0] * 9 + my_pos_rc[1]
        opp_pos = opp_pos_rc[0] * 9 + opp_pos_rc[1]
        opp_goal = state_dict.get("goal_rows", [0, 8])[opp]

        # Instant-win move
        for action in legal:
            if action.startswith("MOVE_"):
                if self._move_dest_row(my_pos_rc, action, opp_pos_rc) == goal:
                    return action

        opp_dist, opp_path = _CORE.bfs_path(opp_pos, opp_goal, h_mask, v_mask)
        my_dist, my_path = _CORE.bfs_path(my_pos, goal, h_mask, v_mask)

        # Opening: only move (toward goal)
        if turn < 12 or walls_rem[me] == 0:
            return self._best_move(legal, my_pos_rc, opp_pos_rc, goal, h_mask, v_mask)

        # Late game: try to build a wall on opp's path that extends them maximally
        best_wall = None
        best_gain = 0
        # Also: prefer walls that increase opp's fork vulnerability (i.e. set up a fork)
        for action in legal:
            if not action.startswith("WALL_"):
                continue
            _, d, rs, cs = action.split("_")
            r, c = int(rs), int(cs)
            if not _CORE.is_valid_wall(d, r, c, h_walls, v_walls):
                continue
            new_h, new_v, new_hw, new_vw = _CORE.apply_wall_masks(
                d, r, c, h_mask, v_mask, h_walls, v_walls,
            )
            new_opp_dist, _ = _CORE.bfs_path(opp_pos, opp_goal, new_h, new_v)
            if new_opp_dist >= 1000:
                continue
            gain = new_opp_dist - opp_dist
            # Make sure this wall doesn't extend our own path too much
            new_my_dist, _ = _CORE.bfs_path(my_pos, goal, new_h, new_v)
            if new_my_dist > my_dist:
                continue
            # Score: gain + bonus for fork potential
            score = gain * 100
            if gain >= 1:
                # Reward walls that also threaten an additional extension
                # (proxy: check if intersecting walls of new opp path can extend further)
                _, new_opp_path = _CORE.bfs_path(opp_pos, opp_goal, new_h, new_v)
                threat_count = 0
                for w2 in _CORE.get_intersecting_walls(new_opp_path):
                    d2, r2, c2 = w2
                    if not _CORE.is_valid_wall(d2, r2, c2, new_hw, new_vw):
                        continue
                    nh, nv, _, _ = _CORE.apply_wall_masks(d2, r2, c2, new_h, new_v, new_hw, new_vw)
                    nd, _ = _CORE.bfs_path(opp_pos, opp_goal, nh, nv)
                    if nd > new_opp_dist:
                        threat_count += 1
                score += threat_count * 5
            if score > best_gain:
                best_gain = score
                best_wall = action

        # Decide: wall vs move
        # Strategy: place walls when opp is close (within 7) AND we have walls,
        # AND wall has nontrivial impact.
        place_wall = best_wall and (opp_dist <= 7 or best_gain >= 200 or turn >= 25)

        if place_wall:
            return best_wall

        return self._best_move(legal, my_pos_rc, opp_pos_rc, goal, h_mask, v_mask)

    def _move_dest_row(self, my_rc, action, opp_rc):
        deltas = {
            "MOVE_UP": (-1, 0), "MOVE_DOWN": (1, 0),
            "MOVE_LEFT": (0, -1), "MOVE_RIGHT": (0, 1),
            "MOVE_UP_LEFT": (-1, -1), "MOVE_UP_RIGHT": (-1, 1),
            "MOVE_DOWN_LEFT": (1, -1), "MOVE_DOWN_RIGHT": (1, 1),
        }
        d = deltas.get(action, (0, 0))
        nr = my_rc[0] + d[0]
        nc = my_rc[1] + d[1]
        if not ("LEFT" in action or "RIGHT" in action) or action in ("MOVE_LEFT", "MOVE_RIGHT"):
            if [nr, nc] == list(opp_rc):
                nr += d[0]
                nc += d[1]
        return nr

    def _best_move(self, legal, my_rc, opp_rc, goal, h_mask, v_mask):
        # Pick the move whose resulting cell has the lowest BFS-distance to goal
        from importlib import import_module
        best = None
        best_d = 99999
        for action in legal:
            if not action.startswith("MOVE_"):
                continue
            deltas = {
                "MOVE_UP": (-1, 0), "MOVE_DOWN": (1, 0),
                "MOVE_LEFT": (0, -1), "MOVE_RIGHT": (0, 1),
                "MOVE_UP_LEFT": (-1, -1), "MOVE_UP_RIGHT": (-1, 1),
                "MOVE_DOWN_LEFT": (1, -1), "MOVE_DOWN_RIGHT": (1, 1),
            }
            d = deltas[action]
            nr = my_rc[0] + d[0]
            nc = my_rc[1] + d[1]
            if not ("LEFT" in action and "UP" not in action and "DOWN" not in action) and \
               not ("RIGHT" in action and "UP" not in action and "DOWN" not in action):
                # check for jump
                if [nr, nc] == list(opp_rc):
                    nr += d[0]
                    nc += d[1]
            if not (0 <= nr <= 8 and 0 <= nc <= 8):
                continue
            cell = nr * 9 + nc
            dist, _ = _CORE.bfs_path(cell, goal, h_mask, v_mask)
            if dist < best_d:
                best_d = dist
                best = action
        if best is None:
            for a in legal:
                if a.startswith("MOVE_"):
                    return a
            return legal[0]
        return best
