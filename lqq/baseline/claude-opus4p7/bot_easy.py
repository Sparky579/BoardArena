"""Easy Quoridor-like bot.

Beats the trivial forward-walker (`bot_greedy.py`) by using its walls in a
*targeted* way instead of wasting them. The only "smart" piece of reasoning
is wall placement; everything else is naive row-distance heuristics.

Strategy:
  - Always take the winning move if one is legal.
  - Walk forward (or take a forward jump when the framework offers one).
  - When the race has gotten close (within ROW_DISTANCE_TO_ENGAGE rows of
    our own goal) AND the opponent is not strictly behind us in row-distance,
    drop a horizontal wall directly in front of the opponent with some
    probability. The wall is one of the at-most-two `WALL_H` slots that sit
    on the opponent's forward edge — no BFS, no path reasoning, just "the
    cell the opponent wants to step into next".
  - When forward is blocked, sidestep (left/right, then any non-backward move).

No search, no BFS, no path computation — just row-distance and the four
candidate wall slots adjacent to the opponent.
"""

from __future__ import annotations

import random


_FORWARD_OR_JUMP = {
    # Try straight first, then diagonal side-jumps the framework exposes when
    # the straight jump is blocked.
    0: ("MOVE_UP", "MOVE_UP_LEFT", "MOVE_UP_RIGHT"),
    1: ("MOVE_DOWN", "MOVE_DOWN_LEFT", "MOVE_DOWN_RIGHT"),
}
_BACKWARD = {0: "MOVE_DOWN", 1: "MOVE_UP"}
_SIDES = ("MOVE_LEFT", "MOVE_RIGHT")

ROW_DISTANCE_TO_ENGAGE = 5
BLOCK_PROB = 0.55


def _wall_in_front_of(opp_row: int, opp_col: int, opp_goal: int) -> list[str]:
    """Return the (up to two) horizontal-wall actions whose blocked edge sits
    immediately in front of the opponent's current cell."""
    if opp_goal == opp_row:
        return []
    step = 1 if opp_goal > opp_row else -1
    wr = min(opp_row, opp_row + step)
    if not 0 <= wr <= 7:
        return []
    return [f"WALL_H_{wr}_{wc}" for wc in (opp_col - 1, opp_col) if 0 <= wc <= 7]


class Bot:
    name = "claude_opus4p7_easy"

    def __init__(self) -> None:
        self._rng = random.Random()

    def choose_action(self, state):
        legal = state["legal_actions"]
        if not legal:
            return ""
        if len(legal) == 1:
            return legal[0]

        me = state.get("player_id", state.get("actor", 0))
        opp = 1 - me
        my_row = state["positions"][me][0]
        opp_row, opp_col = state["positions"][opp]
        my_goal = state["goal_rows"][me]
        opp_goal = state["goal_rows"][opp]
        walls_left = state["walls_remaining"][me]

        my_dist = abs(my_goal - my_row)
        opp_dist = abs(opp_goal - opp_row)

        # 1. Take the win.
        if my_dist == 1:
            for action in _FORWARD_OR_JUMP[me]:
                if action in legal:
                    return action

        # 2. Drop a wall in front of the opponent when:
        #    - we still have walls,
        #    - we're inside the engagement window,
        #    - opponent is not strictly behind us in row distance,
        #    - opponent isn't 1 from goal (a single wall won't save us),
        #    - and the dice roll says so.
        if (
            walls_left > 0
            and my_dist <= ROW_DISTANCE_TO_ENGAGE
            and opp_dist <= my_dist
            and opp_dist >= 2
            and self._rng.random() < BLOCK_PROB
        ):
            candidates = [
                w for w in _wall_in_front_of(opp_row, opp_col, opp_goal)
                if w in legal
            ]
            if candidates:
                return self._rng.choice(candidates)

        # 3. Walk forward.
        for action in _FORWARD_OR_JUMP[me]:
            if action in legal:
                return action

        # 4. Sidestep.
        sides = [a for a in _SIDES if a in legal]
        if sides:
            return self._rng.choice(sides)

        # 5. Any non-backward move.
        nonback = [
            a for a in legal
            if a.startswith("MOVE_") and a != _BACKWARD[me]
        ]
        if nonback:
            return self._rng.choice(nonback)

        return legal[0]


_BOT = Bot()


def choose_action(state):
    return _BOT.choose_action(state)
