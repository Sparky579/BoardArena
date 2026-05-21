"""Easy Othello bot — the classic "greedy max-flip" beginner.

Plays like someone who just learned the rules:
  - Takes a corner immediately when one is available (this is the only
    real piece of "smarts" — most absolute beginners learn the corner
    rule fast).
  - Otherwise picks the move that flips the most discs right now.
    This is the famous "max-disc greedy" — a well-known *bad* heuristic
    in Othello, because flipping a lot of discs early opens up your
    structure and hands the opponent control. Strong play often does
    the opposite (low-flip "starve the opponent of mobility").
  - 10 % of the time picks a random non-PASS move for some variety.

No search, no positional table, no lookahead beyond "what would this one
move flip right now". Will reliably beat random; will reliably lose to
anything with a real position-aware evaluation.
"""

from __future__ import annotations

import random


_CORNERS = frozenset({"a1", "a8", "h1", "h8"})
_RANDOM_PROB = 0.10


class Bot:
    name = "claude_opus4p7_easy"

    def __init__(self) -> None:
        self._rng = random.Random()

    def choose_action(self, state):
        legal = state["legal_actions"]
        if not legal:
            return "PASS"
        if len(legal) == 1:
            return legal[0]

        # Take a corner if one is offered. Real beginners learn this fast.
        for action in legal:
            if action in _CORNERS:
                return action

        non_pass = [a for a in legal if a != "PASS"]
        if not non_pass:
            return "PASS"

        # Occasionally pick at random for variety.
        if self._rng.random() < _RANDOM_PROB:
            return self._rng.choice(non_pass)

        # Greedy max-flip — the textbook beginner mistake.
        flips = state.get("legal_flips", {})
        best_action = non_pass[0]
        best_count = -1
        for action in non_pass:
            count = len(flips.get(action, ()))
            if count > best_count or (
                count == best_count and self._rng.random() < 0.4
            ):
                best_count = count
                best_action = action
        return best_action


_BOT = Bot()


def choose_action(state):
    return _BOT.choose_action(state)
