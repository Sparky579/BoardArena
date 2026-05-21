"""Easy 15x15 Gomoku / Renju bot — beginner level.

Plays like someone who just learned the rules:
  - Always takes a winning move if one is offered (any move that completes a
    5-in-a-row — these are detected by checking each candidate against the
    "if I play here, do I get 5+ stones in a row?" rule).
  - Always blocks an opponent's *immediate* 5-in-a-row threat. (Misses any
    threat that needs more than 1 move to read.)
  - Otherwise plays a stone adjacent to one of its own recent stones (greedy
    extension), with a mild center bias and small randomness — no real
    pattern reading, no open-four / double-three awareness.

`legal_actions` is pre-filtered by the environment to exclude black's
forbidden moves, so this bot never has to think about renju forbidden
moves itself.
"""

from __future__ import annotations

import random


BOARD_SIZE = 15
FILES = "abcdefghijklmno"
CENTER = BOARD_SIZE // 2  # h8 = (7, 7)


def _sq_to_coords(square: str) -> tuple[int, int]:
    file_char = square[0]
    rank_text = square[1:]
    return int(rank_text) - 1, FILES.index(file_char)


def _coords_to_sq(row: int, col: int) -> str:
    return f"{FILES[col]}{row + 1}"


def _on_board(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def _max_run_if_play(board: list[str], row: int, col: int, my_symbol: str) -> int:
    """Maximum run length in any direction if `my_symbol` is placed at (row,col)."""
    # board is rows top→bottom (rank 15 first); convert internally.
    # Internal row index: 0 = rank 1 (bottom). board[0] = rank 15 (top).
    # We work with board indexed by [display_index][col] where display 0 = top rank.
    # But the env's state board is from rank N down to 1, so display index 0 = rank 15.
    # For run-length we just need to step in one of 4 directions on the *cell grid*.
    # Easier: convert board to a 2D array of cells indexed by (rank-1, file).
    best = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        # forward
        r, c = row + dr, col + dc
        while _on_board(r, c) and _cell(board, r, c) == my_symbol:
            count += 1
            r += dr
            c += dc
        # backward
        r, c = row - dr, col - dc
        while _on_board(r, c) and _cell(board, r, c) == my_symbol:
            count += 1
            r -= dr
            c -= dc
        if count > best:
            best = count
    return best


def _cell(board: list[str], row: int, col: int) -> str:
    """Cell at (row, col) where row=0 is rank 1.

    The state's board is laid out with board[0] = rank 15 (top). So our
    internal row r (1-indexed bottom-up minus one) maps to display
    index (BOARD_SIZE - 1 - r).
    """
    display_index = BOARD_SIZE - 1 - row
    return board[display_index][col]


def _has_any_stone(board: list[str]) -> bool:
    return any(ch != "." for row in board for ch in row)


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

        actor = state.get("actor", state.get("player_id", 0))
        my_symbol = "B" if actor == 0 else "W"
        opp_symbol = "W" if actor == 0 else "B"
        board = state["board"]

        # Opening: empty board → play the center.
        if not _has_any_stone(board) and "h8" in legal:
            return "h8"

        # 1. Take an immediate winning move (any direction reaching 5+).
        for action in legal:
            row, col = _sq_to_coords(action)
            if _max_run_if_play(board, row, col, my_symbol) >= 5:
                return action

        # 2. Block the opponent's immediate 5-in-a-row threats.
        block_candidates: list[str] = []
        for action in legal:
            row, col = _sq_to_coords(action)
            if _max_run_if_play(board, row, col, opp_symbol) >= 5:
                block_candidates.append(action)
        if block_candidates:
            return self._rng.choice(block_candidates)

        # 3. Otherwise greedily extend our own stones, with a mild center bias
        #    and small randomness.
        scored: list[tuple[int, str]] = []
        for action in legal:
            row, col = _sq_to_coords(action)
            score = 0
            # Bonus per nearby own stone (within Chebyshev distance 2).
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if not _on_board(nr, nc):
                        continue
                    cell = _cell(board, nr, nc)
                    if cell == my_symbol:
                        weight = 3 if max(abs(dr), abs(dc)) == 1 else 1
                        score += weight
                    elif cell == opp_symbol:
                        score += 1  # extending near the contact too
            # Center attraction (Chebyshev distance to h8).
            dist = max(abs(row - CENTER), abs(col - CENTER))
            score += max(0, 6 - dist)
            scored.append((score, action))

        # Pick from the top with a bit of randomness.
        scored.sort(reverse=True)
        top = []
        if scored:
            best_score = scored[0][0]
            cutoff = best_score - 2
            for s, action in scored:
                if s >= cutoff:
                    top.append(action)
                else:
                    break
        if not top:
            return self._rng.choice(legal)
        return self._rng.choice(top)


_BOT = Bot()


def choose_action(state):
    return _BOT.choose_action(state)
