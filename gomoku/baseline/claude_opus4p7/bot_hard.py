"""Strong 15x15 Gomoku / Renju bot. Pattern-based α-β with iterative deepening.

Time budget: ~1.7 s per move (under the env's typical 2.0 s decision timeout).

Components:
  - Board representation: list-of-lists, identical shape to `state["board"]`
    (display row 0 = rank 15 top, display row 14 = rank 1 bottom).
  - Fast per-stone "run" evaluation: for every stone on the board, walk the
    4 directions exactly once (only at "run-start" cells) and score the run
    by length and open-end count. About 50-150 µs per evaluation in Python.
  - Pattern-augmented evaluation: a small set of split-pattern checks
    (`XX.XX`, `X.XXX`, `.X.XX.`) catches gap threats the consecutive-run
    scan misses.
  - Move generation: only empty cells within Chebyshev distance 2 of any
    existing stone (or h8 if the board is empty).
  - Move ordering: cheap per-candidate scoring based on the new run lengths
    in each of 4 directions (no full-board eval per candidate). Costs
    ~10 µs / candidate, so ordering ~24 candidates takes ~250 µs / node.
  - Tactical shortcuts at the root: take an immediate five; block an
    opponent's immediate five.
  - Iterative deepening α-β + PVS + transposition table keyed on
    `(board, side_to_move)`. Time check every 256 nodes so deep slow
    iterations bail quickly.
  - Renju forbidden moves are filtered at the *root* using the env's
    `legal_actions`; the search interior does not re-validate (rare
    forbidden-move "leaks" are clamped to the legal set at the root).
"""

from __future__ import annotations

import time


BOARD_SIZE = 15
FILES = "abcdefghijklmno"
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

TIME_BUDGET = 1.7
TIME_HARD_RATIO = 0.92
TIME_SOFT_RATIO = 0.55

WIN_SCORE = 10**9
WIN_BOUND = WIN_SCORE - 1000
INF = 10**12
MAX_PLY = 32

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# Run-length × open-end score table. Indexed as RUN_SCORE[opens][run].
#   opens = number of open ends (0, 1, or 2)
#   run   = length of the consecutive same-color run (1..5+)
# Five-in-a-row beats everything else and is treated as a win.
RUN_SCORE = {
    0: [0, 0,    0,     0,    0,    10_000_000],  # blocked on both sides
    1: [0, 1,   20,   300, 50_000, 10_000_000],  # half-open
    2: [0, 5,  100, 5_000, 1_000_000, 10_000_000],  # fully open
}


# Move-ordering bonuses (very large so they dominate static score).
ORDER_BONUS_WIN = 10**8
ORDER_BONUS_BLOCK = 10**7

# Branching limits by depth (root branching is controlled separately).
DEPTH_BRANCH = {
    1: 10,
    2: 12,
    3: 14,
    4: 16,
}
DEFAULT_BRANCH_DEEP = 18
ROOT_BRANCH = 22


# --------------------------------------------------------------------------
# Board indexing helpers
# --------------------------------------------------------------------------


def sq_to_idx(square: str) -> tuple[int, int]:
    """Map 'h8' to (display_row, col) where display_row 0 is rank 15."""
    file = ord(square[0]) - ord("a")
    rank = int(square[1:])
    return BOARD_SIZE - rank, file


def idx_to_sq(display_row: int, col: int) -> str:
    return f"{FILES[col]}{BOARD_SIZE - display_row}"


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


# --------------------------------------------------------------------------
# Run scanning
# --------------------------------------------------------------------------


def run_length(board: list[list[str]], r: int, c: int, dr: int, dc: int) -> int:
    """Length of consecutive same-color run through (r, c), >= 1 if non-empty."""
    target = board[r][c]
    if target == ".":
        return 0
    count = 1
    nr, nc = r + dr, c + dc
    while in_bounds(nr, nc) and board[nr][nc] == target:
        count += 1
        nr += dr
        nc += dc
    nr, nc = r - dr, c - dc
    while in_bounds(nr, nc) and board[nr][nc] == target:
        count += 1
        nr -= dr
        nc -= dc
    return count


def evaluate_board(board: list[list[str]], side_to_move: int) -> int:
    """Static evaluation from side-to-move's POV.

    Two terms: (a) consecutive-run scores (counted once per run via the
    "start-of-run" convention); (b) split-pattern bumps to catch gap threats.
    """
    my_sym = "B" if side_to_move == 0 else "W"
    opp_sym = "W" if side_to_move == 0 else "B"

    my_score = 0
    opp_score = 0

    # ---- Consecutive runs ----
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = board[r][c]
            if cell == ".":
                continue
            for dr, dc in DIRECTIONS:
                # Only count from the run-start cell.
                pr, pc = r - dr, c - dc
                if in_bounds(pr, pc) and board[pr][pc] == cell:
                    continue
                # Walk forward.
                run = 1
                nr, nc = r + dr, c + dc
                while in_bounds(nr, nc) and board[nr][nc] == cell:
                    run += 1
                    nr += dr
                    nc += dc
                # Open ends.
                left_open = in_bounds(pr, pc) and board[pr][pc] == "."
                right_open = in_bounds(nr, nc) and board[nr][nc] == "."
                opens = int(left_open) + int(right_open)
                run_clamped = run if run < 5 else 5
                pat = RUN_SCORE[opens][run_clamped]
                if cell == my_sym:
                    my_score += pat
                else:
                    opp_score += pat

    # ---- Split-pattern bumps (catch gap threats the run scan misses) ----
    # XX.XX (split four), X.XXX / XXX.X (split four), .X.XX. / .XX.X. (split open three)
    for line in _all_lines(board):
        my_score += _split_score(line, my_sym, opp_sym)
        opp_score += _split_score(line, opp_sym, my_sym)

    return my_score - int(opp_score * 1.10)


def _all_lines(board: list[list[str]]) -> list[str]:
    out: list[str] = []
    for r in range(BOARD_SIZE):
        out.append("".join(board[r]))
    for c in range(BOARD_SIZE):
        out.append("".join(board[r][c] for r in range(BOARD_SIZE)))
    for d in range(-(BOARD_SIZE - 1), BOARD_SIZE):
        cells = [board[r][r - d] for r in range(BOARD_SIZE) if 0 <= r - d < BOARD_SIZE]
        if len(cells) >= 5:
            out.append("".join(cells))
    for d in range(2 * BOARD_SIZE - 1):
        cells = [board[r][d - r] for r in range(BOARD_SIZE) if 0 <= d - r < BOARD_SIZE]
        if len(cells) >= 5:
            out.append("".join(cells))
    return out


_SPLIT_PATTERNS_4 = ("XX.XX", "X.XXX", "XXX.X")
_SPLIT_PATTERNS_3 = (".X.XX.", ".XX.X.")


def _split_score(line: str, my_sym: str, opp_sym: str) -> int:
    s = line.replace(my_sym, "X").replace(opp_sym, "O")
    score = 0
    for pat in _SPLIT_PATTERNS_4:
        if pat in s:
            score += s.count(pat) * 50_000
    for pat in _SPLIT_PATTERNS_3:
        if pat in s:
            score += s.count(pat) * 5_000
    return score


# --------------------------------------------------------------------------
# Move ordering (cheap, no full eval)
# --------------------------------------------------------------------------


def _move_threat_score(board: list[list[str]], r: int, c: int, sym: str) -> int:
    """Sum of run-pattern scores for the 4 directions if `sym` is placed."""
    board[r][c] = sym
    try:
        total = 0
        for dr, dc in DIRECTIONS:
            run = 1
            # forward
            nr, nc = r + dr, c + dc
            while in_bounds(nr, nc) and board[nr][nc] == sym:
                run += 1
                nr += dr
                nc += dc
            right_open = in_bounds(nr, nc) and board[nr][nc] == "."
            # backward
            nr, nc = r - dr, c - dc
            while in_bounds(nr, nc) and board[nr][nc] == sym:
                run += 1
                nr -= dr
                nc -= dc
            left_open = in_bounds(nr, nc) and board[nr][nc] == "."
            opens = int(left_open) + int(right_open)
            r_clamped = run if run < 5 else 5
            total += RUN_SCORE[opens][r_clamped]
        return total
    finally:
        board[r][c] = "."


def _move_creates_five(board: list[list[str]], r: int, c: int, sym: str) -> bool:
    board[r][c] = sym
    try:
        for dr, dc in DIRECTIONS:
            if run_length(board, r, c, dr, dc) >= 5:
                return True
        return False
    finally:
        board[r][c] = "."


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------


def candidate_moves(board: list[list[str]], radius: int = 2) -> list[tuple[int, int]]:
    occ_mask = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    has_stone = False
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ".":
                has_stone = True
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if in_bounds(nr, nc) and board[nr][nc] == ".":
                            occ_mask[nr][nc] = True
    if not has_stone:
        return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
    result = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if occ_mask[r][c]:
                result.append((r, c))
    return result


# --------------------------------------------------------------------------
# Search engine
# --------------------------------------------------------------------------


class TimeUp(Exception):
    pass


class Engine:
    def __init__(self) -> None:
        self.tt: dict[tuple, tuple[int, int, int, tuple[int, int] | None]] = {}
        self.deadline = 0.0
        self.deadline_soft = 0.0
        self.nodes = 0

    def _check_time(self) -> None:
        if (self.nodes & 255) == 0:
            if time.perf_counter() > self.deadline:
                raise TimeUp()

    def _order_moves(
        self,
        board: list[list[str]],
        cands: list[tuple[int, int]],
        side_to_move: int,
        tt_move: tuple[int, int] | None,
    ) -> tuple[list[tuple[int, int, int]], tuple[int, int] | None]:
        """Return (scored_moves, winning_move).

        `winning_move` is the first candidate that creates a 5-in-a-row for
        the side to move (or None). We return it separately so the search
        can shortcut without ambiguity — using a high score for the TT move
        previously collided with the win-shortcut threshold.
        """
        my_sym = "B" if side_to_move == 0 else "W"
        opp_sym = "W" if side_to_move == 0 else "B"
        scored: list[tuple[int, int, int]] = []
        win_move: tuple[int, int] | None = None
        for r, c in cands:
            if _move_creates_five(board, r, c, my_sym):
                if win_move is None:
                    win_move = (r, c)
                score = ORDER_BONUS_WIN
            elif (r, c) == tt_move:
                # High but explicitly less than the "this is a real win"
                # marker, so the win shortcut only fires on true wins.
                score = ORDER_BONUS_WIN - 1
            elif _move_creates_five(board, r, c, opp_sym):
                score = ORDER_BONUS_BLOCK
            else:
                score = _move_threat_score(board, r, c, my_sym) \
                    + _move_threat_score(board, r, c, opp_sym) // 2
            scored.append((score, r, c))
        scored.sort(reverse=True)
        return scored, win_move

    def search(
        self,
        board: list[list[str]],
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        side_to_move: int,
    ) -> int:
        self.nodes += 1
        self._check_time()

        if alpha < -(WIN_SCORE - ply):
            alpha = -(WIN_SCORE - ply)
        if beta > WIN_SCORE - ply - 1:
            beta = WIN_SCORE - ply - 1
        if alpha >= beta:
            return alpha

        key = (tuple("".join(row) for row in board), side_to_move)
        tt_move: tuple[int, int] | None = None
        original_alpha = alpha
        entry = self.tt.get(key)
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_depth >= depth and ply > 0:
                s = tt_score
                if s > WIN_BOUND:
                    s -= ply
                elif s < -WIN_BOUND:
                    s += ply
                if tt_flag == TT_EXACT:
                    return s
                if tt_flag == TT_LOWER and s >= beta:
                    return s
                if tt_flag == TT_UPPER and s <= alpha:
                    return s

        if depth <= 0:
            return evaluate_board(board, side_to_move)

        my_sym = "B" if side_to_move == 0 else "W"

        cands = candidate_moves(board, radius=2)
        if not cands:
            return -(WIN_SCORE - ply)

        scored, win_move = self._order_moves(board, cands, side_to_move, tt_move)

        # Immediate winning move available → return now.
        if win_move is not None:
            best_value = WIN_SCORE - ply
            self.tt[key] = (depth, best_value, TT_LOWER, win_move)
            return best_value

        # Branch limiting.
        branch = DEPTH_BRANCH.get(depth, DEFAULT_BRANCH_DEEP)
        scored = scored[:branch]

        best_value = -INF
        best_move = None
        searched = 0

        for _, r, c in scored:
            board[r][c] = my_sym
            if searched == 0:
                value = -self.search(board, depth - 1, -beta, -alpha, ply + 1, 1 - side_to_move)
            else:
                value = -self.search(board, depth - 1, -alpha - 1, -alpha, ply + 1, 1 - side_to_move)
                if alpha < value < beta:
                    value = -self.search(board, depth - 1, -beta, -alpha, ply + 1, 1 - side_to_move)
            board[r][c] = "."
            searched += 1

            if value > best_value:
                best_value = value
                best_move = (r, c)
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break

        if best_value <= original_alpha:
            flag = TT_UPPER
        elif best_value >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT

        store = best_value
        if store > WIN_BOUND:
            store += ply
        elif store < -WIN_BOUND:
            store -= ply
        self.tt[key] = (depth, store, flag, best_move)
        return best_value

    def _search_root(
        self,
        board: list[list[str]],
        legal_set: set[str],
        depth: int,
        side_to_move: int,
        prev_best: tuple[int, int] | None,
    ) -> tuple[int, tuple[int, int] | None]:
        my_sym = "B" if side_to_move == 0 else "W"

        cands = candidate_moves(board, radius=2)
        cands = [(r, c) for r, c in cands if idx_to_sq(r, c) in legal_set]
        if not cands:
            return -WIN_SCORE, None

        scored, win_move = self._order_moves(board, cands, side_to_move, prev_best)

        # Immediate winning move at the root.
        if win_move is not None:
            return WIN_SCORE, win_move

        scored = scored[:ROOT_BRANCH]

        alpha = -INF
        beta = INF
        best_value = -INF
        best_move = (scored[0][1], scored[0][2])
        searched = 0

        for _, r, c in scored:
            board[r][c] = my_sym
            if searched == 0:
                value = -self.search(board, depth - 1, -beta, -alpha, 1, 1 - side_to_move)
            else:
                value = -self.search(board, depth - 1, -alpha - 1, -alpha, 1, 1 - side_to_move)
                if alpha < value < beta:
                    value = -self.search(board, depth - 1, -beta, -alpha, 1, 1 - side_to_move)
            board[r][c] = "."
            searched += 1

            if value > best_value:
                best_value = value
                best_move = (r, c)
                if value > alpha:
                    alpha = value

        return best_value, best_move

    def choose(self, state, budget: float = TIME_BUDGET) -> str:
        legal = state["legal_actions"]
        if not legal:
            return ""
        if len(legal) == 1:
            return legal[0]

        start = time.perf_counter()
        self.deadline = start + budget * TIME_HARD_RATIO
        self.deadline_soft = start + budget * TIME_SOFT_RATIO
        self.nodes = 0
        if len(self.tt) > 200_000:
            self.tt.clear()

        side_to_move = state.get("actor", state.get("player_id", 0))
        rows = state["board"]
        board = [list(row) for row in rows]
        legal_set = set(legal)
        my_sym = "B" if side_to_move == 0 else "W"
        opp_sym = "W" if side_to_move == 0 else "B"

        # Opening: play the center.
        if not any(ch != "." for row in rows for ch in row):
            return "h8" if "h8" in legal_set else legal[0]

        # 1. Take an immediate winning move.
        for sq in legal:
            r, c = sq_to_idx(sq)
            if _move_creates_five(board, r, c, my_sym):
                return sq

        # 2. Block opponent's immediate 5-in-a-row threats.
        opp_wins: list[str] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != ".":
                    continue
                if _move_creates_five(board, r, c, opp_sym):
                    opp_wins.append(idx_to_sq(r, c))
        if opp_wins:
            for sq in opp_wins:
                if sq in legal_set:
                    return sq

        # 3. Iterative deepening.
        best_move: tuple[int, int] | None = None
        for depth in range(2, MAX_PLY + 1):
            if depth > 2 and time.perf_counter() > self.deadline_soft:
                break
            try:
                value, m = self._search_root(board, legal_set, depth, side_to_move, best_move)
            except TimeUp:
                break
            if m is not None:
                best_move = m
            if abs(value) >= WIN_BOUND:
                break

        if best_move is not None:
            sq = idx_to_sq(*best_move)
            if sq in legal_set:
                return sq

        # Fallback: any legal candidate.
        for r, c in candidate_moves(board):
            sq = idx_to_sq(r, c)
            if sq in legal_set:
                return sq
        return legal[0]


# --------------------------------------------------------------------------
# Public Bot class
# --------------------------------------------------------------------------


class Bot:
    name = "claude_opus4p7_hard"

    def __init__(self) -> None:
        self.engine = Engine()

    def choose_action(self, state) -> str:
        return self.engine.choose(state)


_MODULE_ENGINE = Engine()


def choose_action(state) -> str:
    return _MODULE_ENGINE.choose(state)
