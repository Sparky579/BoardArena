"""CPU-only easy baseline bot for BoardArena Othello."""

from __future__ import annotations


name = "gpt5p5_othello_easy"

BOARD_SIZE = 8
EMPTY = "."
PASS_ACTION = "PASS"
FILES = "abcdefgh"
RANKS = "12345678"
DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
SYMBOLS = ("B", "W")

POSITION_WEIGHTS = (
    (120, -20, 20, 5, 5, 20, -20, 120),
    (-20, -40, -5, -5, -5, -5, -40, -20),
    (20, -5, 15, 3, 3, 15, -5, 20),
    (5, -5, 3, 3, 3, 3, -5, 5),
    (5, -5, 3, 3, 3, 3, -5, 5),
    (20, -5, 15, 3, 3, 15, -5, 20),
    (-20, -40, -5, -5, -5, -5, -40, -20),
    (120, -20, 20, 5, 5, 20, -20, 120),
)

CORNERS = {"a1", "a8", "h1", "h8"}
CORNER_DANGER = {
    "a1": {"a2", "b1", "b2"},
    "a8": {"a7", "b8", "b7"},
    "h1": {"h2", "g1", "g2"},
    "h8": {"h7", "g8", "g7"},
}


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        raise ValueError("no legal actions")
    if legal == [PASS_ACTION]:
        return PASS_ACTION

    player = state["actor"]
    board = _rows_to_board(state["board"])
    legal_flips = state.get("legal_flips", {})

    scored = []
    for action in legal:
        row, col = _square_to_coords(action)
        flips = legal_flips.get(action) or _flips_for(board, player, action)
        score = POSITION_WEIGHTS[row][col]
        score += 6 * len(flips)
        score += _corner_score(board, player, action)

        next_board = _apply_move(board, player, action, flips)
        opp_moves = len(_legal_placements(next_board, 1 - player))
        own_moves = len(_legal_placements(next_board, player))
        score += own_moves - 3 * opp_moves

        empties = sum(cell == EMPTY for line in next_board for cell in line)
        if empties <= 12:
            score += _disc_margin(next_board, player)

        scored.append((score, action))

    best_score = max(score for score, _ in scored)
    best_actions = [action for score, action in scored if score == best_score]
    return sorted(best_actions)[0]


def _rows_to_board(rows):
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for display_index, row_text in enumerate(rows):
        internal_row = BOARD_SIZE - 1 - display_index
        for col, value in enumerate(row_text):
            board[internal_row][col] = value
    return board


def _square_to_coords(square):
    return int(square[1]) - 1, FILES.index(square[0])


def _coords_to_square(row, col):
    return f"{FILES[col]}{row + 1}"


def _on_board(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def _flips_for(board, player, square):
    row, col = _square_to_coords(square)
    if board[row][col] != EMPTY:
        return []

    mine = SYMBOLS[player]
    theirs = SYMBOLS[1 - player]
    flips = []
    for delta_row, delta_col in DIRECTIONS:
        path = []
        scan_row = row + delta_row
        scan_col = col + delta_col
        while _on_board(scan_row, scan_col) and board[scan_row][scan_col] == theirs:
            path.append((scan_row, scan_col))
            scan_row += delta_row
            scan_col += delta_col
        if path and _on_board(scan_row, scan_col) and board[scan_row][scan_col] == mine:
            flips.extend(_coords_to_square(path_row, path_col) for path_row, path_col in path)
    return sorted(flips)


def _legal_placements(board, player):
    actions = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            square = _coords_to_square(row, col)
            if _flips_for(board, player, square):
                actions.append(square)
    return actions


def _apply_move(board, player, action, flips):
    next_board = [list(row) for row in board]
    row, col = _square_to_coords(action)
    next_board[row][col] = SYMBOLS[player]
    for square in flips:
        flip_row, flip_col = _square_to_coords(square)
        next_board[flip_row][flip_col] = SYMBOLS[player]
    return next_board


def _corner_score(board, player, action):
    if action in CORNERS:
        return 600

    score = 0
    mine = SYMBOLS[player]
    for corner, danger_squares in CORNER_DANGER.items():
        if action not in danger_squares:
            continue
        row, col = _square_to_coords(corner)
        if board[row][col] == EMPTY:
            score -= 90
        elif board[row][col] == mine:
            score += 20
    return score


def _disc_margin(board, player):
    mine = SYMBOLS[player]
    theirs = SYMBOLS[1 - player]
    my_count = sum(cell == mine for row in board for cell in row)
    their_count = sum(cell == theirs for row in board for cell in row)
    return my_count - their_count
