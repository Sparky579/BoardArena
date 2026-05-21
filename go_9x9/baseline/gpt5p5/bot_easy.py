"""CPU-only easy baseline bot for BoardArena 9x9 Go."""

from __future__ import annotations


name = "gpt5p5_go9_easy"

BOARD_SIZE = 9
EMPTY = "."
PASS_ACTION = "PASS"
FILES = "abcdefghi"
SYMBOLS = ("B", "W")


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        raise ValueError("no legal actions")
    moves = [action for action in legal if action != PASS_ACTION]
    if not moves:
        return PASS_ACTION

    board = _rows_to_board(state["board"])
    player = state["actor"]
    if state["empty_count"] <= 10 and _score_margin(state, player) > 2:
        return PASS_ACTION

    scored = []
    for action in moves:
        played = _play(board, player, action)
        if played is None:
            continue
        next_board, captured = played
        row, col = _square_to_coords(action)
        group, liberties = _collect_group(next_board, row, col)
        opp_moves = _mobility(next_board, 1 - player)
        score = 0
        score += 45 * len(captured)
        score += 6 * min(len(liberties), 6)
        score -= 22 if len(liberties) == 1 and not captured else 0
        score += _shape_score(board, next_board, player, row, col)
        score -= opp_moves // 2
        score += _opening_bias(row, col, state["plies"])
        scored.append((score, action))

    if not scored:
        return PASS_ACTION
    best_score = max(score for score, _ in scored)
    best = [action for score, action in scored if score == best_score]
    return sorted(best)[0]


def _rows_to_board(rows):
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for display_index, row_text in enumerate(rows):
        internal_row = BOARD_SIZE - 1 - display_index
        for col, value in enumerate(row_text):
            board[internal_row][col] = value
    return board


def _square_to_coords(square):
    return int(square[1:]) - 1, FILES.index(square[0])


def _coords_to_square(row, col):
    return f"{FILES[col]}{row + 1}"


def _neighbors(row, col):
    if row > 0:
        yield row - 1, col
    if row + 1 < BOARD_SIZE:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col + 1 < BOARD_SIZE:
        yield row, col + 1


def _collect_group(board, row, col):
    color = board[row][col]
    group = {(row, col)}
    liberties = set()
    stack = [(row, col)]
    while stack:
        current_row, current_col = stack.pop()
        for next_row, next_col in _neighbors(current_row, current_col):
            value = board[next_row][next_col]
            if value == EMPTY:
                liberties.add((next_row, next_col))
            elif value == color and (next_row, next_col) not in group:
                group.add((next_row, next_col))
                stack.append((next_row, next_col))
    return group, liberties


def _play(board, player, action):
    row, col = _square_to_coords(action)
    if board[row][col] != EMPTY:
        return None
    mine = SYMBOLS[player]
    theirs = SYMBOLS[1 - player]
    next_board = [list(line) for line in board]
    next_board[row][col] = mine
    captured = []

    for next_row, next_col in _neighbors(row, col):
        if next_board[next_row][next_col] != theirs:
            continue
        group, liberties = _collect_group(next_board, next_row, next_col)
        if liberties:
            continue
        for stone_row, stone_col in group:
            next_board[stone_row][stone_col] = EMPTY
            captured.append(_coords_to_square(stone_row, stone_col))

    group, liberties = _collect_group(next_board, row, col)
    if not liberties:
        return None
    return next_board, captured


def _mobility(board, player):
    count = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != EMPTY:
                continue
            if _play(board, player, _coords_to_square(row, col)) is not None:
                count += 1
    return count


def _shape_score(old_board, board, player, row, col):
    mine = SYMBOLS[player]
    theirs = SYMBOLS[1 - player]
    friendly = 0
    enemy = 0
    empty = 0
    atari_targets = 0
    for next_row, next_col in _neighbors(row, col):
        value = old_board[next_row][next_col]
        if value == mine:
            friendly += 1
        elif value == theirs:
            enemy += 1
            _, liberties = _collect_group(old_board, next_row, next_col)
            if len(liberties) == 1:
                atari_targets += 1
        else:
            empty += 1

    score = 8 * friendly + 5 * enemy + 3 * empty + 25 * atari_targets
    if _is_likely_own_eye(old_board, player, row, col):
        score -= 70
    return score


def _is_likely_own_eye(board, player, row, col):
    mine = SYMBOLS[player]
    for next_row, next_col in _neighbors(row, col):
        if board[next_row][next_col] != mine:
            return False
    diagonal_friends = 0
    diagonal_total = 0
    for delta_row, delta_col in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        scan_row = row + delta_row
        scan_col = col + delta_col
        if 0 <= scan_row < BOARD_SIZE and 0 <= scan_col < BOARD_SIZE:
            diagonal_total += 1
            if board[scan_row][scan_col] == mine:
                diagonal_friends += 1
    return diagonal_total == 0 or diagonal_friends >= max(1, diagonal_total - 1)


def _opening_bias(row, col, plies):
    if plies > 18:
        return 0
    star_points = {(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)}
    if (row, col) in star_points:
        return 28
    distance_to_center = abs(row - 4) + abs(col - 4)
    return 8 - distance_to_center


def _score_margin(state, player):
    scores = state["scores"]
    return scores[player] - scores[1 - player]
