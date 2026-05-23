"""CPU-only hard baseline bot for BoardArena 9x9 Go.

Pure rules-based MCTS for 9x9 Go. The search is deliberately Go-shaped:
urgent captures and saves are considered first, true-eye filling is suppressed,
rollouts stay local around fights, and the tree only expands plausible tactical
and territorial candidates instead of scattering visits over all empty points.
"""

from __future__ import annotations

import math
import random
import time


name = "gpt5p5_go9_hard"

BOARD_SIZE = 9
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE
EMPTY = "."
PASS_ACTION = "PASS"
FILES = "abcdefghi"
SYMBOLS = ("B", "W")
KOMI = 6.5
TIME_LIMIT_SECONDS = 1.82
SAFETY_MARGIN_SECONDS = 0.15
ROLLOUT_LIMIT = 180
EXPLORATION = 0.95
ROOT_CANDIDATES = 30
CHILD_CANDIDATES = 24
ROLLOUT_CANDIDATES = 14
ROOT_PRIOR_WEIGHT = 0.44
WIN_SCORE = 100_000

INDEX_TO_ACTION = tuple(f"{FILES[index % BOARD_SIZE]}{index // BOARD_SIZE + 1}" for index in range(BOARD_POINTS))
ACTION_TO_INDEX = {action: index for index, action in enumerate(INDEX_TO_ACTION)}

NEIGHBORS = tuple(
    tuple(
        row * BOARD_SIZE + col
        for row, col in (
            (index // BOARD_SIZE - 1, index % BOARD_SIZE),
            (index // BOARD_SIZE + 1, index % BOARD_SIZE),
            (index // BOARD_SIZE, index % BOARD_SIZE - 1),
            (index // BOARD_SIZE, index % BOARD_SIZE + 1),
        )
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
    )
    for index in range(BOARD_POINTS)
)
DIAGONALS = tuple(
    tuple(
        row * BOARD_SIZE + col
        for row, col in (
            (index // BOARD_SIZE - 1, index % BOARD_SIZE - 1),
            (index // BOARD_SIZE - 1, index % BOARD_SIZE + 1),
            (index // BOARD_SIZE + 1, index % BOARD_SIZE - 1),
            (index // BOARD_SIZE + 1, index % BOARD_SIZE + 1),
        )
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
    )
    for index in range(BOARD_POINTS)
)
STAR_POINTS = {20, 24, 40, 56, 60}


class Node:
    __slots__ = (
        "board",
        "player",
        "previous_board",
        "pass_count",
        "parent",
        "action",
        "children",
        "untried",
        "visits",
        "value",
        "prior",
    )

    def __init__(self, board, player, previous_board, pass_count, parent=None, action=None, legal=None, prior=0.0):
        self.board = board
        self.player = player
        self.previous_board = previous_board
        self.pass_count = pass_count
        self.parent = parent
        self.action = action
        self.children = {}
        self.untried = [] if legal is None else list(legal)
        self.visits = 0
        self.value = 0.0
        self.prior = prior


class SearchTimeout(Exception):
    pass


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        raise ValueError("no legal actions")
    legal_moves = [action for action in legal if action != PASS_ACTION]
    if not legal_moves:
        return PASS_ACTION
    if len(legal_moves) == 1:
        return legal_moves[0]
    if state.get("plies", 0) == 0 and "e5" in legal_moves:
        return "e5"

    board = _parse_board(state["board"])
    player = state["actor"]
    pass_count = state.get("pass_count", 0)
    last_move = state.get("last_move")
    legal_set = set(legal_moves)
    deadline = time.perf_counter() + _time_budget(state)

    urgent = _urgent_action(board, player, None, legal_set)
    if urgent is not None:
        return urgent

    root_actions = _candidate_actions(board, player, None, legal, last_move, ROOT_CANDIDATES)
    fallback = root_actions[0] if root_actions else legal_moves[0]
    root_actions = [action for action in root_actions if action != PASS_ACTION]
    if not root_actions:
        return PASS_ACTION if _pass_is_sensible(board, player, pass_count) else fallback

    best_action = fallback
    best_value = -math.inf
    previous_values: dict[str, float] = {}
    max_depth = _max_search_depth(_empty_count(board))

    for depth in range(1, max_depth + 1):
        if time.perf_counter() >= deadline - 0.035:
            break
        table: dict[tuple[str, int, str | None, int, int], float] = {}
        completed = True
        depth_best = best_action
        depth_best_value = -math.inf
        alpha = -math.inf
        ordered = sorted(
            root_actions,
            key=lambda action: (previous_values.get(action, -math.inf), _action_rank(board, player, None, action, last_move)),
            reverse=True,
        )
        depth_values: dict[str, float] = {}
        for action in ordered:
            if time.perf_counter() >= deadline - 0.025:
                completed = False
                break
            next_state = _next_state(board, player, None, pass_count, action)
            if next_state is None:
                continue
            next_board, next_player, next_previous, next_pass_count = next_state
            try:
                value = -_negamax(
                    next_board,
                    next_player,
                    next_previous,
                    next_pass_count,
                    depth - 1,
                    -math.inf,
                    -alpha,
                    deadline,
                    table,
                    action,
                )
            except SearchTimeout:
                completed = False
                break
            value += ROOT_PRIOR_WEIGHT * _move_score(board, player, None, action)
            depth_values[action] = value
            if value > depth_best_value:
                depth_best_value = value
                depth_best = action
            if value > alpha:
                alpha = value

        if completed and depth_values:
            previous_values = depth_values
            best_action = depth_best
            best_value = depth_best_value

    if best_value == -math.inf:
        return fallback
    return best_action


def _negamax(board, player, previous_board, pass_count, depth, alpha, beta, deadline, table, last_action):
    if time.perf_counter() >= deadline - 0.015:
        raise SearchTimeout()
    if _is_terminal(board, pass_count):
        return _terminal_score(board, player)
    if depth <= 0:
        return _evaluate(board, player)

    key = (board, player, previous_board, pass_count, depth)
    cached = table.get(key)
    if cached is not None:
        return cached

    legal = _legal_actions(board, player, previous_board)
    actions = _candidate_actions(board, player, previous_board, legal, last_action, _search_width(depth, _empty_count(board)))
    if not actions:
        actions = [PASS_ACTION]

    best = -math.inf
    original_alpha = alpha
    ordered = sorted(actions, key=lambda action: _action_rank(board, player, previous_board, action, last_action), reverse=True)
    for action in ordered:
        if action == PASS_ACTION and not _pass_is_sensible(board, player, pass_count):
            continue
        next_state = _next_state(board, player, previous_board, pass_count, action)
        if next_state is None:
            continue
        next_board, next_player, next_previous, next_pass_count = next_state
        value = -_negamax(
            next_board,
            next_player,
            next_previous,
            next_pass_count,
            depth - 1,
            -beta,
            -alpha,
            deadline,
            table,
            action,
        )
        value += 0.045 * _move_score(board, player, previous_board, action)
        if value > best:
            best = value
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if best == -math.inf:
        best = _evaluate(board, player)
    if best > original_alpha and best < beta:
        table[key] = best
    return best


def _max_search_depth(empties):
    if empties > 22:
        return 2
    return 3


def _search_width(depth, empties):
    if empties > 58:
        return 18 if depth <= 1 else 12
    if empties > 34:
        return 22 if depth <= 1 else 14
    if empties > 16:
        return 28 if depth <= 1 else 18
    return 36 if depth <= 1 else 24


def _select_child(node, root_player):
    log_visits = math.log(node.visits + 1)

    def uct(child):
        if child.visits == 0:
            return float("inf")
        mean = child.value / child.visits
        if node.player != root_player:
            mean = -mean
        explore = EXPLORATION * math.sqrt(log_visits / child.visits)
        prior = 0.10 * child.prior / (1 + child.visits)
        return mean + explore + prior

    return max(node.children.values(), key=uct)


def _rollout(board, player, previous_board, pass_count, root_player, rng, deadline):
    current_board = board
    current_player = player
    current_previous = previous_board
    current_passes = pass_count
    last_action = None

    for _ in range(ROLLOUT_LIMIT):
        if _is_terminal(current_board, current_passes):
            return _terminal_value(current_board, root_player)
        if time.perf_counter() >= deadline:
            return _soft_value(current_board, root_player)

        legal = _legal_actions(current_board, current_player, current_previous)
        action = _rollout_action(current_board, current_player, current_previous, current_passes, legal, last_action, rng)
        next_state = _next_state(current_board, current_player, current_previous, current_passes, action)
        if next_state is None:
            next_state = _next_state(current_board, current_player, current_previous, current_passes, PASS_ACTION)
        current_board, current_player, current_previous, current_passes = next_state
        last_action = action

    return _soft_value(current_board, root_player)


def _rollout_action(board, player, previous_board, pass_count, legal, last_action, rng):
    moves = [action for action in legal if action != PASS_ACTION]
    if not moves:
        return PASS_ACTION
    legal_set = set(moves)

    urgent = _urgent_action(board, player, previous_board, legal_set)
    if urgent is not None:
        return urgent

    if pass_count == 1 and _score_margin(board, player) > 0:
        return PASS_ACTION
    if _empty_count(board) <= 7 and _score_margin(board, player) > 3:
        return PASS_ACTION

    actions = _candidate_actions(board, player, previous_board, legal, last_action, ROLLOUT_CANDIDATES)
    actions = [action for action in actions if action != PASS_ACTION]
    if not actions:
        return PASS_ACTION

    scored = [(_action_rank(board, player, previous_board, action, last_action), action) for action in actions]
    scored.sort(reverse=True)
    top = scored[: min(4, len(scored))]
    if rng.random() < 0.82:
        return top[0][1]
    return rng.choice([action for _, action in top])


def _candidate_actions(board, player, previous_board, legal, last_action, limit):
    legal_moves = [action for action in legal if action != PASS_ACTION]
    legal_set = set(legal_moves)
    if not legal_moves:
        return [PASS_ACTION]

    candidates: set[str] = set()
    tactical = _tactical_points(board, player)
    candidates.update(INDEX_TO_ACTION[index] for index in tactical if INDEX_TO_ACTION[index] in legal_set)

    if last_action in ACTION_TO_INDEX:
        center = ACTION_TO_INDEX[last_action]
        for index in _distance_points(center, 2):
            action = INDEX_TO_ACTION[index]
            if action in legal_set:
                candidates.add(action)

    empties = _empty_count(board)
    if empties > 62:
        candidates.update(INDEX_TO_ACTION[index] for index in STAR_POINTS if INDEX_TO_ACTION[index] in legal_set)

    for index, value in enumerate(board):
        if value != EMPTY:
            continue
        if any(board[neighbor] != EMPTY for neighbor in NEIGHBORS[index]):
            action = INDEX_TO_ACTION[index]
            if action in legal_set:
                candidates.add(action)

    if not candidates or empties <= 18:
        candidates.update(legal_moves)

    scored = sorted(
        ((_action_rank(board, player, previous_board, action, last_action), action) for action in candidates),
        reverse=True,
    )
    ranked = [action for score, action in scored if score > -300]
    if not ranked:
        ranked = [
            action
            for _, action in sorted(
                ((_action_rank(board, player, previous_board, action, last_action), action) for action in legal_moves),
                reverse=True,
            )
        ]

    ranked = ranked[:limit]
    if PASS_ACTION in legal and _pass_is_sensible(board, player, 0):
        ranked.append(PASS_ACTION)
    return ranked


def _urgent_action(board, player, previous_board, legal_set):
    best_action = None
    best_score = -1_000_000
    for action in legal_set:
        score = _urgent_score(board, player, previous_board, action)
        if score > best_score:
            best_score = score
            best_action = action
    if best_score >= 500:
        return best_action
    return None


def _urgent_score(board, player, previous_board, action):
    played = _play(board, player, action, previous_board)
    if played is None:
        return -1_000_000
    next_board, captured = played
    index = ACTION_TO_INDEX[action]
    _, liberties = _group(next_board, index)
    score = 0

    if captured:
        score += 440 + 150 * len(captured)
    score += _save_score(board, player, index)
    score += _attack_score(board, player, index)
    if len(liberties) == 1 and not captured:
        score -= 500
    if _is_true_eye(board, player, index):
        score -= 900
    return score


def _action_rank(board, player, previous_board, action, last_action):
    return _move_score(board, player, previous_board, action) + _local_response_bonus(board, player, action, last_action)


def _move_score(board, player, previous_board, action):
    if action == PASS_ACTION:
        return 10 if _pass_is_sensible(board, player, 0) else -420

    played = _play(board, player, action, previous_board)
    if played is None:
        return -1_000_000

    next_board, captured = played
    index = ACTION_TO_INDEX[action]
    group, liberties = _group(next_board, index)
    libs = len(liberties)
    empties = _empty_count(board)

    score = 0
    score += 125 * len(captured)
    score += 16 * min(libs, 8)
    score += 18 * _friendly_adjacent(board, player, index)
    score += 26 * _enemy_adjacent(board, player, index)
    score += _save_score(board, player, index)
    score += _attack_score(board, player, index)
    score += _connection_score(board, player, index)
    score += _shape_score(board, player, index)
    score += _opening_score(board, player, index)
    score -= _opponent_capture_penalty(next_board, player, board)
    score += 13 * (_safety_margin(next_board, player) - _safety_margin(board, player))
    if empties <= 52:
        score += _territory_delta(board, next_board, player)

    if libs == 1 and not captured:
        score -= 420
    elif libs == 2 and not captured:
        score -= 85
    if _is_true_eye(board, player, index):
        score -= 780
    if _fills_opponent_eye(board, player, index):
        score += 28
    if len(group) >= 4 and libs >= 4:
        score += 22
    if empties <= 16:
        score += 8 * (_score_margin(next_board, player) - _score_margin(board, player))
    return score


def _local_response_bonus(board, player, action, last_action):
    if action == PASS_ACTION or last_action not in ACTION_TO_INDEX:
        return 0
    index = ACTION_TO_INDEX[action]
    last_index = ACTION_TO_INDEX[last_action]
    last_row, last_col = divmod(last_index, BOARD_SIZE)
    row, col = divmod(index, BOARD_SIZE)
    distance = abs(row - last_row) + abs(col - last_col)
    if distance > 3:
        return 0

    bonus = 0
    empties = _empty_count(board)
    if distance == 1:
        bonus += 105 if empties > 44 else 58
    elif distance == 2:
        bonus += 52 if empties > 44 else 28
    elif distance == 3:
        bonus += 16

    last_value = board[last_index]
    if last_value == SYMBOLS[1 - player]:
        group, liberties = _group(board, last_index)
        if index in liberties:
            bonus += 45
        if len(group) >= 3 and index in _distance_points(last_index, 2):
            bonus += 28
    return bonus


def _tactical_points(board, player):
    mine = SYMBOLS[player]
    theirs = SYMBOLS[1 - player]
    points = set()
    seen = set()
    for index, value in enumerate(board):
        if value == EMPTY or index in seen:
            continue
        group, liberties = _group(board, index)
        seen.update(group)
        if value == theirs and len(liberties) <= 5:
            points.update(liberties)
        elif value == mine and len(liberties) <= 5:
            points.update(liberties)
    return points


def _save_score(board, player, index):
    mine = SYMBOLS[player]
    score = 0
    seen = set()
    for neighbor in NEIGHBORS[index]:
        if board[neighbor] != mine or neighbor in seen:
            continue
        group, liberties = _group(board, neighbor)
        seen.update(group)
        if index not in liberties:
            continue
        if len(liberties) == 1:
            score += 360 + 42 * len(group)
        elif len(liberties) == 2:
            score += 90 + 10 * len(group)
        elif len(liberties) == 3:
            score += 24 + 4 * len(group)
        elif len(liberties) == 4:
            score += 14 + 2 * len(group)
        elif len(liberties) == 5:
            score += 7 + len(group)
    return score


def _attack_score(board, player, index):
    theirs = SYMBOLS[1 - player]
    score = 0
    seen = set()
    for neighbor in NEIGHBORS[index]:
        if board[neighbor] != theirs or neighbor in seen:
            continue
        group, liberties = _group(board, neighbor)
        seen.update(group)
        if index not in liberties:
            continue
        if len(liberties) == 1:
            score += 420 + 55 * len(group)
        elif len(liberties) == 2:
            score += 100 + 12 * len(group)
        elif len(liberties) == 3:
            score += 28 + 4 * len(group)
        elif len(liberties) == 4:
            score += 18 + 3 * len(group)
        elif len(liberties) == 5:
            score += 10 + 2 * len(group)
        elif len(liberties) == 6:
            score += 5 + len(group)
    return score


def _connection_score(board, player, index):
    friend_groups = {}
    enemy_groups = {}
    for neighbor in NEIGHBORS[index]:
        value = board[neighbor]
        if value == EMPTY:
            continue
        group, _ = _group(board, neighbor)
        key = min(group)
        if value == SYMBOLS[player]:
            friend_groups[key] = len(group)
        else:
            enemy_groups[key] = len(group)

    score = 0
    if len(friend_groups) >= 2:
        score += 88 * (len(friend_groups) - 1) + 8 * sum(friend_groups.values())
    if len(enemy_groups) >= 2:
        score += 54 * (len(enemy_groups) - 1) + 4 * sum(enemy_groups.values())
    return score


def _opponent_capture_penalty(board, player, previous_board):
    opponent = 1 - player
    mine = SYMBOLS[player]
    best_capture = 0
    for index, value in enumerate(board):
        if value != EMPTY:
            continue
        if not any(board[neighbor] == mine for neighbor in NEIGHBORS[index]):
            continue
        played = _play(board, opponent, INDEX_TO_ACTION[index], previous_board)
        if played is not None:
            _, captured = played
            if len(captured) > best_capture:
                best_capture = len(captured)

    if best_capture == 0:
        return 0
    return 165 * best_capture + (55 if best_capture >= 2 else 0)


def _shape_score(board, player, index):
    row, col = divmod(index, BOARD_SIZE)
    friendly = _friendly_adjacent(board, player, index)
    enemy = _enemy_adjacent(board, player, index)
    friendly_diag = sum(board[neighbor] == SYMBOLS[player] for neighbor in DIAGONALS[index])
    score = 12 * friendly + 7 * enemy + 4 * friendly_diag

    if row in (0, 8) or col in (0, 8):
        score -= 5
    if row in (1, 7) or col in (1, 7):
        score += 5
    if 2 <= row <= 6 and 2 <= col <= 6:
        score += 5
    if index in STAR_POINTS:
        score += 10
    return score


def _opening_score(board, player, index):
    empties = _empty_count(board)
    if empties <= 56:
        return 0
    row, col = divmod(index, BOARD_SIZE)
    own = [stone for stone, value in enumerate(board) if value == SYMBOLS[player]]
    if not own:
        if index == 40:
            return 115
        return 72 if index in STAR_POINTS else 22 - 2 * (abs(row - 4) + abs(col - 4))

    nearest = min(abs(row - own_stone // BOARD_SIZE) + abs(col - own_stone % BOARD_SIZE) for own_stone in own)
    if nearest == 1:
        return -50
    if nearest in (2, 3, 4):
        return 28
    if index in STAR_POINTS:
        return 20
    return 0


def _territory_delta(board, next_board, player):
    before = _score_margin(board, player)
    after = _score_margin(next_board, player)
    return int(5 * (after - before))


def _friendly_adjacent(board, player, index):
    mine = SYMBOLS[player]
    return sum(board[neighbor] == mine for neighbor in NEIGHBORS[index])


def _enemy_adjacent(board, player, index):
    theirs = SYMBOLS[1 - player]
    return sum(board[neighbor] == theirs for neighbor in NEIGHBORS[index])


def _is_true_eye(board, player, index):
    mine = SYMBOLS[player]
    if board[index] != EMPTY:
        return False
    for neighbor in NEIGHBORS[index]:
        if board[neighbor] != mine:
            return False
    bad_diagonals = 0
    for diagonal in DIAGONALS[index]:
        if board[diagonal] == SYMBOLS[1 - player]:
            bad_diagonals += 1
    edge_bonus = 1 if len(DIAGONALS[index]) < 4 else 0
    return bad_diagonals <= edge_bonus


def _fills_opponent_eye(board, player, index):
    return _is_true_eye(board, 1 - player, index)


def _next_state(board, player, previous_board, pass_count, action):
    if action == PASS_ACTION:
        return board, 1 - player, board, pass_count + 1
    played = _play(board, player, action, previous_board)
    if played is None:
        return None
    next_board, _ = played
    return next_board, 1 - player, board, 0


def _legal_actions(board, player, previous_board):
    actions = []
    for index, value in enumerate(board):
        if value != EMPTY:
            continue
        action = INDEX_TO_ACTION[index]
        if _play(board, player, action, previous_board) is not None:
            actions.append(action)
    actions.append(PASS_ACTION)
    return actions


def _play(board, player, action, previous_board):
    if action == PASS_ACTION:
        return board, []
    index = ACTION_TO_INDEX.get(action)
    if index is None or board[index] != EMPTY:
        return None

    mine = SYMBOLS[player]
    theirs = SYMBOLS[1 - player]
    next_board = list(board)
    next_board[index] = mine
    captured = []
    seen = set()

    for neighbor in NEIGHBORS[index]:
        if next_board[neighbor] != theirs or neighbor in seen:
            continue
        group, liberties = _group("".join(next_board), neighbor)
        seen.update(group)
        if liberties:
            continue
        for stone in group:
            next_board[stone] = EMPTY
            captured.append(stone)

    next_text = "".join(next_board)
    _, liberties = _group(next_text, index)
    if not liberties:
        return None
    if previous_board is not None and next_text == previous_board:
        return None
    return next_text, captured


def _group(board, start):
    color = board[start]
    group = {start}
    liberties = set()
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in NEIGHBORS[current]:
            value = board[neighbor]
            if value == EMPTY:
                liberties.add(neighbor)
            elif value == color and neighbor not in group:
                group.add(neighbor)
                stack.append(neighbor)
    return group, liberties


def _is_terminal(board, pass_count):
    return pass_count >= 2 or EMPTY not in board


def _terminal_value(board, root_player):
    margin = _score_margin(board, root_player)
    if margin > 0:
        return 1.0
    if margin < 0:
        return -1.0
    return 0.0


def _soft_value(board, root_player):
    margin = _score_margin(board, root_player)
    safety = _safety_margin(board, root_player)
    return max(-1.0, min(1.0, (margin + 0.12 * safety) / 18.0))


def _terminal_score(board, player):
    margin = _final_score_margin(board, player)
    if margin > 0:
        return WIN_SCORE + 100 * margin
    if margin < 0:
        return -WIN_SCORE + 100 * margin
    return 0.0


def _evaluate(board, player):
    empties = _empty_count(board)
    margin = _score_margin(board, player)
    value = 112 * margin
    value += 12 * _safety_margin(board, player)
    value += 18 * _influence_margin(board, player)
    value += 42 * _eye_margin(board, player)
    if empties <= 24:
        value += 36 * _final_score_margin(board, player)
    return value


def _influence_margin(board, player):
    black_stones = [index for index, value in enumerate(board) if value == SYMBOLS[0]]
    white_stones = [index for index, value in enumerate(board) if value == SYMBOLS[1]]
    value = 0.0
    for index, point in enumerate(board):
        if point != EMPTY:
            continue
        row, col = divmod(index, BOARD_SIZE)
        black = _influence_at(row, col, black_stones)
        white = _influence_at(row, col, white_stones)
        if black > white:
            value += min(1.4, black - white)
        elif white > black:
            value -= min(1.4, white - black)
    return value if player == 0 else -value


def _influence_at(row, col, stones):
    total = 0.0
    for stone in stones:
        stone_row, stone_col = divmod(stone, BOARD_SIZE)
        distance = abs(row - stone_row) + abs(col - stone_col)
        if distance <= 4:
            total += (5 - distance) / 5
    return total


def _eye_margin(board, player):
    value = 0
    for index, point in enumerate(board):
        if point != EMPTY:
            continue
        if _is_true_eye(board, player, index):
            value += 1
        if _is_true_eye(board, 1 - player, index):
            value -= 1
    return value


def _score_margin(board, player):
    black_score, white_score = _score(board)
    return black_score - white_score if player == 0 else white_score - black_score


def _final_score_margin(board, player):
    black_score, white_score = _area_score(board)
    return black_score - white_score if player == 0 else white_score - black_score


def _score(board):
    black = board.count(SYMBOLS[0])
    white = board.count(SYMBOLS[1])
    territory = [0, 0]
    visited = set()
    empties = _empty_count(board)
    for index, value in enumerate(board):
        if value != EMPTY or index in visited:
            continue
        region = {index}
        borders = set()
        border_stones = {0: set(), 1: set()}
        touches_edge = False
        stack = [index]
        visited.add(index)
        while stack:
            current = stack.pop()
            row, col = divmod(current, BOARD_SIZE)
            if row in (0, BOARD_SIZE - 1) or col in (0, BOARD_SIZE - 1):
                touches_edge = True
            for neighbor in NEIGHBORS[current]:
                neighbor_value = board[neighbor]
                if neighbor_value == EMPTY and neighbor not in visited:
                    visited.add(neighbor)
                    region.add(neighbor)
                    stack.append(neighbor)
                elif neighbor_value == SYMBOLS[0]:
                    borders.add(0)
                    border_stones[0].add(neighbor)
                elif neighbor_value == SYMBOLS[1]:
                    borders.add(1)
                    border_stones[1].add(neighbor)
        if len(borders) == 1:
            owner = next(iter(borders))
            if _count_as_territory(region, border_stones[owner], touches_edge, empties):
                territory[owner] += len(region)
    return float(black + territory[0]), float(white + territory[1]) + KOMI


def _area_score(board):
    black = board.count(SYMBOLS[0])
    white = board.count(SYMBOLS[1])
    territory = [0, 0]
    visited = set()
    for index, value in enumerate(board):
        if value != EMPTY or index in visited:
            continue
        region = {index}
        borders = set()
        stack = [index]
        visited.add(index)
        while stack:
            current = stack.pop()
            for neighbor in NEIGHBORS[current]:
                neighbor_value = board[neighbor]
                if neighbor_value == EMPTY and neighbor not in visited:
                    visited.add(neighbor)
                    region.add(neighbor)
                    stack.append(neighbor)
                elif neighbor_value == SYMBOLS[0]:
                    borders.add(0)
                elif neighbor_value == SYMBOLS[1]:
                    borders.add(1)
        if len(borders) == 1:
            territory[next(iter(borders))] += len(region)
    return float(black + territory[0]), float(white + territory[1]) + KOMI


def _count_as_territory(region, border_stones, touches_edge, empties):
    size = len(region)
    if empties <= 12:
        return True
    if size <= 2 and len(border_stones) >= 2:
        return True
    if size <= 8 and len(border_stones) >= 4:
        return True
    if not touches_edge and size <= 16 and len(border_stones) >= 5:
        return True
    return False


def _safety_margin(board, player):
    mine = SYMBOLS[player]
    theirs = SYMBOLS[1 - player]
    value = 0
    seen = set()
    for index, stone in enumerate(board):
        if stone == EMPTY or index in seen:
            continue
        group, liberties = _group(board, index)
        seen.update(group)
        factor = 1 if stone == mine else -1
        if len(liberties) == 1:
            value -= factor * 8 * len(group)
        elif len(liberties) == 2:
            value += factor * 2 * len(group)
        else:
            value += factor * min(5, len(liberties))
    return value


def _pass_is_sensible(board, player, pass_count):
    margin = _final_score_margin(board, player)
    if pass_count == 1 and margin > 0:
        return True
    return _empty_count(board) <= 7 and margin > 4


def _empty_count(board):
    return board.count(EMPTY)


def _distance_points(center, distance):
    center_row, center_col = divmod(center, BOARD_SIZE)
    result = set()
    for row in range(max(0, center_row - distance), min(BOARD_SIZE, center_row + distance + 1)):
        for col in range(max(0, center_col - distance), min(BOARD_SIZE, center_col + distance + 1)):
            if abs(row - center_row) + abs(col - center_col) <= distance:
                result.add(row * BOARD_SIZE + col)
    return result


def _parse_board(rows):
    values = [EMPTY for _ in range(BOARD_POINTS)]
    for display_index, row_text in enumerate(rows):
        row = BOARD_SIZE - 1 - display_index
        for col, value in enumerate(row_text):
            values[row * BOARD_SIZE + col] = value
    return "".join(values)


def _time_budget(state):
    timeout = state.get("decision_timeout") or state.get("time_limit")
    if timeout:
        return max(0.05, float(timeout) - SAFETY_MARGIN_SECONDS)
    return TIME_LIMIT_SECONDS


def _seed_from_state(board, player, plies):
    value = 1469598103934665603
    for char in board:
        value ^= ord(char)
        value *= 1099511628211
        value &= (1 << 64) - 1
    return value ^ (player << 8) ^ plies
