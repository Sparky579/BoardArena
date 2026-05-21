"""CPU-only hard baseline bot for BoardArena 9x9 Go.

This is a pure rules-based Monte Carlo tree search bot with no learned model
and no external dependency. It uses UCT, Go-specific tactical priors, simple ko
simulation, and heuristic rollouts tuned for 9x9 games.
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
ROLLOUT_LIMIT = 220
EXPLORATION = 1.12

INDEX_TO_ACTION = tuple(f"{FILES[index % BOARD_SIZE]}{index // BOARD_SIZE + 1}" for index in range(BOARD_POINTS))
ACTION_TO_INDEX = {action: index for index, action in enumerate(INDEX_TO_ACTION)}
NEIGHBORS = tuple(
    tuple(
        next_index
        for next_row, next_col in (
            (index // BOARD_SIZE - 1, index % BOARD_SIZE),
            (index // BOARD_SIZE + 1, index % BOARD_SIZE),
            (index // BOARD_SIZE, index % BOARD_SIZE - 1),
            (index // BOARD_SIZE, index % BOARD_SIZE + 1),
        )
        if 0 <= next_row < BOARD_SIZE
        if 0 <= next_col < BOARD_SIZE
        for next_index in (next_row * BOARD_SIZE + next_col,)
    )
    for index in range(BOARD_POINTS)
)
DIAGONALS = tuple(
    tuple(
        next_index
        for next_row, next_col in (
            (index // BOARD_SIZE - 1, index % BOARD_SIZE - 1),
            (index // BOARD_SIZE - 1, index % BOARD_SIZE + 1),
            (index // BOARD_SIZE + 1, index % BOARD_SIZE - 1),
            (index // BOARD_SIZE + 1, index % BOARD_SIZE + 1),
        )
        if 0 <= next_row < BOARD_SIZE
        if 0 <= next_col < BOARD_SIZE
        for next_index in (next_row * BOARD_SIZE + next_col,)
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


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        raise ValueError("no legal actions")
    moves = [action for action in legal if action != PASS_ACTION]
    if not moves:
        return PASS_ACTION
    if len(moves) == 1 and PASS_ACTION not in legal:
        return moves[0]

    board = _parse_board(state["board"])
    player = state["actor"]
    rng = random.Random(_seed_from_state(board, player, state["plies"]))
    deadline = time.perf_counter() + TIME_LIMIT_SECONDS

    ranked_root = _ranked_actions(board, player, None, legal, state.get("last_move"))
    fallback = ranked_root[0] if ranked_root else PASS_ACTION
    root = Node(board, player, None, state.get("pass_count", 0), legal=ranked_root)
    root_player = player
    simulations = 0

    while time.perf_counter() < deadline:
        node = root
        path = [node]

        while not node.untried and node.children:
            node = _select_child(node, root_player)
            path.append(node)

        if node.untried and time.perf_counter() < deadline:
            action = node.untried.pop(0)
            next_state = _next_state(node.board, node.player, node.previous_board, node.pass_count, action)
            if next_state is not None:
                next_board, next_player, next_previous, next_pass_count = next_state
                legal_child = [] if _is_terminal(next_board, next_pass_count) else _ranked_actions(
                    next_board,
                    next_player,
                    next_previous,
                    _legal_actions(next_board, next_player, next_previous),
                    action,
                )
                child = Node(
                    next_board,
                    next_player,
                    next_previous,
                    next_pass_count,
                    parent=node,
                    action=action,
                    legal=legal_child,
                    prior=_prior_score(node.board, node.player, node.previous_board, action),
                )
                node.children[action] = child
                node = child
                path.append(node)

        value = _rollout(node.board, node.player, node.previous_board, node.pass_count, root_player, rng, deadline)
        for item in path:
            item.visits += 1
            item.value += value
        simulations += 1

    if not root.children:
        return fallback

    best = max(root.children.values(), key=lambda child: (child.visits, child.value / max(1, child.visits), child.prior))
    if best.action == PASS_ACTION and moves and not _pass_is_sensible(board, player, root.pass_count):
        non_pass_children = [child for child in root.children.values() if child.action != PASS_ACTION]
        if non_pass_children:
            best = max(non_pass_children, key=lambda child: (child.visits, child.value / max(1, child.visits), child.prior))
    return best.action


def _select_child(node, root_player):
    log_visits = math.log(node.visits + 1)

    def ucb(child):
        if child.visits == 0:
            return float("inf")
        mean = child.value / child.visits
        if node.player != root_player:
            mean = -mean
        explore = EXPLORATION * math.sqrt(log_visits / child.visits)
        prior = 0.18 * child.prior / (1 + child.visits)
        return mean + explore + prior

    return max(node.children.values(), key=ucb)


def _rollout(board, player, previous_board, pass_count, root_player, rng, deadline):
    current_board = board
    current_player = player
    current_previous = previous_board
    current_passes = pass_count
    last_action = None

    for _ in range(ROLLOUT_LIMIT):
        if _is_terminal(current_board, current_passes):
            return _result_value(current_board, root_player)
        if time.perf_counter() >= deadline:
            return _soft_value(current_board, root_player)

        legal = _legal_actions(current_board, current_player, current_previous)
        action = _rollout_action(current_board, current_player, current_previous, current_passes, legal, last_action, rng)
        next_state = _next_state(current_board, current_player, current_previous, current_passes, action)
        if next_state is None:
            action = PASS_ACTION
            next_state = _next_state(current_board, current_player, current_previous, current_passes, action)
        current_board, current_player, current_previous, current_passes = next_state
        last_action = action

    return _soft_value(current_board, root_player)


def _rollout_action(board, player, previous_board, pass_count, legal, last_action, rng):
    moves = [action for action in legal if action != PASS_ACTION]
    if not moves:
        return PASS_ACTION
    if pass_count == 1 and _score_margin(board, player) > 0:
        return PASS_ACTION
    if _empty_count(board) <= 8 and _score_margin(board, player) > 3:
        return PASS_ACTION

    sampled = moves
    if last_action in ACTION_TO_INDEX:
        local = _local_moves(board, player, previous_board, ACTION_TO_INDEX[last_action], set(moves))
        if local:
            sampled = local + [action for action in moves if action not in local]

    scored = []
    for action in sampled[:24]:
        score = _prior_score(board, player, previous_board, action)
        if rng.random() < 0.08:
            score += rng.uniform(-18, 18)
        scored.append((score, action))
    scored.sort(reverse=True)
    top = scored[: min(5, len(scored))]
    weights = [max(1.0, score - top[-1][0] + 1.0) for score, _ in top]
    total = sum(weights)
    pick = rng.random() * total
    running = 0.0
    for weight, (_, action) in zip(weights, top):
        running += weight
        if pick <= running:
            return action
    return top[0][1]


def _ranked_actions(board, player, previous_board, legal, last_action=None):
    actions = list(legal)
    actions.sort(key=lambda action: _prior_score(board, player, previous_board, action, last_action), reverse=True)
    return actions


def _prior_score(board, player, previous_board, action, last_action=None):
    if action == PASS_ACTION:
        return 12 if _pass_is_sensible(board, player, 0) else -140

    played = _play(board, player, action, previous_board)
    if played is None:
        return -10_000
    next_board, captured = played
    index = ACTION_TO_INDEX[action]
    group, liberties = _group(next_board, index)
    score = 0
    score += 75 * len(captured)
    score += 9 * min(len(liberties), 8)
    score -= 65 if len(liberties) == 1 and not captured else 0
    score += 18 if len(liberties) >= 4 else 0
    score += _attack_and_save_score(board, player, index)
    score += _shape_score(board, player, index)
    score += _opening_score(index, board.count(EMPTY))
    score += _development_score(board, player, index)
    if last_action in ACTION_TO_INDEX and index in _distance_two(ACTION_TO_INDEX[last_action]):
        score += 8
    return score


def _attack_and_save_score(board, player, index):
    mine = SYMBOLS[player]
    theirs = SYMBOLS[1 - player]
    score = 0
    seen = set()
    for neighbor in NEIGHBORS[index]:
        value = board[neighbor]
        if value == EMPTY or neighbor in seen:
            continue
        group, liberties = _group(board, neighbor)
        seen.update(group)
        if value == theirs:
            if len(liberties) == 1 and index in liberties:
                score += 85 + 8 * len(group)
            elif len(liberties) == 2 and index in liberties:
                score += 24
            elif len(liberties) <= 4 and index in liberties:
                score += 10 + 3 * len(group)
        elif value == mine:
            if len(liberties) == 1 and index in liberties:
                score += 70 + 5 * len(group)
            elif len(liberties) == 2 and index in liberties:
                score += 18
    return score


def _shape_score(board, player, index):
    mine = SYMBOLS[player]
    theirs = SYMBOLS[1 - player]
    friendly = sum(board[n] == mine for n in NEIGHBORS[index])
    enemy = sum(board[n] == theirs for n in NEIGHBORS[index])
    empty = sum(board[n] == EMPTY for n in NEIGHBORS[index])
    diagonals = sum(board[n] == mine for n in DIAGONALS[index])
    score = 9 * friendly + 5 * enemy + 3 * empty + 3 * diagonals
    if _is_likely_eye(board, player, index):
        score -= 90
    row, col = divmod(index, BOARD_SIZE)
    if row in (0, 8) or col in (0, 8):
        score -= 3
    if row in (1, 7) or col in (1, 7):
        score += 2
    return score


def _opening_score(index, empties):
    if empties < 62:
        return 0
    if index in STAR_POINTS:
        return 45
    row, col = divmod(index, BOARD_SIZE)
    distance = abs(row - 4) + abs(col - 4)
    return 15 - 2 * distance


def _development_score(board, player, index):
    empties = board.count(EMPTY)
    if empties <= 52:
        return 0

    mine = SYMBOLS[player]
    own = [stone for stone, value in enumerate(board) if value == mine]
    if not own:
        return 0

    row, col = divmod(index, BOARD_SIZE)
    distances = []
    for stone in own:
        stone_row, stone_col = divmod(stone, BOARD_SIZE)
        distances.append(abs(row - stone_row) + abs(col - stone_col))
    nearest = min(distances)
    score = 0
    if empties > 66:
        if nearest == 1:
            score -= 42
        elif nearest in (2, 3, 4):
            score += 24
        elif nearest >= 6:
            score -= 6
    else:
        if nearest == 1:
            score -= 16
        elif nearest in (2, 3):
            score += 12
    return score


def _local_moves(board, player, previous_board, center, legal_set):
    result = []
    center_row, center_col = divmod(center, BOARD_SIZE)
    for action in legal_set:
        index = ACTION_TO_INDEX[action]
        row, col = divmod(index, BOARD_SIZE)
        if abs(row - center_row) + abs(col - center_col) <= 2:
            result.append(action)
    result.sort(key=lambda action: _prior_score(board, player, previous_board, action), reverse=True)
    return result


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


def _result_value(board, root_player):
    margin = _score_margin(board, root_player)
    if margin > 0:
        return 1.0
    if margin < 0:
        return -1.0
    return 0.0


def _soft_value(board, root_player):
    margin = _score_margin(board, root_player)
    return max(-1.0, min(1.0, margin / 18.0))


def _score_margin(board, player):
    black_score, white_score = _score(board)
    return black_score - white_score if player == 0 else white_score - black_score


def _score(board):
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


def _pass_is_sensible(board, player, pass_count):
    if pass_count == 1 and _score_margin(board, player) > 0:
        return True
    return _empty_count(board) <= 8 and _score_margin(board, player) > 4


def _empty_count(board):
    return board.count(EMPTY)


def _is_likely_eye(board, player, index):
    mine = SYMBOLS[player]
    if any(board[neighbor] != mine for neighbor in NEIGHBORS[index]):
        return False
    diagonals = DIAGONALS[index]
    if not diagonals:
        return True
    friendly = sum(board[neighbor] == mine for neighbor in diagonals)
    return friendly >= max(1, len(diagonals) - 1)


def _distance_two(center):
    center_row, center_col = divmod(center, BOARD_SIZE)
    result = set()
    for row in range(max(0, center_row - 2), min(BOARD_SIZE, center_row + 3)):
        for col in range(max(0, center_col - 2), min(BOARD_SIZE, center_col + 3)):
            if abs(row - center_row) + abs(col - center_col) <= 2:
                result.add(row * BOARD_SIZE + col)
    return result


def _parse_board(rows):
    values = [EMPTY for _ in range(BOARD_POINTS)]
    for display_index, row_text in enumerate(rows):
        row = BOARD_SIZE - 1 - display_index
        for col, value in enumerate(row_text):
            values[row * BOARD_SIZE + col] = value
    return "".join(values)


def _seed_from_state(board, player, plies):
    value = 1469598103934665603
    for char in board:
        value ^= ord(char)
        value *= 1099511628211
        value &= (1 << 64) - 1
    return value ^ (player << 8) ^ plies
