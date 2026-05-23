"""CPU-only hard baseline bot for BoardArena Othello.

The bot uses bitboards, iterative deepening alpha-beta search, a small
transposition table, and a stage-aware handcrafted evaluation. It keeps its own
deadline below the referee's usual 2 second decision timeout.
"""

from __future__ import annotations

import time


name = "gpt5p5_othello_hard"

PASS_ACTION = "PASS"
BOARD_SIZE = 8
MASK_64 = (1 << 64) - 1
NOT_A_FILE = 0xFEFEFEFEFEFEFEFE
NOT_H_FILE = 0x7F7F7F7F7F7F7F7F
CORNER_MASK = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)
TIME_LIMIT_SECONDS = 1.82
SAFETY_MARGIN_SECONDS = 0.15
WIN_SCORE = 1_000_000
INF = 10_000_000
TT_MAX_SIZE = 220_000

FILES = "abcdefgh"
SYMBOLS = ("B", "W")

POSITION_WEIGHTS = (
    120, -25, 20, 5, 5, 20, -25, 120,
    -25, -55, -8, -8, -8, -8, -55, -25,
    20, -8, 16, 4, 4, 16, -8, 20,
    5, -8, 4, 2, 2, 4, -8, 5,
    5, -8, 4, 2, 2, 4, -8, 5,
    20, -8, 16, 4, 4, 16, -8, 20,
    -25, -55, -8, -8, -8, -8, -55, -25,
    120, -25, 20, 5, 5, 20, -25, 120,
)

INDEX_TO_ACTION = tuple(f"{FILES[index % 8]}{index // 8 + 1}" for index in range(64))
ACTION_TO_BIT = {action: 1 << index for index, action in enumerate(INDEX_TO_ACTION)}

X_SQUARES = {
    0: (1 << 9),
    7: (1 << 14),
    56: (1 << 49),
    63: (1 << 54),
}
C_SQUARES = {
    0: (1 << 1) | (1 << 8),
    7: (1 << 6) | (1 << 15),
    56: (1 << 48) | (1 << 57),
    63: (1 << 55) | (1 << 62),
}
CORNER_BITS = tuple(1 << index for index in (0, 7, 56, 63))
EDGE_RAYS = (
    (0, (tuple(range(1, 8)), tuple(range(8, 64, 8)))),
    (7, (tuple(range(6, -1, -1)), tuple(range(15, 64, 8)))),
    (56, (tuple(range(57, 64)), tuple(range(48, -1, -8)))),
    (63, (tuple(range(62, 55, -1)), tuple(range(55, -1, -8)))),
)

_TT: dict[tuple[int, int], tuple[int, int, int, int]] = {}
_deadline = 0.0
_nodes = 0


class SearchTimeout(Exception):
    pass


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        raise ValueError("no legal actions")
    if legal == [PASS_ACTION]:
        return PASS_ACTION
    if len(legal) == 1:
        return legal[0]

    black, white = _parse_board(state["board"])
    player = state["actor"]
    me, opp = (black, white) if player == 0 else (white, black)
    root_legal = [ACTION_TO_BIT[action] for action in legal if action != PASS_ACTION]
    empties = 64 - (me | opp).bit_count()

    global _deadline, _nodes
    _deadline = time.perf_counter() + _time_budget(state)
    _nodes = 0

    best_move = _fallback_move(me, opp, root_legal)
    max_depth = empties + 2 if empties <= 14 else 64

    for depth in range(1, max_depth + 1):
        try:
            score, move = _search_root(me, opp, depth, best_move)
        except SearchTimeout:
            break
        if move:
            best_move = move
        if abs(score) >= WIN_SCORE - 128:
            break
        if time.perf_counter() >= _deadline:
            break

    _trim_tt()
    return INDEX_TO_ACTION[best_move.bit_length() - 1]


def _search_root(me, opp, depth, previous_best):
    alpha = -INF
    beta = INF
    best_score = -INF
    best_move = previous_best
    moves = _ordered_moves(me, opp, depth, previous_best)

    for move in moves:
        _check_time()
        flips = _flips_for_move(move, me, opp)
        next_me = me | move | flips
        next_opp = opp & ~flips
        score = -_negamax(next_opp, next_me, depth - 1, -beta, -alpha)
        if score > best_score or (score == best_score and _move_tiebreak(move) > _move_tiebreak(best_move)):
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

    return best_score, best_move


def _negamax(me, opp, depth, alpha, beta):
    global _nodes
    _nodes += 1
    if (_nodes & 1023) == 0:
        _check_time()

    moves = _legal_moves(me, opp)
    if not moves:
        if not _legal_moves(opp, me):
            return _terminal_score(me, opp)
        return -_negamax(opp, me, depth, -beta, -alpha)

    if depth <= 0:
        return _evaluate(me, opp)

    key = (me, opp)
    alpha_start = alpha
    entry = _TT.get(key)
    tt_move = 0
    if entry is not None:
        entry_depth, entry_value, entry_flag, entry_move = entry
        tt_move = entry_move
        if entry_depth >= depth:
            if entry_flag == 0:
                return entry_value
            if entry_flag < 0 and entry_value <= alpha:
                return entry_value
            if entry_flag > 0 and entry_value >= beta:
                return entry_value

    best_value = -INF
    best_move = 0
    for move in _ordered_moves(me, opp, depth, tt_move):
        flips = _flips_for_move(move, me, opp)
        next_me = me | move | flips
        next_opp = opp & ~flips
        value = -_negamax(next_opp, next_me, depth - 1, -beta, -alpha)
        if value > best_value:
            best_value = value
            best_move = move
        if value > alpha:
            alpha = value
        if alpha >= beta:
            break

    if best_value <= alpha_start:
        flag = -1
    elif best_value >= beta:
        flag = 1
    else:
        flag = 0
    _TT[key] = (depth, best_value, flag, best_move)
    return best_value


def _ordered_moves(me, opp, depth, preferred):
    moves = list(_iter_bits(_legal_moves(me, opp)))
    moves.sort(key=lambda move: _move_order_score(move, me, opp, preferred), reverse=True)
    return moves


def _move_order_score(move, me, opp, preferred):
    if move == preferred:
        return 1_000_000

    flips = _flips_for_move(move, me, opp)
    next_me = me | move | flips
    next_opp = opp & ~flips
    index = move.bit_length() - 1
    score = POSITION_WEIGHTS[index] * 10 + flips.bit_count() * 4

    if move & CORNER_MASK:
        score += 20_000
    score += _corner_danger_score(move, me, opp)

    opponent_moves = _legal_moves(next_opp, next_me)
    if opponent_moves & CORNER_MASK:
        score -= 7_500
    score -= opponent_moves.bit_count() * 35
    if not opponent_moves:
        score += 1_500
    return score


def _fallback_move(me, opp, legal_moves):
    return max(legal_moves, key=lambda move: _move_order_score(move, me, opp, 0))


def _move_tiebreak(move):
    if not move:
        return -INF
    index = move.bit_length() - 1
    return POSITION_WEIGHTS[index]


def _legal_moves(me, opp):
    empty = (~(me | opp)) & MASK_64
    moves = 0
    for shift in _SHIFTS:
        candidates = shift(me) & opp
        while candidates:
            moves |= shift(candidates) & empty
            candidates = shift(candidates) & opp
    return moves


def _flips_for_move(move, me, opp):
    flips = 0
    for shift in _SHIFTS:
        captured = 0
        candidates = shift(move) & opp
        while candidates:
            captured |= candidates
            next_bits = shift(candidates)
            if next_bits & me:
                flips |= captured
                break
            candidates = next_bits & opp
    return flips


def _evaluate(me, opp):
    occupied = me | opp
    empties = 64 - occupied.bit_count()
    my_count = me.bit_count()
    opp_count = opp.bit_count()
    disc_diff = my_count - opp_count

    if empties <= 8:
        return disc_diff * 160

    my_moves = _legal_moves(me, opp).bit_count()
    opp_moves = _legal_moves(opp, me).bit_count()
    mobility_diff = my_moves - opp_moves

    corner_diff = (me & CORNER_MASK).bit_count() - (opp & CORNER_MASK).bit_count()
    stable_diff = _stable_edge_count(me) - _stable_edge_count(opp)
    position_diff = _weighted_sum(me) - _weighted_sum(opp)
    frontier_diff = _frontier_count(me, occupied) - _frontier_count(opp, occupied)
    potential_diff = _potential_mobility(me, opp, occupied) - _potential_mobility(opp, me, occupied)
    danger = _corner_exposure_score(me, opp, occupied)

    if empties > 42:
        return (
            corner_diff * 900
            + mobility_diff * 80
            + potential_diff * 18
            + stable_diff * 80
            + position_diff * 5
            - frontier_diff * 28
            - disc_diff * 8
            + danger
        )
    if empties > 18:
        return (
            corner_diff * 1050
            + mobility_diff * 65
            + potential_diff * 10
            + stable_diff * 95
            + position_diff * 6
            - frontier_diff * 20
            + disc_diff * 4
            + danger
        )
    return (
        corner_diff * 1200
        + mobility_diff * 35
        + stable_diff * 115
        + position_diff * 4
        - frontier_diff * 10
        + disc_diff * 32
        + danger
    )


def _terminal_score(me, opp):
    diff = me.bit_count() - opp.bit_count()
    if diff > 0:
        return WIN_SCORE + diff
    if diff < 0:
        return -WIN_SCORE + diff
    return 0


def _weighted_sum(bits):
    total = 0
    while bits:
        bit = bits & -bits
        total += POSITION_WEIGHTS[bit.bit_length() - 1]
        bits ^= bit
    return total


def _frontier_count(bits, occupied):
    empty = (~occupied) & MASK_64
    return (bits & _neighbors(empty)).bit_count()


def _potential_mobility(me, opp, occupied):
    empty = (~occupied) & MASK_64
    return (_neighbors(opp) & empty).bit_count()


def _stable_edge_count(bits):
    stable = 0
    for corner_index, rays in EDGE_RAYS:
        corner = 1 << corner_index
        if not bits & corner:
            continue
        stable |= corner
        for ray in rays:
            for index in ray:
                if not bits & (1 << index):
                    break
                stable |= 1 << index
    return stable.bit_count()


def _corner_exposure_score(me, opp, occupied):
    score = 0
    for corner_index, corner in zip((0, 7, 56, 63), CORNER_BITS):
        if occupied & corner:
            continue
        x_mask = X_SQUARES[corner_index]
        c_mask = C_SQUARES[corner_index]
        score -= (me & x_mask).bit_count() * 420
        score -= (me & c_mask).bit_count() * 170
        score += (opp & x_mask).bit_count() * 360
        score += (opp & c_mask).bit_count() * 130
    return score


def _corner_danger_score(move, me, opp):
    occupied = me | opp
    score = 0
    for corner_index, corner in zip((0, 7, 56, 63), CORNER_BITS):
        if occupied & corner:
            continue
        if move & X_SQUARES[corner_index]:
            score -= 5_000
        if move & C_SQUARES[corner_index]:
            score -= 2_000
    return score


def _neighbors(bits):
    neighbors = 0
    for shift in _SHIFTS:
        neighbors |= shift(bits)
    return neighbors


def _iter_bits(bits):
    while bits:
        bit = bits & -bits
        yield bit
        bits ^= bit


def _parse_board(rows):
    black = 0
    white = 0
    for display_index, row_text in enumerate(rows):
        row = BOARD_SIZE - 1 - display_index
        for col, symbol in enumerate(row_text):
            bit = 1 << (row * BOARD_SIZE + col)
            if symbol == SYMBOLS[0]:
                black |= bit
            elif symbol == SYMBOLS[1]:
                white |= bit
    return black, white


def _check_time():
    if time.perf_counter() >= _deadline:
        raise SearchTimeout


def _time_budget(state):
    timeout = state.get("decision_timeout") or state.get("time_limit")
    if timeout:
        return max(0.05, float(timeout) - SAFETY_MARGIN_SECONDS)
    return TIME_LIMIT_SECONDS


def _trim_tt():
    if len(_TT) <= TT_MAX_SIZE:
        return
    remove_count = len(_TT) - TT_MAX_SIZE
    for index, key in enumerate(list(_TT)):
        if index >= remove_count:
            break
        _TT.pop(key, None)


def _shift_n(bits):
    return (bits << 8) & MASK_64


def _shift_s(bits):
    return bits >> 8


def _shift_e(bits):
    return (bits & NOT_H_FILE) << 1


def _shift_w(bits):
    return (bits & NOT_A_FILE) >> 1


def _shift_ne(bits):
    return ((bits & NOT_H_FILE) << 9) & MASK_64


def _shift_nw(bits):
    return ((bits & NOT_A_FILE) << 7) & MASK_64


def _shift_se(bits):
    return (bits & NOT_H_FILE) >> 7


def _shift_sw(bits):
    return (bits & NOT_A_FILE) >> 9


_SHIFTS = (
    _shift_n,
    _shift_s,
    _shift_e,
    _shift_w,
    _shift_ne,
    _shift_nw,
    _shift_se,
    _shift_sw,
)
