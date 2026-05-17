"""Medium BoardArena chess bot.

This bot is still compact, but it searches instead of only scoring legal moves:
iterative deepening, alpha-beta, capture quiescence, move ordering, and a
material/position evaluation. It has no opening book or tablebase.
"""

from __future__ import annotations

import time

try:
    import chess
except ImportError:  # pragma: no cover - lets the judge surface a clear result.
    chess = None


name = "gpt5p5_medium"

MAX_TIME_SECONDS = 1.2
MAX_DEPTH = 4
QUIESCENCE_DEPTH = 4
INF = 10_000_000
MATE = 1_000_000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
} if chess else {}

PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 12, 15, 15, 12, 5, -30,
    -30, 0, 15, 22, 22, 15, 0, -30,
    -30, 5, 15, 22, 22, 15, 5, -30,
    -30, 0, 12, 15, 15, 12, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 12, 12, 10, 0, -10,
    -10, 5, 5, 12, 12, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]

QUEEN_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]

KING_MID_TABLE = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]

KING_END_TABLE = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -10, 0, 0, 0, 0, -10, -30,
    -30, 0, 20, 30, 30, 20, 0, -30,
    -30, 0, 30, 40, 40, 30, 0, -30,
    -30, 0, 30, 40, 40, 30, 0, -30,
    -30, 0, 20, 30, 30, 20, 0, -30,
    -30, -10, 0, 0, 0, 0, -10, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

PIECE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
} if chess else {}


class SearchTimeout(Exception):
    pass


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        raise ValueError("no legal actions")
    if chess is None:
        return sorted(legal)[0]

    board = chess.Board(state["fen"])
    perspective = board.turn
    deadline = time.monotonic() + MAX_TIME_SECONDS
    transposition = {}

    best_move = _fallback_move(board, legal)
    best_score = -INF
    ordered = _ordered_moves(board, list(board.legal_moves), None)

    try:
        for depth in range(1, MAX_DEPTH + 1):
            depth_best = best_move
            depth_score = -INF
            for move in ordered:
                _check_time(deadline)
                board.push(move)
                score = _search(board, depth - 1, -INF, INF, perspective, deadline, transposition, 1)
                board.pop()
                if score > depth_score or (score == depth_score and move.uci() < depth_best):
                    depth_score = score
                    depth_best = move.uci()
            best_move = depth_best
            best_score = depth_score
            ordered = _ordered_moves(board, list(board.legal_moves), chess.Move.from_uci(best_move))
    except SearchTimeout:
        pass

    return best_move if best_score > -INF else _fallback_move(board, legal)


def _search(board, depth, alpha, beta, perspective, deadline, transposition, ply):
    _check_time(deadline)
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return _terminal_score(outcome, perspective, ply)

    key = (board.transposition_key() if hasattr(board, "transposition_key") else board._transposition_key(), depth, perspective)
    cached = transposition.get(key)
    if cached is not None:
        return cached

    if depth <= 0:
        return _quiescence(board, alpha, beta, perspective, deadline, QUIESCENCE_DEPTH, ply)

    maximizing = board.turn == perspective
    moves = _ordered_moves(board, list(board.legal_moves), None)

    if maximizing:
        value = -INF
        for move in moves:
            board.push(move)
            value = max(value, _search(board, depth - 1, alpha, beta, perspective, deadline, transposition, ply + 1))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = INF
        for move in moves:
            board.push(move)
            value = min(value, _search(board, depth - 1, alpha, beta, perspective, deadline, transposition, ply + 1))
            board.pop()
            beta = min(beta, value)
            if alpha >= beta:
                break

    if len(transposition) < 100_000:
        transposition[key] = value
    return value


def _quiescence(board, alpha, beta, perspective, deadline, depth, ply):
    _check_time(deadline)
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return _terminal_score(outcome, perspective, ply)

    stand_pat = _evaluate(board, perspective)
    if depth <= 0:
        return stand_pat

    maximizing = board.turn == perspective
    tactical_moves = [
        move for move in board.legal_moves
        if board.is_capture(move) or move.promotion or board.gives_check(move)
    ]
    tactical_moves = _ordered_moves(board, tactical_moves, None)

    if maximizing:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        value = stand_pat
        for move in tactical_moves:
            board.push(move)
            value = max(value, _quiescence(board, alpha, beta, perspective, deadline, depth - 1, ply + 1))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    if stand_pat <= alpha:
        return alpha
    beta = min(beta, stand_pat)
    value = stand_pat
    for move in tactical_moves:
        board.push(move)
        value = min(value, _quiescence(board, alpha, beta, perspective, deadline, depth - 1, ply + 1))
        board.pop()
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def _evaluate(board, perspective):
    score = 0
    phase_material = 0
    bishops = {chess.WHITE: 0, chess.BLACK: 0}

    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == perspective else -1
        value = PIECE_VALUES[piece.piece_type]
        score += sign * value
        if piece.piece_type != chess.KING:
            phase_material += value
        if piece.piece_type == chess.BISHOP:
            bishops[piece.color] += 1

        table_square = square if piece.color == chess.WHITE else chess.square_mirror(square)
        if piece.piece_type == chess.KING:
            king_table = KING_END_TABLE if phase_material <= 2600 else KING_MID_TABLE
            score += sign * king_table[table_square]
        else:
            score += sign * PIECE_TABLES[piece.piece_type][table_square]

    for color in (chess.WHITE, chess.BLACK):
        if bishops[color] >= 2:
            score += 35 if color == perspective else -35

    score += _pawn_structure(board, perspective)
    score += _king_safety(board, perspective)

    turn = board.turn
    board.turn = perspective
    own_mobility = board.legal_moves.count()
    board.turn = not perspective
    opp_mobility = board.legal_moves.count()
    board.turn = turn
    score += 3 * (own_mobility - opp_mobility)

    return score


def _pawn_structure(board, perspective):
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == perspective else -1
        files = [0 for _ in range(8)]
        pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        for square in pawns:
            files[chess.square_file(square)] += 1

        for file_index, count in enumerate(files):
            if count > 1:
                score -= sign * 12 * (count - 1)

        for square in pawns:
            file_index = chess.square_file(square)
            rank = chess.square_rank(square)
            adjacent_files = [idx for idx in (file_index - 1, file_index + 1) if 0 <= idx < 8]
            if all(files[idx] == 0 for idx in adjacent_files):
                score -= sign * 10
            if _is_passed_pawn(square, color, enemy_pawns):
                advance = rank if color == chess.WHITE else 7 - rank
                score += sign * (15 + advance * 8)
    return score


def _is_passed_pawn(square, color, enemy_pawns):
    file_index = chess.square_file(square)
    rank = chess.square_rank(square)
    candidate_files = [idx for idx in (file_index - 1, file_index, file_index + 1) if 0 <= idx < 8]
    for enemy in enemy_pawns:
        if chess.square_file(enemy) not in candidate_files:
            continue
        enemy_rank = chess.square_rank(enemy)
        if color == chess.WHITE and enemy_rank > rank:
            return False
        if color == chess.BLACK and enemy_rank < rank:
            return False
    return True


def _king_safety(board, perspective):
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        king = board.king(color)
        if king is None:
            continue
        sign = 1 if color == perspective else -1
        attackers = len(board.attackers(not color, king))
        score -= sign * 35 * attackers
        shield = 0
        rank_dir = 1 if color == chess.WHITE else -1
        king_file = chess.square_file(king)
        king_rank = chess.square_rank(king)
        for file_index in (king_file - 1, king_file, king_file + 1):
            shield_rank = king_rank + rank_dir
            if 0 <= file_index < 8 and 0 <= shield_rank < 8:
                piece = board.piece_at(chess.square(file_index, shield_rank))
                if piece and piece.color == color and piece.piece_type == chess.PAWN:
                    shield += 1
        score += sign * shield * 10
    return score


def _ordered_moves(board, moves, preferred):
    return sorted(moves, key=lambda move: _move_order_score(board, move, preferred), reverse=True)


def _move_order_score(board, move, preferred):
    if preferred is not None and move == preferred:
        return 1_000_000
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        if victim is None and board.is_en_passant(move):
            victim = chess.Piece(chess.PAWN, not board.turn)
        attacker = board.piece_at(move.from_square)
        if victim:
            score += 10_000 + 10 * PIECE_VALUES[victim.piece_type]
        if attacker:
            score -= PIECE_VALUES[attacker.piece_type]
    if move.promotion:
        score += 8_000 + PIECE_VALUES.get(move.promotion, 0)
    if board.gives_check(move):
        score += 2_000
    if board.is_castling(move):
        score += 500
    mover = board.piece_at(move.from_square)
    if mover and mover.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(move.from_square) in (0, 7):
        score += 80
    return score


def _terminal_score(outcome, perspective, ply):
    if outcome.winner is None:
        return 0
    if outcome.winner == perspective:
        return MATE - ply
    return -MATE + ply


def _fallback_move(board, legal):
    moves = _ordered_moves(board, [chess.Move.from_uci(action) for action in legal], None)
    return moves[0].uci()


def _check_time(deadline):
    if time.monotonic() >= deadline:
        raise SearchTimeout
