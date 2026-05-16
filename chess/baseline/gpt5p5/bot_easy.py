"""Easy baseline chess bot for BoardArena/chess.

This is a one-ply heuristic bot. It is intentionally simple and fast.
"""

from __future__ import annotations

try:
    import chess
except ImportError:  # pragma: no cover - lets the judge surface a clear result.
    chess = None


name = "gpt5p5_easy"

PIECE_VALUES = {
    "p": 100,
    "n": 320,
    "b": 330,
    "r": 500,
    "q": 900,
    "k": 0,
}

CENTER_SQUARES = {"d4", "e4", "d5", "e5"}
NEAR_CENTER_SQUARES = {"c3", "d3", "e3", "f3", "c4", "f4", "c5", "f5", "c6", "d6", "e6", "f6"}


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        raise ValueError("no legal actions")
    if chess is None:
        return sorted(legal)[0]

    board = chess.Board(state["fen"])
    perspective = board.turn
    scored = []
    for action in legal:
        move = chess.Move.from_uci(action)
        score = _move_score(board, move, perspective)
        scored.append((score, action))

    best = max(score for score, _ in scored)
    candidates = [action for score, action in scored if score == best]
    return sorted(candidates)[0]


def _move_score(board, move, perspective):
    score = 0
    mover = board.piece_at(move.from_square)

    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured is None and board.is_en_passant(move):
            captured = chess.Piece(chess.PAWN, not board.turn)
        if captured is not None:
            score += 8 * _piece_value(captured)
        if mover is not None:
            score -= _piece_value(mover) // 12

    if move.promotion:
        score += _promotion_value(move.promotion)

    destination = chess.square_name(move.to_square)
    if destination in CENTER_SQUARES:
        score += 24
    elif destination in NEAR_CENTER_SQUARES:
        score += 10

    if mover and mover.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(move.from_square) in (0, 7):
        score += 16

    if board.is_castling(move):
        score += 35

    board.push(move)
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner == perspective:
            score += 1_000_000
        elif outcome.winner is None:
            score += 0
        else:
            score -= 1_000_000
    else:
        if board.is_check():
            score += 40
        score += _material_balance(board, perspective)
        score -= len(list(board.legal_moves))
        score -= _hanging_penalty(board, perspective)
    board.pop()
    return score


def _piece_value(piece):
    return PIECE_VALUES[piece.symbol().lower()]


def _promotion_value(piece_type):
    return {
        chess.QUEEN: 900,
        chess.ROOK: 500,
        chess.BISHOP: 330,
        chess.KNIGHT: 320,
    }.get(piece_type, 0)


def _material_balance(board, perspective):
    total = 0
    for piece in board.piece_map().values():
        value = _piece_value(piece)
        total += value if piece.color == perspective else -value
    return total


def _hanging_penalty(board, perspective):
    penalty = 0
    for square, piece in board.piece_map().items():
        if piece.color != perspective or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(not perspective, square)
        defenders = board.attackers(perspective, square)
        if attackers and not defenders:
            penalty += _piece_value(piece) // 3
    return penalty
