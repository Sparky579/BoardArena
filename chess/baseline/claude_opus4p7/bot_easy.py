"""Easy chess bot, roughly 800 ELO.

Strategy is intentionally shallow so the bot plays like a casual beginner:
  - always plays mate in 1 when it exists,
  - prefers captures, weighted by the value of the piece taken,
  - avoids moving a piece onto a square where it would be lost for less material
    (a one-square static exchange check, not a full SEE),
  - has light opening sense (develop minor pieces, discourage early king/queen
    sallies, mild center pull),
  - adds Gaussian-ish noise and an occasional pure random move so it is not
    deterministic and not too strong.
"""

import random

import chess


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}

_RNG = random.Random()


def _captured_value(board, move):
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    piece = board.piece_at(move.to_square)
    return PIECE_VALUES[piece.piece_type] if piece is not None else 0


def _score_move(board, move):
    score = 0.0
    score += _captured_value(board, move)

    if move.promotion is not None:
        score += PIECE_VALUES.get(move.promotion, 0) - PIECE_VALUES[chess.PAWN]

    if board.is_castling(move):
        score += 50

    mover = board.piece_at(move.from_square)
    mover_color = mover.color if mover is not None else board.turn
    fullmove = board.fullmove_number

    gives_check = False
    board.push(move)
    try:
        if board.is_checkmate():
            return 1_000_000.0
        if board.is_stalemate():
            return -1500.0
        gives_check = board.is_check()

        dest_piece = board.piece_at(move.to_square)
        if dest_piece is not None:
            our_color = dest_piece.color
            opp_color = not our_color
            new_value = PIECE_VALUES[dest_piece.piece_type]
            attackers = board.attackers(opp_color, move.to_square)
            if attackers:
                defenders = board.attackers(our_color, move.to_square)
                cheapest_attacker = min(
                    PIECE_VALUES[board.piece_at(sq).piece_type]
                    for sq in attackers
                )
                if not defenders:
                    score -= new_value
                elif cheapest_attacker < new_value:
                    score -= (new_value - cheapest_attacker)
    finally:
        board.pop()

    if gives_check:
        score += 8

    if fullmove <= 10 and mover is not None:
        back_rank = 0 if mover_color == chess.WHITE else 7
        from_rank = chess.square_rank(move.from_square)

        if mover.piece_type in (chess.KNIGHT, chess.BISHOP) and from_rank == back_rank:
            score += 18

        if mover.piece_type == chess.QUEEN and fullmove <= 4:
            score -= 25

        if mover.piece_type == chess.KING and not board.is_castling(move):
            score -= 35

        if (
            move.to_square in CENTER_SQUARES
            and mover.piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP)
        ):
            score += 10

    score += _RNG.uniform(-22, 22)
    return score


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        return ""

    if _RNG.random() < 0.03:
        return _RNG.choice(legal)

    board = chess.Board(state["fen"])

    best_score = -float("inf")
    best_moves = []
    for uci in legal:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if move not in board.legal_moves:
            continue
        s = _score_move(board, move)
        if s > best_score:
            best_score = s
            best_moves = [uci]
        elif s == best_score:
            best_moves.append(uci)

    if not best_moves:
        return legal[0]
    return _RNG.choice(best_moves)


class Bot:
    name = "claude_opus4p7_easy"

    def choose_action(self, state):
        return choose_action(state)
