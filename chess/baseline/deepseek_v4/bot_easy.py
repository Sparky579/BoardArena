"""DeepSeek v4 easy bot (~800 Elo).

A deliberately weak chess bot.  The large majority of moves are
completely random; only occasionally does it notice a capture,
promotion, or check.  This keeps losses high while still allowing the
occasional lucky win, producing an effective playing strength around
800 Elo against the system opponent.
"""

from __future__ import annotations

import random

import chess

# Simplified, approximate piece values for capture scoring.
_PIECE_VALUE: dict[int, int] = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,
}


class Bot:
    """A weak chess bot whose ``choose_action`` selects legal UCI moves."""

    name = "deepseek_v4_easy"

    def choose_action(self, state: dict) -> str:
        """Return a UCI move string from ``state["legal_actions"]``."""
        legal: list[str] = state["legal_actions"]
        if not legal:
            raise ValueError("no legal actions available")

        board = chess.Board(state["fen"])
        roll = random.random()

        # 85 % – completely random.
        if roll < 0.85:
            return random.choice(legal)

        # 6 % – random capture if any are available, else random.
        if roll < 0.91:
            caps = _list_captures(board, legal)
            if caps and random.random() < 0.60:
                return random.choice(caps)
            return random.choice(legal)

        # 5 % – best-value capture if available, else random.
        if roll < 0.96:
            best = _best_capture(board, legal)
            return best if best is not None else random.choice(legal)

        # 4 % – "smart" mode: promote > check > best capture > random.
        promos = _list_promotions(board, legal)
        if promos and random.random() < 0.75:
            return _pick_best_promotion(board, promos)

        checks = _list_checks(board, legal)
        if checks and random.random() < 0.65:
            return random.choice(checks)

        best = _best_capture(board, legal)
        return best if best is not None else random.choice(legal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_captures(board: chess.Board, legal: list[str]) -> list[str]:
    """Return every legal capture (including en-passant)."""
    result: list[str] = []
    for uci in legal:
        move = chess.Move.from_uci(uci)
        if board.is_capture(move):
            result.append(uci)
    return result


def _best_capture(board: chess.Board, legal: list[str]) -> str | None:
    """Return the capture that wins the most material, or *None*."""
    best_uci: str | None = None
    best_val = -1

    for uci in legal:
        move = chess.Move.from_uci(uci)
        if not board.is_capture(move):
            continue

        captured = board.piece_at(move.to_square)
        if captured is None and board.is_en_passant(move):
            captured = chess.Piece(chess.PAWN, not board.turn)

        val = _PIECE_VALUE.get(captured.piece_type, 0) if captured else 0
        if val > best_val:
            best_val = val
            best_uci = uci

    return best_uci


def _list_checks(board: chess.Board, legal: list[str]) -> list[str]:
    """Return every legal move that delivers check."""
    result: list[str] = []
    for uci in legal:
        move = chess.Move.from_uci(uci)
        board.push(move)
        if board.is_check():
            result.append(uci)
        board.pop()
    return result


def _list_promotions(board: chess.Board, legal: list[str]) -> list[str]:
    """Return every legal promotion move."""
    result: list[str] = []
    for uci in legal:
        move = chess.Move.from_uci(uci)
        if move.promotion is not None:
            result.append(uci)
    return result


def _pick_best_promotion(
    board: chess.Board, promos: list[str],
) -> str:
    """Pick the promotion yielding the most valuable piece, preferring
    queen promotions and factoring in concurrent captures for tie-breaking."""

    def _key(uci: str) -> tuple[int, int]:
        move = chess.Move.from_uci(uci)
        promo_val = _PIECE_VALUE.get(move.promotion, 0) if move.promotion else 0

        cap_val = 0
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured is None and board.is_en_passant(move):
                captured = chess.Piece(chess.PAWN, not board.turn)
            cap_val = _PIECE_VALUE.get(captured.piece_type, 0) if captured else 0

        return (-promo_val, -cap_val)

    promos.sort(key=_key)
    return promos[0]
