"""Hard BoardArena chess bot.

This is a self-contained, CPU-only classical chess engine.  It deliberately
does not use neural networks, tablebases, external engines, or GPU-dependent
packages; strength comes from alpha-beta search, tactical quiescence, move
ordering, a transposition table, and a hand-written evaluation.
"""

from __future__ import annotations

import os
import time

try:
    import chess
except ImportError:  # pragma: no cover - lets the judge surface a clear result.
    chess = None


name = "gpt5p5_hard"

MAX_TIME_SECONDS = float(os.environ.get("BOARDARENA_CHESS_HARD_TIME", "1.6"))
ENDGAME_TIME_SECONDS = float(os.environ.get("BOARDARENA_CHESS_HARD_ENDGAME_TIME", "1.6"))
MAX_DEPTH = 64
QUIESCENCE_DEPTH = 8
INF = 10_000_000
MATE = 1_000_000
MATE_BOUND = 900_000
TT_LIMIT = 300_000
EVAL_LIMIT = 160_000
MAX_PLY = 128
CHECK_TIME_EVERY = 2048

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

TEMPO = 14
ASPIRATION_WINDOW = 32
REPETITION_ESCAPE_SCORE = -260
REPETITION_SECOND_PENALTY = 90
REPETITION_DRAW_PENALTY = 280

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
} if chess else {}

MG_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 325,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 975,
    chess.KING: 0,
} if chess else {}

EG_VALUES = {
    chess.PAWN: 118,
    chess.KNIGHT: 305,
    chess.BISHOP: 330,
    chess.ROOK: 535,
    chess.QUEEN: 935,
    chess.KING: 0,
} if chess else {}

PHASE_WEIGHTS = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
} if chess else {}

MOVE_PIECES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
) if chess else ()

PAWN_MG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    -8, -4, 0, 8, 8, 0, -4, -8,
    -4, 2, 8, 18, 18, 8, 2, -4,
    0, 4, 14, 28, 28, 14, 4, 0,
    6, 10, 20, 34, 34, 20, 10, 6,
    12, 16, 28, 42, 42, 28, 16, 12,
    78, 84, 90, 96, 96, 90, 84, 78,
    0, 0, 0, 0, 0, 0, 0, 0,
]

PAWN_EG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    4, 6, 8, 10, 10, 8, 6, 4,
    8, 10, 16, 22, 22, 16, 10, 8,
    14, 18, 26, 36, 36, 26, 18, 14,
    26, 32, 42, 56, 56, 42, 32, 26,
    50, 58, 72, 88, 88, 72, 58, 50,
    100, 112, 124, 136, 136, 124, 112, 100,
    0, 0, 0, 0, 0, 0, 0, 0,
]

KNIGHT_MG = [
    -70, -46, -32, -24, -24, -32, -46, -70,
    -44, -24, -6, 2, 2, -6, -24, -44,
    -30, -4, 14, 22, 22, 14, -4, -30,
    -24, 4, 24, 34, 34, 24, 4, -24,
    -24, 8, 26, 36, 36, 26, 8, -24,
    -30, -2, 18, 26, 26, 18, -2, -30,
    -44, -22, -2, 6, 6, -2, -22, -44,
    -70, -46, -30, -22, -22, -30, -46, -70,
]

KNIGHT_EG = [
    -58, -38, -26, -18, -18, -26, -38, -58,
    -38, -18, -2, 6, 6, -2, -18, -38,
    -26, -2, 14, 22, 22, 14, -2, -26,
    -18, 6, 24, 32, 32, 24, 6, -18,
    -18, 6, 24, 32, 32, 24, 6, -18,
    -26, -2, 14, 22, 22, 14, -2, -26,
    -38, -18, -2, 6, 6, -2, -18, -38,
    -58, -38, -26, -18, -18, -26, -38, -58,
]

BISHOP_MG = [
    -24, -12, -10, -8, -8, -10, -12, -24,
    -10, 6, 2, 4, 4, 2, 6, -10,
    -8, 10, 12, 14, 14, 12, 10, -8,
    -6, 4, 16, 22, 22, 16, 4, -6,
    -6, 8, 16, 24, 24, 16, 8, -6,
    -8, 12, 14, 18, 18, 14, 12, -8,
    -10, 8, 8, 6, 6, 8, 8, -10,
    -24, -12, -10, -8, -8, -10, -12, -24,
]

BISHOP_EG = [
    -18, -8, -6, -4, -4, -6, -8, -18,
    -8, 4, 8, 8, 8, 8, 4, -8,
    -6, 10, 12, 14, 14, 12, 10, -6,
    -4, 8, 16, 20, 20, 16, 8, -4,
    -4, 8, 16, 20, 20, 16, 8, -4,
    -6, 10, 12, 14, 14, 12, 10, -6,
    -8, 4, 8, 8, 8, 8, 4, -8,
    -18, -8, -6, -4, -4, -6, -8, -18,
]

ROOK_MG = [
    0, 0, 4, 8, 8, 4, 0, 0,
    -4, 2, 4, 8, 8, 4, 2, -4,
    -6, 0, 2, 6, 6, 2, 0, -6,
    -6, 0, 2, 6, 6, 2, 0, -6,
    -4, 2, 6, 10, 10, 6, 2, -4,
    0, 4, 8, 12, 12, 8, 4, 0,
    18, 22, 24, 28, 28, 24, 22, 18,
    2, 4, 8, 12, 12, 8, 4, 2,
]

ROOK_EG = [
    0, 2, 4, 6, 6, 4, 2, 0,
    2, 4, 6, 8, 8, 6, 4, 2,
    2, 4, 6, 8, 8, 6, 4, 2,
    4, 6, 8, 10, 10, 8, 6, 4,
    6, 8, 10, 12, 12, 10, 8, 6,
    8, 10, 12, 14, 14, 12, 10, 8,
    12, 14, 16, 18, 18, 16, 14, 12,
    8, 10, 12, 14, 14, 12, 10, 8,
]

QUEEN_MG = [
    -18, -10, -8, -4, -4, -8, -10, -18,
    -10, -2, 2, 4, 4, 2, -2, -10,
    -8, 2, 8, 10, 10, 8, 2, -8,
    -4, 4, 10, 14, 14, 10, 4, -4,
    -2, 6, 12, 16, 16, 12, 6, -2,
    -8, 4, 10, 12, 12, 10, 4, -8,
    -10, -2, 4, 6, 6, 4, -2, -10,
    -18, -10, -8, -4, -4, -8, -10, -18,
]

QUEEN_EG = [
    -10, -6, -4, -2, -2, -4, -6, -10,
    -6, 2, 4, 6, 6, 4, 2, -6,
    -4, 4, 8, 10, 10, 8, 4, -4,
    -2, 6, 10, 14, 14, 10, 6, -2,
    -2, 6, 10, 14, 14, 10, 6, -2,
    -4, 4, 8, 10, 10, 8, 4, -4,
    -6, 2, 4, 6, 6, 4, 2, -6,
    -10, -6, -4, -2, -2, -4, -6, -10,
]

KING_MG = [
    34, 44, 18, -28, -32, 4, 46, 34,
    20, 22, 0, -12, -12, 0, 22, 20,
    -16, -22, -28, -38, -38, -28, -22, -16,
    -28, -36, -44, -54, -54, -44, -36, -28,
    -38, -46, -56, -66, -66, -56, -46, -38,
    -46, -56, -66, -76, -76, -66, -56, -46,
    -54, -64, -74, -84, -84, -74, -64, -54,
    -62, -72, -82, -92, -92, -82, -72, -62,
]

KING_EG = [
    -58, -38, -26, -16, -16, -26, -38, -58,
    -38, -18, -4, 6, 6, -4, -18, -38,
    -26, -4, 18, 30, 30, 18, -4, -26,
    -16, 6, 30, 44, 44, 30, 6, -16,
    -16, 6, 30, 44, 44, 30, 6, -16,
    -26, -4, 18, 30, 30, 18, -4, -26,
    -38, -18, -4, 6, 6, -4, -18, -38,
    -58, -38, -26, -16, -16, -26, -38, -58,
]

MG_TABLES = {
    chess.PAWN: PAWN_MG,
    chess.KNIGHT: KNIGHT_MG,
    chess.BISHOP: BISHOP_MG,
    chess.ROOK: ROOK_MG,
    chess.QUEEN: QUEEN_MG,
    chess.KING: KING_MG,
} if chess else {}

EG_TABLES = {
    chess.PAWN: PAWN_EG,
    chess.KNIGHT: KNIGHT_EG,
    chess.BISHOP: BISHOP_EG,
    chess.ROOK: ROOK_EG,
    chess.QUEEN: QUEEN_EG,
    chess.KING: KING_EG,
} if chess else {}

CENTER_MASK = chess.BB_CENTER if chess else 0
EXTENDED_CENTER = (
    chess.BB_C3 | chess.BB_D3 | chess.BB_E3 | chess.BB_F3 |
    chess.BB_C4 | chess.BB_D4 | chess.BB_E4 | chess.BB_F4 |
    chess.BB_C5 | chess.BB_D5 | chess.BB_E5 | chess.BB_F5 |
    chess.BB_C6 | chess.BB_D6 | chess.BB_E6 | chess.BB_F6
) if chess else 0

OPENING_BOOK = {
    (): ("e2e4", "d2d4", "g1f3", "c2c4"),
    ("e4",): ("c7c5", "e7e5", "c7c6", "e7e6"),
    ("d4",): ("g8f6", "d7d5", "e7e6", "c7c5"),
    ("Nf3",): ("g8f6", "d7d5", "c7c5"),
    ("c4",): ("g8f6", "e7e5", "c7c5"),
    ("e4", "c5"): ("g1f3", "d2d4", "c2c3"),
    ("e4", "e5"): ("g1f3", "f1c4", "d2d4"),
    ("e4", "c6"): ("d2d4", "b1c3"),
    ("e4", "e6"): ("d2d4", "b1c3"),
    ("e4", "d6"): ("d2d4", "g1f3"),
    ("e4", "Nf6"): ("e4e5", "b1c3"),
    ("d4", "Nf6"): ("c2c4", "g1f3"),
    ("d4", "d5"): ("c2c4", "g1f3"),
    ("d4", "e6"): ("c2c4", "e2e4", "g1f3"),
    ("d4", "c5"): ("d4d5", "g1f3"),
    ("c4", "e5"): ("b1c3", "g2g3"),
    ("c4", "Nf6"): ("b1c3", "g1f3", "g2g3"),
    ("Nf3", "d5"): ("d2d4", "c2c4"),
    ("Nf3", "Nf6"): ("c2c4", "d2d4", "g2g3"),

    ("e4", "c5", "Nf3"): ("d7d6", "b8c6", "e7e6"),
    ("e4", "c5", "Nf3", "d6"): ("d2d4",),
    ("e4", "c5", "Nf3", "Nc6"): ("d2d4", "f1b5"),
    ("e4", "c5", "Nf3", "e6"): ("d2d4",),
    ("e4", "c5", "Nf3", "d6", "d4"): ("c5d4",),
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4"): ("f3d4",),
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4"): ("g8f6",),
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6"): ("b1c3",),
    ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3"): ("a7a6", "e7e6"),

    ("e4", "e5", "Nf3"): ("b8c6", "g8f6"),
    ("e4", "e5", "Nf3", "Nc6"): ("f1b5", "f1c4", "d2d4"),
    ("e4", "e5", "Nf3", "Nc6", "Bb5"): ("a7a6", "g8f6"),
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"): ("b5a4",),
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4"): ("g8f6",),
    ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"): ("e1g1",),
    ("e4", "e5", "Nf3", "Nc6", "Bc4"): ("g8f6", "f8c5"),

    ("e4", "c6", "d4"): ("d7d5",),
    ("e4", "c6", "d4", "d5"): ("b1c3", "e4e5"),
    ("e4", "c6", "d4", "d5", "Nc3"): ("d5e4", "g8f6"),
    ("e4", "e6", "d4"): ("d7d5",),
    ("e4", "e6", "d4", "d5"): ("b1c3", "e4e5"),
    ("e4", "e6", "d4", "d5", "Nc3"): ("g8f6", "f8b4"),

    ("d4", "d5", "c4"): ("e7e6", "c7c6", "d5c4"),
    ("d4", "d5", "c4", "e6"): ("b1c3", "g1f3"),
    ("d4", "d5", "c4", "e6", "Nc3"): ("g8f6", "f8e7"),
    ("d4", "d5", "c4", "e6", "Nc3", "Nf6"): ("c1g5", "g1f3"),
    ("d4", "Nf6", "c4"): ("e7e6", "g7g6", "c7c5"),
    ("d4", "Nf6", "c4", "e6"): ("g1f3", "b1c3"),
    ("d4", "Nf6", "c4", "e6", "Nf3"): ("d7d5", "f8b4"),
    ("d4", "Nf6", "c4", "g6"): ("b1c3", "g2g3"),
}

_GLOBAL_TT = {}
_GLOBAL_EVAL = {}


class SearchTimeout(Exception):
    """Raised internally when the per-move time budget is exhausted."""


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        raise ValueError("no legal actions")
    if len(legal) == 1:
        return legal[0]
    if chess is None:
        return sorted(legal)[0]

    board = _board_from_state(state)
    book_move = _book_move(state, legal)
    if book_move is not None:
        return book_move

    searcher = Searcher(board, legal)
    try:
        action = searcher.search()
    except Exception:  # noqa: BLE001 - a safe legal fallback is better than forfeiting.
        action = _fallback_move(board, legal)
    return action if action in legal else _fallback_move(board, legal)


class Searcher:
    def __init__(self, board, legal_actions):
        self.board = board
        self.legal_actions = set(legal_actions)
        self.start_time = time.monotonic()
        self.time_budget = _time_budget(board, len(legal_actions))
        self.deadline = self.start_time + self.time_budget
        self.nodes = 0
        self.tt = _GLOBAL_TT
        self.eval_cache = _GLOBAL_EVAL
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {}

        if len(self.tt) > TT_LIMIT:
            self.tt.clear()
        if len(self.eval_cache) > EVAL_LIMIT:
            self.eval_cache.clear()

    def search(self):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            raise ValueError("no legal actions")
        if len(legal_moves) == 1:
            return legal_moves[0].uci()

        best_move = chess.Move.from_uci(_fallback_move(self.board, legal_moves))
        best_score = -INF
        last_score = 0
        completed_depth = 0

        try:
            for depth in range(1, MAX_DEPTH + 1):
                if self._soft_stop(completed_depth):
                    break

                if completed_depth >= 3:
                    window = ASPIRATION_WINDOW
                    alpha = max(-INF, last_score - window)
                    beta = min(INF, last_score + window)
                else:
                    alpha = -INF
                    beta = INF

                while True:
                    score, move = self._search_root(depth, alpha, beta, best_move)
                    if score <= alpha and alpha > -INF // 2:
                        alpha = max(-INF, alpha - window)
                        window *= 2
                        continue
                    if score >= beta and beta < INF // 2:
                        beta = min(INF, beta + window)
                        window *= 2
                        continue
                    break

                best_move = move
                best_score = score
                last_score = score
                completed_depth = depth

                if abs(best_score) > MATE_BOUND and MATE - abs(best_score) <= depth + 2:
                    break
        except SearchTimeout:
            pass

        return best_move.uci()

    def _search_root(self, depth, alpha, beta, preferred):
        alpha_original = alpha
        best_score = -INF
        best_move = preferred
        moves = self._ordered_moves(list(self.board.legal_moves), preferred, 0)
        root_static = self._evaluate()
        searched = 0

        for move in moves:
            self._check_time()
            gives_check = self.board.gives_check(move)
            extension = 1 if gives_check and depth <= 6 else 0
            self.board.push(move)
            if searched == 0:
                score = -self._search(depth - 1 + extension, -beta, -alpha, 1, True, True)
            else:
                score = -self._search(depth - 1 + extension, -alpha - 1, -alpha, 1, True, False)
                if alpha < score < beta:
                    score = -self._search(depth - 1 + extension, -beta, -alpha, 1, True, True)
            score -= self._root_repetition_penalty(root_static)
            self.board.pop()
            searched += 1

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not self.board.is_capture(move) and not move.promotion:
                    self._record_quiet_cutoff(move, depth, 0)
                break

        flag = TT_EXACT
        if best_score <= alpha_original:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        self._store_tt(_position_key(self.board), depth, best_score, flag, best_move, self._evaluate(), 0)
        return best_score, best_move

    def _search(self, depth, alpha, beta, ply, allow_null, pv_node):
        self.nodes += 1
        if self.nodes & (CHECK_TIME_EVERY - 1) == 0:
            self._check_time()
        if ply >= MAX_PLY - 1:
            return self._evaluate()

        terminal = self._terminal_score(ply)
        if terminal is not None:
            return terminal

        alpha = max(alpha, -MATE + ply)
        beta = min(beta, MATE - ply)
        if alpha >= beta:
            return alpha

        in_check = self.board.is_check()
        if depth <= 0:
            return self._quiescence(alpha, beta, ply, QUIESCENCE_DEPTH)

        key = _position_key(self.board)
        alpha_original = alpha
        tt_entry = self.tt.get(key)
        tt_move = None
        static_eval = None

        if tt_entry is not None:
            tt_depth, tt_score, tt_flag, tt_move, tt_static = tt_entry
            static_eval = tt_static
            if tt_depth >= depth:
                score = _score_from_tt(tt_score, ply)
                if tt_flag == TT_EXACT:
                    return score
                if tt_flag == TT_LOWER and score >= beta:
                    return score
                if tt_flag == TT_UPPER and score <= alpha:
                    return score

        if not in_check:
            if static_eval is None:
                static_eval = self._evaluate()

            if not pv_node and abs(beta) < MATE_BOUND:
                if depth <= 3 and static_eval - (70 + 45 * depth) >= beta:
                    return static_eval - (35 * depth)
                if depth <= 2 and static_eval + (150 + 120 * depth) <= alpha:
                    q_score = self._quiescence(alpha, beta, ply, max(2, QUIESCENCE_DEPTH - 3))
                    if q_score <= alpha:
                        return q_score

            if (
                allow_null
                and not pv_node
                and depth >= 3
                and static_eval >= beta
                and self._has_non_pawn_material(self.board.turn)
                and abs(beta) < MATE_BOUND
            ):
                reduction = 2 + depth // 4 + min(2, max(0, (static_eval - beta) // 180))
                self.board.push(chess.Move.null())
                score = -self._search(depth - reduction - 1, -beta, -beta + 1, ply + 1, False, False)
                self.board.pop()
                if score >= beta:
                    return score if score > MATE_BOUND else beta

        moves = self._ordered_moves(list(self.board.legal_moves), tt_move, ply)
        best_score = -INF
        best_move = None
        searched = 0

        for move in moves:
            capture = self.board.is_capture(move)
            gives_check = self.board.gives_check(move)
            quiet = not capture and not move.promotion

            if not pv_node and not in_check and quiet and not gives_check and searched > 0:
                if depth <= 2 and static_eval + (85 + 90 * depth) <= alpha:
                    continue

            if not pv_node and not in_check and capture and not move.promotion and depth <= 3:
                if self._see(move) < -90 * depth:
                    continue

            extension = 1 if gives_check and depth <= 6 else 0
            new_depth = depth - 1 + extension
            reduction = 0

            if quiet and not gives_check and not in_check and depth >= 3 and searched >= 3:
                reduction = 1
                if depth >= 6:
                    reduction += 1
                if searched >= 10:
                    reduction += 1
                if self.history.get(_history_key(self.board.turn, move), 0) > depth * depth * 8:
                    reduction -= 1
                reduction = max(0, min(reduction, max(0, new_depth - 1)))

            self.board.push(move)
            if searched == 0:
                score = -self._search(new_depth, -beta, -alpha, ply + 1, True, pv_node)
            else:
                score = -self._search(new_depth - reduction, -alpha - 1, -alpha, ply + 1, True, False)
                if reduction and score > alpha:
                    score = -self._search(new_depth, -alpha - 1, -alpha, ply + 1, True, False)
                if alpha < score < beta:
                    score = -self._search(new_depth, -beta, -alpha, ply + 1, True, True)
            self.board.pop()
            searched += 1

            if score > best_score or best_move is None:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if quiet:
                    self._record_quiet_cutoff(move, depth, ply)
                self._store_tt(key, depth, score, TT_LOWER, move, static_eval, ply)
                return score

        if searched == 0:
            return self._quiescence(alpha_original, beta, ply, max(2, QUIESCENCE_DEPTH - 4))

        flag = TT_EXACT
        if best_score <= alpha_original:
            flag = TT_UPPER
        self._store_tt(key, depth, best_score, flag, best_move, static_eval, ply)
        return best_score

    def _quiescence(self, alpha, beta, ply, depth_left):
        self.nodes += 1
        if self.nodes & (CHECK_TIME_EVERY - 1) == 0:
            self._check_time()

        terminal = self._terminal_score(ply)
        if terminal is not None:
            return terminal

        in_check = self.board.is_check()
        if depth_left <= -2:
            return self._evaluate()

        if not in_check:
            stand_pat = self._evaluate()
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
            if depth_left <= 0:
                return alpha
        else:
            stand_pat = -INF

        moves = []
        allow_quiet_checks = depth_left >= 3
        for move in self.board.legal_moves:
            if in_check or self.board.is_capture(move) or move.promotion:
                moves.append(move)
            elif allow_quiet_checks and self.board.gives_check(move):
                moves.append(move)

        moves = self._ordered_moves(moves, None, ply)
        for move in moves:
            capture = self.board.is_capture(move)
            gives_check = self.board.gives_check(move)
            if not in_check:
                swing = self._capture_value(move) + _promotion_delta(move)
                if not gives_check and stand_pat + swing + 170 < alpha:
                    continue
                if capture and not move.promotion and self._see(move) < -110:
                    continue

            self.board.push(move)
            score = -self._quiescence(-beta, -alpha, ply + 1, depth_left - 1)
            self.board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _evaluate(self):
        key = _position_key(self.board)
        cached = self.eval_cache.get(key)
        if cached is not None:
            return cached

        score = _raw_evaluate(self.board)
        score = score if self.board.turn == chess.WHITE else -score
        score += TEMPO

        if len(self.eval_cache) < EVAL_LIMIT:
            self.eval_cache[key] = score
        return score

    def _terminal_score(self, ply):
        outcome = self.board.outcome(claim_draw=False)
        if outcome is not None:
            if outcome.winner is None:
                return 0
            return MATE - ply if outcome.winner == self.board.turn else -MATE + ply
        if self.board.can_claim_fifty_moves() or self.board.is_repetition(3):
            return 0
        return None

    def _ordered_moves(self, moves, preferred, ply):
        return sorted(moves, key=lambda move: self._move_order_score(move, preferred, ply), reverse=True)

    def _move_order_score(self, move, preferred, ply):
        if preferred is not None and move == preferred:
            return 5_000_000

        score = 0
        mover = self.board.piece_at(move.from_square)
        capture = self.board.is_capture(move)

        if capture:
            victim_value = self._capture_value(move)
            attacker_value = PIECE_VALUES.get(mover.piece_type, 0) if mover else 0
            score += 1_000_000 + 16 * victim_value - attacker_value
            see = self._see(move)
            score += max(-300, min(900, see))

        if move.promotion:
            score += 900_000 + PIECE_VALUES.get(move.promotion, 0)
            if move.promotion == chess.QUEEN:
                score += 40

        killer_0, killer_1 = self.killers[ply] if ply < MAX_PLY else (None, None)
        if move == killer_0:
            score += 720_000
        elif move == killer_1:
            score += 700_000

        if not capture and not move.promotion:
            score += self.history.get(_history_key(self.board.turn, move), 0)

        if self.board.gives_check(move):
            score += 80_000
        if self.board.is_castling(move):
            score += 35_000

        if mover is not None:
            to_bb = chess.BB_SQUARES[move.to_square]
            if to_bb & CENTER_MASK:
                score += 900
            elif to_bb & EXTENDED_CENTER:
                score += 350
            if mover.piece_type in (chess.KNIGHT, chess.BISHOP) and _is_back_rank(move.from_square, mover.color):
                score += 850
            if mover.piece_type == chess.PAWN and _is_passed_after_push(self.board, move):
                score += 1200 + 150 * _relative_rank(move.to_square, mover.color)

        return score

    def _see(self, move):
        if not self.board.is_capture(move):
            return _promotion_delta(move)

        target = move.to_square
        from_square = move.from_square
        color = self.board.turn
        attacker = self.board.piece_at(from_square)
        if attacker is None:
            return 0

        captured_type = _captured_piece_type(self.board, move)
        gain = [PIECE_VALUES.get(captured_type, 0) + _promotion_delta(move)]
        occupied = self.board.occupied
        occupied ^= chess.BB_SQUARES[from_square]
        if self.board.is_en_passant(move):
            ep_rank = chess.square_rank(target) - (1 if color == chess.WHITE else -1)
            occupied ^= chess.BB_SQUARES[chess.square(chess.square_file(target), ep_rank)]
            occupied |= chess.BB_SQUARES[target]

        side = not color
        victim_type = move.promotion or attacker.piece_type
        attackers = (
            self.board.attackers_mask(chess.WHITE, target, occupied)
            | self.board.attackers_mask(chess.BLACK, target, occupied)
        ) & occupied

        index = 0
        while True:
            from_set = attackers & self.board.occupied_co[side]
            if not from_set:
                break
            least_square = None
            least_type = None
            for piece_type in MOVE_PIECES:
                candidates = from_set & self.board.pieces_mask(piece_type, side)
                if candidates:
                    least_square = chess.lsb(candidates)
                    least_type = piece_type
                    break
            if least_square is None or least_type is None:
                break

            index += 1
            gain.append(PIECE_VALUES.get(victim_type, 0) - gain[index - 1])
            occupied ^= chess.BB_SQUARES[least_square]
            attackers = (
                self.board.attackers_mask(chess.WHITE, target, occupied)
                | self.board.attackers_mask(chess.BLACK, target, occupied)
            ) & occupied
            side = not side
            victim_type = least_type

        while index:
            gain[index - 1] = -max(-gain[index - 1], gain[index])
            index -= 1
        return gain[0]

    def _capture_value(self, move):
        return PIECE_VALUES.get(_captured_piece_type(self.board, move), 0)

    def _root_repetition_penalty(self, root_static):
        if root_static <= REPETITION_ESCAPE_SCORE:
            return 0
        if self.board.can_claim_threefold_repetition() or self.board.is_repetition(3):
            return REPETITION_DRAW_PENALTY + max(0, root_static) // 4
        if self.board.is_repetition(2):
            return REPETITION_SECOND_PENALTY + max(0, root_static) // 10
        return 0

    def _record_quiet_cutoff(self, move, depth, ply):
        if ply < MAX_PLY:
            first, second = self.killers[ply]
            if move != first:
                self.killers[ply] = [move, first if first != move else second]
        key = _history_key(self.board.turn, move)
        self.history[key] = min(900_000, self.history.get(key, 0) + depth * depth * 24)

    def _store_tt(self, key, depth, score, flag, move, static_eval, ply):
        if len(self.tt) >= TT_LIMIT:
            return
        self.tt[key] = (depth, _score_to_tt(score, ply), flag, move, static_eval)

    def _has_non_pawn_material(self, color):
        return bool(self.board.occupied_co[color] & (self.board.knights | self.board.bishops | self.board.rooks | self.board.queens))

    def _check_time(self):
        if time.monotonic() >= self.deadline:
            raise SearchTimeout

    def _soft_stop(self, completed_depth):
        if completed_depth <= 0:
            return False
        elapsed = time.monotonic() - self.start_time
        return elapsed >= self.time_budget * 0.62


def _raw_evaluate(board):
    if board.is_insufficient_material():
        return 0

    mg = 0
    eg = 0
    phase = 0
    bishops = {chess.WHITE: 0, chess.BLACK: 0}
    material = {chess.WHITE: 0, chess.BLACK: 0}
    non_pawn_material = {chess.WHITE: 0, chess.BLACK: 0}
    pawn_files = {chess.WHITE: [0] * 8, chess.BLACK: [0] * 8}
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        sign = 1 if piece.color == chess.WHITE else -1
        oriented = square if piece.color == chess.WHITE else chess.square_mirror(square)
        piece_type = piece.piece_type

        mg += sign * (MG_VALUES[piece_type] + MG_TABLES[piece_type][oriented])
        eg += sign * (EG_VALUES[piece_type] + EG_TABLES[piece_type][oriented])
        phase += PHASE_WEIGHTS[piece_type]
        material[piece.color] += PIECE_VALUES[piece_type]
        if piece_type != chess.PAWN:
            non_pawn_material[piece.color] += PIECE_VALUES[piece_type]
        else:
            pawn_files[piece.color][chess.square_file(square)] += 1
        if piece_type == chess.BISHOP:
            bishops[piece.color] += 1

    phase = min(24, phase)

    for square in board.pieces(chess.ROOK, chess.WHITE):
        file_index = chess.square_file(square)
        own_pawns = pawn_files[chess.WHITE][file_index]
        enemy_pawns = pawn_files[chess.BLACK][file_index]
        if own_pawns == 0 and enemy_pawns == 0:
            mg += 22
            eg += 16
        elif own_pawns == 0:
            mg += 11
            eg += 8
        if _relative_rank(square, chess.WHITE) == 6:
            mg += 20
            eg += 28
    for square in board.pieces(chess.ROOK, chess.BLACK):
        file_index = chess.square_file(square)
        own_pawns = pawn_files[chess.BLACK][file_index]
        enemy_pawns = pawn_files[chess.WHITE][file_index]
        if own_pawns == 0 and enemy_pawns == 0:
            mg -= 22
            eg -= 16
        elif own_pawns == 0:
            mg -= 11
            eg -= 8
        if _relative_rank(square, chess.BLACK) == 6:
            mg -= 20
            eg -= 28

    score = (mg * phase + eg * (24 - phase)) // 24

    if bishops[chess.WHITE] >= 2:
        score += 38
    if bishops[chess.BLACK] >= 2:
        score -= 38

    score += _pawn_structure(board, pawn_files)
    score += _mobility(board, piece_map, phase)
    score += _king_safety(board, piece_map, pawn_files, phase)
    score += _threats(board, piece_map)
    score += _development(board)
    score += _mop_up(board, score, material, non_pawn_material)
    return _scale_drawish_endings(board, score, material, non_pawn_material)


def _pawn_structure(board, pawn_files):
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        pawns = list(board.pieces(chess.PAWN, color))
        enemy_pawns = board.pieces(chess.PAWN, not color)
        files = pawn_files[color]

        for file_index, count in enumerate(files):
            if count > 1:
                score -= sign * 14 * (count - 1)

        for square in pawns:
            file_index = chess.square_file(square)
            rank = _relative_rank(square, color)
            adjacent_files = [idx for idx in (file_index - 1, file_index + 1) if 0 <= idx < 8]

            if all(files[idx] == 0 for idx in adjacent_files):
                score -= sign * (10 + 2 * rank)
            else:
                score += sign * 5

            front = _front_square(square, color)
            if front is not None:
                front_piece = board.piece_at(front)
                if front_piece is not None and front_piece.color != color:
                    score -= sign * 9
                if board.attackers(not color, front) and not board.attackers(color, front):
                    score -= sign * 7

            if _is_passed_pawn(square, color, enemy_pawns):
                bonus = [0, 10, 20, 36, 62, 104, 170, 0][rank]
                if front is not None and board.piece_at(front) is None:
                    bonus += 8 + 3 * rank
                if board.attackers(color, square):
                    bonus += 8 + 2 * rank
                enemy_king = board.king(not color)
                own_king = board.king(color)
                if enemy_king is not None:
                    bonus += 4 * (7 - _king_distance(enemy_king, square))
                if own_king is not None:
                    bonus += 2 * (7 - _king_distance(own_king, square))
                score += sign * bonus
            elif _is_candidate_passer(square, color, enemy_pawns, files):
                score += sign * (8 + 5 * rank)

    return score


def _mobility(board, piece_map, phase):
    score = 0
    pawn_attacks = {
        chess.WHITE: _pawn_attack_mask(board, chess.WHITE),
        chess.BLACK: _pawn_attack_mask(board, chess.BLACK),
    }
    weights = {
        chess.KNIGHT: 5,
        chess.BISHOP: 5,
        chess.ROOK: 3,
        chess.QUEEN: 2,
    }

    for square, piece in piece_map.items():
        piece_type = piece.piece_type
        if piece_type not in weights:
            continue
        sign = 1 if piece.color == chess.WHITE else -1
        attacks = _attacks_mask(board, square) & ~board.occupied_co[piece.color]
        if piece_type in (chess.KNIGHT, chess.BISHOP):
            attacks &= ~pawn_attacks[not piece.color]
        count = chess.popcount(attacks)
        score += sign * weights[piece_type] * count
        score += sign * 5 * chess.popcount(attacks & CENTER_MASK)
        score += sign * chess.popcount(attacks & EXTENDED_CENTER)

        if piece_type == chess.BISHOP:
            own_pawns_on_color = int(board.pieces(chess.PAWN, piece.color)) & (
                chess.BB_LIGHT_SQUARES if chess.BB_SQUARES[square] & chess.BB_LIGHT_SQUARES else chess.BB_DARK_SQUARES
            )
            score -= sign * min(18, 3 * chess.popcount(own_pawns_on_color))

    return score * (12 + phase) // 36


def _king_safety(board, piece_map, pawn_files, phase):
    if phase <= 5:
        return 0

    score = 0
    attack_weight = max(1, phase)
    attacker_weights = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 7,
        chess.KING: 0,
    }

    for color in (chess.WHITE, chess.BLACK):
        king = board.king(color)
        if king is None:
            continue
        sign = 1 if color == chess.WHITE else -1
        ring = chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]
        enemy = not color
        attack_units = 0

        for square, piece in piece_map.items():
            if piece.color != enemy:
                continue
            hits = chess.popcount(_attacks_mask(board, square) & ring)
            if hits:
                attack_units += hits * attacker_weights[piece.piece_type]

        shield = 0
        king_file = chess.square_file(king)
        king_rank = chess.square_rank(king)
        direction = 1 if color == chess.WHITE else -1
        for file_index in (king_file - 1, king_file, king_file + 1):
            if not 0 <= file_index < 8:
                continue
            for distance, value in ((1, 13), (2, 6)):
                rank = king_rank + direction * distance
                if not 0 <= rank < 8:
                    continue
                piece = board.piece_at(chess.square(file_index, rank))
                if piece and piece.color == color and piece.piece_type == chess.PAWN:
                    shield += value
                    break

        open_file_penalty = 0
        for file_index in (king_file - 1, king_file, king_file + 1):
            if not 0 <= file_index < 8:
                continue
            if pawn_files[color][file_index] == 0:
                open_file_penalty += 9
                if pawn_files[enemy][file_index] == 0:
                    open_file_penalty += 7

        center_penalty = 0
        if chess.square_file(king) in (3, 4) and _relative_rank(king, color) <= 1:
            center_penalty = 18

        danger = (attack_units * attack_units * attack_weight) // 18
        score += sign * (shield - open_file_penalty - center_penalty - danger)

    return score


def _threats(board, piece_map):
    score = 0
    pawn_attacks = {
        chess.WHITE: _pawn_attack_mask(board, chess.WHITE),
        chess.BLACK: _pawn_attack_mask(board, chess.BLACK),
    }

    for square, piece in piece_map.items():
        if piece.piece_type == chess.KING:
            continue
        sign = 1 if piece.color == chess.WHITE else -1
        value = PIECE_VALUES[piece.piece_type]
        enemy = not piece.color
        square_bb = chess.BB_SQUARES[square]
        penalty = 0

        attackers = board.attackers(enemy, square)
        if attackers:
            defenders = board.attackers(piece.color, square)
            attacker_values = [
                PIECE_VALUES[piece_map[attacker].piece_type]
                for attacker in attackers
                if attacker in piece_map
            ]
            if attacker_values and min(attacker_values) < value:
                penalty += min(120, (value - min(attacker_values)) // 5)
            if not defenders:
                penalty += min(95, value // 8)
        elif piece.piece_type != chess.PAWN and not board.attackers(piece.color, square):
            penalty += 5

        if square_bb & pawn_attacks[enemy]:
            penalty += min(160, value // 4)

        score -= sign * penalty

    return score


def _development(board):
    if board.fullmove_number > 16:
        return 0

    score = 0
    starts = {
        chess.WHITE: (
            (chess.B1, chess.KNIGHT),
            (chess.G1, chess.KNIGHT),
            (chess.C1, chess.BISHOP),
            (chess.F1, chess.BISHOP),
        ),
        chess.BLACK: (
            (chess.B8, chess.KNIGHT),
            (chess.G8, chess.KNIGHT),
            (chess.C8, chess.BISHOP),
            (chess.F8, chess.BISHOP),
        ),
    }
    queen_starts = {chess.WHITE: chess.D1, chess.BLACK: chess.D8}

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for square, piece_type in starts[color]:
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type == piece_type:
                score -= sign * 10

        queen_square = queen_starts[color]
        queen = board.piece_at(queen_square)
        if queen is None or queen.color != color or queen.piece_type != chess.QUEEN:
            minor_home = 0
            for square, piece_type in starts[color]:
                piece = board.piece_at(square)
                if piece and piece.color == color and piece.piece_type == piece_type:
                    minor_home += 1
            score -= sign * (7 * minor_home)

        king = board.king(color)
        if king is not None and _relative_rank(king, color) == 0 and chess.square_file(king) == 4:
            rights = board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color)
            if not rights:
                score -= sign * 18

    return score


def _mop_up(board, score, material, non_pawn_material):
    if abs(score) < 360:
        return 0
    if chess.popcount(board.pawns) > 5:
        return 0

    strong = chess.WHITE if score > 0 else chess.BLACK
    weak = not strong
    if non_pawn_material[strong] < non_pawn_material[weak] + 300:
        return 0

    strong_king = board.king(strong)
    weak_king = board.king(weak)
    if strong_king is None or weak_king is None:
        return 0

    edge = _edge_bonus(weak_king)
    distance = _king_distance(strong_king, weak_king)
    bonus = 18 * edge + 6 * (14 - distance)
    return bonus if strong == chess.WHITE else -bonus


def _scale_drawish_endings(board, score, material, non_pawn_material):
    if score == 0 or chess.popcount(board.pawns) > 0:
        return score

    white_np = non_pawn_material[chess.WHITE]
    black_np = non_pawn_material[chess.BLACK]
    if white_np <= PIECE_VALUES[chess.BISHOP] and black_np == 0:
        return 0
    if black_np <= PIECE_VALUES[chess.BISHOP] and white_np == 0:
        return 0
    if abs(material[chess.WHITE] - material[chess.BLACK]) <= 120:
        return score // 2

    white_bishops = board.pieces(chess.BISHOP, chess.WHITE)
    black_bishops = board.pieces(chess.BISHOP, chess.BLACK)
    only_bishops = (
        not board.knights
        and not board.rooks
        and not board.queens
        and chess.popcount(int(white_bishops)) == 1
        and chess.popcount(int(black_bishops)) == 1
    )
    if only_bishops:
        white_light = bool(white_bishops & chess.BB_LIGHT_SQUARES)
        black_light = bool(black_bishops & chess.BB_LIGHT_SQUARES)
        if white_light != black_light:
            return score * 55 // 100
    return score


def _book_move(state, legal):
    history = tuple(state.get("san_history", ()))
    if len(history) > 14:
        return None
    candidates = OPENING_BOOK.get(history)
    if not candidates:
        return None
    legal_set = set(legal)
    for action in candidates:
        if action in legal_set:
            return action
    return None


def _board_from_state(state):
    board = chess.Board(state["fen"])
    history = state.get("san_history") or ()
    if not history:
        return board

    replay = chess.Board()
    try:
        for san in history:
            replay.push_san(san)
    except (ValueError, AssertionError):
        return board

    if _same_position(replay, board):
        replay.halfmove_clock = board.halfmove_clock
        replay.fullmove_number = board.fullmove_number
        return replay
    return board


def _same_position(left, right):
    return (
        left.board_fen() == right.board_fen()
        and left.turn == right.turn
        and left.castling_rights == right.castling_rights
        and left.ep_square == right.ep_square
    )


def _fallback_move(board, legal):
    moves = [chess.Move.from_uci(action) if isinstance(action, str) else action for action in legal]
    best_move = moves[0]
    best_score = -INF
    perspective = board.turn

    for move in moves:
        score = 0
        mover = board.piece_at(move.from_square)
        if board.is_capture(move):
            captured = _captured_piece_type(board, move)
            score += 12 * PIECE_VALUES.get(captured, 0)
            if mover:
                score -= PIECE_VALUES.get(mover.piece_type, 0)
        if move.promotion:
            score += 10 * PIECE_VALUES.get(move.promotion, 0)
        if board.gives_check(move):
            score += 220
        if board.is_castling(move):
            score += 120

        board.push(move)
        try:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                if outcome.winner == perspective:
                    score += MATE
                elif outcome.winner is None:
                    score += 0
                else:
                    score -= MATE
            else:
                score += _raw_evaluate(board) * (1 if perspective == chess.WHITE else -1) // 16
        finally:
            board.pop()

        if score > best_score or (score == best_score and _prefer_move(move, best_move)):
            best_score = score
            best_move = move

    return best_move.uci()


def _time_budget(board, legal_count):
    pieces = len(board.piece_map())
    if pieces <= 12:
        return ENDGAME_TIME_SECONDS
    return MAX_TIME_SECONDS


def _position_key(board):
    key_func = getattr(board, "transposition_key", None)
    if key_func is not None:
        return key_func()
    return board._transposition_key()


def _score_to_tt(score, ply):
    if score > MATE_BOUND:
        return score + ply
    if score < -MATE_BOUND:
        return score - ply
    return score


def _score_from_tt(score, ply):
    if score > MATE_BOUND:
        return score - ply
    if score < -MATE_BOUND:
        return score + ply
    return score


def _history_key(color, move):
    return color, move.from_square, move.to_square, move.promotion


def _prefer_move(move, current):
    if current is None:
        return True
    return move.uci() < current.uci()


def _captured_piece_type(board, move):
    captured = board.piece_at(move.to_square)
    if captured is not None:
        return captured.piece_type
    if board.is_en_passant(move):
        return chess.PAWN
    return None


def _promotion_delta(move):
    if not move.promotion:
        return 0
    return PIECE_VALUES.get(move.promotion, 0) - PIECE_VALUES[chess.PAWN]


def _is_back_rank(square, color):
    rank = chess.square_rank(square)
    return rank == (0 if color == chess.WHITE else 7)


def _relative_rank(square, color):
    rank = chess.square_rank(square)
    return rank if color == chess.WHITE else 7 - rank


def _front_square(square, color):
    file_index = chess.square_file(square)
    rank = chess.square_rank(square) + (1 if color == chess.WHITE else -1)
    if not 0 <= rank < 8:
        return None
    return chess.square(file_index, rank)


def _is_passed_pawn(square, color, enemy_pawns):
    file_index = chess.square_file(square)
    rank = chess.square_rank(square)
    for enemy in enemy_pawns:
        if abs(chess.square_file(enemy) - file_index) > 1:
            continue
        enemy_rank = chess.square_rank(enemy)
        if color == chess.WHITE and enemy_rank > rank:
            return False
        if color == chess.BLACK and enemy_rank < rank:
            return False
    return True


def _is_candidate_passer(square, color, enemy_pawns, files):
    front = _front_square(square, color)
    if front is None:
        return False
    file_index = chess.square_file(square)
    if files[file_index] > 1:
        return False
    blockers = 0
    rank = chess.square_rank(square)
    for enemy in enemy_pawns:
        if abs(chess.square_file(enemy) - file_index) > 1:
            continue
        enemy_rank = chess.square_rank(enemy)
        if color == chess.WHITE and enemy_rank > rank:
            blockers += 1
        if color == chess.BLACK and enemy_rank < rank:
            blockers += 1
    return blockers <= 1


def _is_passed_after_push(board, move):
    mover = board.piece_at(move.from_square)
    if mover is None or mover.piece_type != chess.PAWN:
        return False
    enemy_pawns = board.pieces(chess.PAWN, not mover.color)
    return _is_passed_pawn(move.to_square, mover.color, enemy_pawns)


def _pawn_attack_mask(board, color):
    attacks = 0
    for square in board.pieces(chess.PAWN, color):
        attacks |= chess.BB_PAWN_ATTACKS[color][square]
    return attacks


def _attacks_mask(board, square):
    return int(board.attacks(square))


def _king_distance(square_a, square_b):
    return max(
        abs(chess.square_file(square_a) - chess.square_file(square_b)),
        abs(chess.square_rank(square_a) - chess.square_rank(square_b)),
    )


def _edge_bonus(square):
    file_index = chess.square_file(square)
    rank = chess.square_rank(square)
    file_edge = 3 - min(file_index, 7 - file_index)
    rank_edge = 3 - min(rank, 7 - rank)
    return max(file_edge, rank_edge)
