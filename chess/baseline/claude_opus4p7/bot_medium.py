"""Medium chess bot. Traditional alpha-beta, deliberately capped for play
around ~1400-1600 ELO (clearly above bot_easy.py, clearly below bot_hard.py).

Differences from bot_hard.py:
  - max search depth is 4 (iterative deepening 1 -> 2 -> 3 -> 4 only)
  - no transposition table, no killer / history move ordering
  - no null-move pruning, no late-move reductions
  - evaluation is material + tapered piece-square tables only
    (no bishop-pair, pawn-structure, or rook-file terms)
  - quiescence keeps captures and queen-promotion pushes only
  - check extension is bounded so the search cannot blow past the time budget
"""

from __future__ import annotations

import time

import chess


# ---------- search constants ----------

INF = 30_000
MATE = 20_000
MATE_THRESHOLD = MATE - 1000
MAX_PLY = 48

MAX_DEPTH = 4               # strength cap; never search deeper than this
DEFAULT_BUDGET = 1.0        # safety wall-clock budget (still well under 1.6 s)
TIME_HARD_RATIO = 0.85


# ---------- material and game phase ----------

PIECE_VALUE_MG = [0, 82, 337, 365, 477, 1025, 0]
PIECE_VALUE_EG = [0, 94, 281, 297, 512, 936, 0]
PHASE_INC = [0, 0, 1, 1, 2, 4, 0]
MAX_PHASE = 24


# ---------- piece-square tables (PeSTO) ----------
# Visual layout: index 0 = a8, index 56 = a1.
# Lookup: white piece on square sq uses PST[sq ^ 56]; black uses PST[sq].

_PST_MG_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_PST_MG_KNIGHT = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
]
_PST_MG_BISHOP = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]
_PST_MG_ROOK = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
]
_PST_MG_QUEEN = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]
_PST_MG_KING = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]

_PST_EG_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_PST_EG_KNIGHT = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]
_PST_EG_BISHOP = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
]
_PST_EG_ROOK = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
]
_PST_EG_QUEEN = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]
_PST_EG_KING = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]

_PST_MG = [None, _PST_MG_PAWN, _PST_MG_KNIGHT, _PST_MG_BISHOP,
           _PST_MG_ROOK, _PST_MG_QUEEN, _PST_MG_KING]
_PST_EG = [None, _PST_EG_PAWN, _PST_EG_KNIGHT, _PST_EG_BISHOP,
           _PST_EG_ROOK, _PST_EG_QUEEN, _PST_EG_KING]


def _build_pst():
    """Pre-fold (piece_value + PST) into per-(color, piece_type) lookup arrays.

    Indexed by python-chess square (a1 = 0 .. h8 = 63).
    """
    mg = [[None] * 7, [None] * 7]
    eg = [[None] * 7, [None] * 7]
    for pt in range(1, 7):
        w_mg = [0] * 64
        w_eg = [0] * 64
        b_mg = [0] * 64
        b_eg = [0] * 64
        for sq in range(64):
            w_mg[sq] = PIECE_VALUE_MG[pt] + _PST_MG[pt][sq ^ 56]
            w_eg[sq] = PIECE_VALUE_EG[pt] + _PST_EG[pt][sq ^ 56]
            b_mg[sq] = PIECE_VALUE_MG[pt] + _PST_MG[pt][sq]
            b_eg[sq] = PIECE_VALUE_EG[pt] + _PST_EG[pt][sq]
        mg[chess.WHITE][pt] = w_mg
        mg[chess.BLACK][pt] = b_mg
        eg[chess.WHITE][pt] = w_eg
        eg[chess.BLACK][pt] = b_eg
    return mg, eg


PST_FULL_MG, PST_FULL_EG = _build_pst()


class TimeUp(Exception):
    pass


class Engine:
    def __init__(self):
        self.nodes = 0
        self.start_time = 0.0
        self.time_hard = 0.0

    @staticmethod
    def evaluate(board):
        """Material + tapered PST eval, side-to-move perspective, centipawns."""
        mg_w = mg_b = eg_w = eg_b = 0
        phase = 0

        white_occ = board.occupied_co[chess.WHITE]
        black_occ = board.occupied_co[chess.BLACK]

        for pt in range(1, 7):
            ph = PHASE_INC[pt]
            w_mg_tab = PST_FULL_MG[chess.WHITE][pt]
            w_eg_tab = PST_FULL_EG[chess.WHITE][pt]
            b_mg_tab = PST_FULL_MG[chess.BLACK][pt]
            b_eg_tab = PST_FULL_EG[chess.BLACK][pt]

            if pt == chess.PAWN:
                pt_mask = board.pawns
            elif pt == chess.KNIGHT:
                pt_mask = board.knights
            elif pt == chess.BISHOP:
                pt_mask = board.bishops
            elif pt == chess.ROOK:
                pt_mask = board.rooks
            elif pt == chess.QUEEN:
                pt_mask = board.queens
            else:
                pt_mask = board.kings

            bb = pt_mask & white_occ
            while bb:
                lsb = bb & -bb
                sq = lsb.bit_length() - 1
                bb ^= lsb
                mg_w += w_mg_tab[sq]
                eg_w += w_eg_tab[sq]
                phase += ph

            bb = pt_mask & black_occ
            while bb:
                lsb = bb & -bb
                sq = lsb.bit_length() - 1
                bb ^= lsb
                mg_b += b_mg_tab[sq]
                eg_b += b_eg_tab[sq]
                phase += ph

        mg = mg_w - mg_b
        eg = eg_w - eg_b

        if phase > MAX_PHASE:
            phase = MAX_PHASE
        score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
        return score if board.turn == chess.WHITE else -score

    def _check_time(self):
        if (self.nodes & 1023) == 0:
            if time.perf_counter() - self.start_time > self.time_hard:
                raise TimeUp()

    @staticmethod
    def _score_move(board, move):
        """Lightweight move ordering: promotions > MVV-LVA captures > rest."""
        if move.promotion:
            base = 1_000_000 + PIECE_VALUE_MG[move.promotion]
            if board.is_capture(move):
                base += 800
            return base
        if board.is_capture(move):
            if board.is_en_passant(move):
                victim = chess.PAWN
            else:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
            attacker = board.piece_type_at(move.from_square) or 0
            return 500_000 + PIECE_VALUE_MG[victim] * 16 - attacker
        return 0

    def quiesce(self, board, alpha, beta, ply):
        self.nodes += 1
        self._check_time()

        if ply >= MAX_PLY:
            return self.evaluate(board)

        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            stand_pat = -INF
        else:
            stand_pat = self.evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            moves = list(board.generate_legal_captures())
            for m in board.generate_legal_moves():
                if m.promotion == chess.QUEEN and not board.is_capture(m):
                    moves.append(m)

        moves.sort(key=lambda m: self._score_move(board, m), reverse=True)

        best = stand_pat
        for move in moves:
            board.push(move)
            value = -self.quiesce(board, -beta, -alpha, ply + 1)
            board.pop()

            if value >= beta:
                return value
            if value > best:
                best = value
                if value > alpha:
                    alpha = value

        return best

    def search(self, board, depth, alpha, beta, ply):
        self.nodes += 1
        self._check_time()

        if ply > 0 and (
            board.is_repetition(2)
            or board.halfmove_clock >= 100
            or board.is_insufficient_material()
        ):
            return 0

        in_check = board.is_check()
        if in_check and depth < MAX_DEPTH + 2:
            depth += 1

        if depth <= 0:
            return self.quiesce(board, alpha, beta, ply)

        legal = list(board.generate_legal_moves())
        if not legal:
            return -MATE + ply if in_check else 0

        legal.sort(key=lambda m: self._score_move(board, m), reverse=True)

        best_value = -INF
        for move in legal:
            board.push(move)
            value = -self.search(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if value > best_value:
                best_value = value
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break
        return best_value

    def _search_root(self, board, depth, prev_best):
        legal = list(board.generate_legal_moves())
        if not legal:
            return 0, None

        # PV move from the previous iteration goes first.
        if prev_best in legal:
            legal.remove(prev_best)
            legal.insert(0, prev_best)
            rest_start = 1
        else:
            rest_start = 0
        rest = legal[rest_start:]
        rest.sort(key=lambda m: self._score_move(board, m), reverse=True)
        legal = legal[:rest_start] + rest

        alpha = -INF
        beta = INF
        best_value = -INF
        best_move = legal[0]

        for move in legal:
            board.push(move)
            value = -self.search(board, depth - 1, -beta, -alpha, 1)
            board.pop()

            if value > best_value:
                best_value = value
                best_move = move
                if value > alpha:
                    alpha = value

        return best_value, best_move

    def choose(self, board, budget=DEFAULT_BUDGET):
        self.start_time = time.perf_counter()
        self.time_hard = budget * TIME_HARD_RATIO
        self.nodes = 0

        legal = list(board.legal_moves)
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        best_move = legal[0]
        for depth in range(1, MAX_DEPTH + 1):
            try:
                value, move = self._search_root(board, depth, best_move)
                if move is not None:
                    best_move = move
                if abs(value) >= MATE_THRESHOLD:
                    break
            except TimeUp:
                break

        return best_move


class Bot:
    name = "claude_opus4p7_medium"

    def __init__(self):
        self.engine = Engine()

    def choose_action(self, state):
        legal = state["legal_actions"]
        if not legal:
            return ""
        if len(legal) == 1:
            return legal[0]
        board = chess.Board(state["fen"])
        move = self.engine.choose(board, budget=DEFAULT_BUDGET)
        if move is None:
            return legal[0]
        uci = move.uci()
        return uci if uci in legal else legal[0]


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        return ""
    if len(legal) == 1:
        return legal[0]
    board = chess.Board(state["fen"])
    move = _MODULE_ENGINE.choose(board, budget=DEFAULT_BUDGET)
    if move is None:
        return legal[0]
    uci = move.uci()
    return uci if uci in legal else legal[0]


_MODULE_ENGINE = Engine()
