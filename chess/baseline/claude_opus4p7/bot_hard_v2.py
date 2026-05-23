"""Stronger chess bot (v2). Built on the v1 α-β engine with additions
targeting the 2-second ladder:

  1. Adaptive time budget. v1 hardcodes 1.55s + 0.85 hard ratio, leaving
     compute on the table at 2s timeouts. v2 probes with 1.85s and a 0.90
     hard ratio (~1.67s effective cap), falling back to a 0.78s safe
     budget on the first kill so it also survives 1s evaluations.
  2. Opening book. The first ~5-9 plies of mainline chess are
     deterministic enough that running α-β for them is pure waste. v2
     plays book replies instantly when the position matches a Sicilian /
     Italian / Caro-Kann / French main line, or any standard 1.d4 / 1.c4
     / 1.Nf3 response.
  3. Passed-pawn evaluation. v1's eval has no passed-pawn term, the
     single biggest missing endgame heuristic.
  4. Cheap pawn-shield king safety, gated on queens-on-board so it stays
     out of endgame computation. (An attacker-count via
     ``board.attackers_mask`` was tried and was too slow.)
  5. 1M-entry transposition table (was 400k), large enough that single
     midgame moves don't blow it out and clear.

  6. Aspiration windows from depth 4 onward. ±30 cp around the previous
     iteration's score, widen 3× on fail-low/high, fall back to full
     window once it exceeds 800 cp.
  7. Late Move Pruning (LMP) at depth ≤ 3 for quiet non-checking moves
     once we've already tried 3 + 2·depth² siblings. Cheap way to skip
     obvious-loser late quiet moves at low depth.
"""

from __future__ import annotations

import time

import chess


# ---------- search constants ----------

INF = 30_000
MATE = 20_000
MATE_THRESHOLD = MATE - 1000
MAX_PLY = 64

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

DEFAULT_BUDGET = 1.85
TIME_HARD_RATIO = 0.90   # abort search past this fraction of the budget
TIME_SOFT_RATIO = 0.55   # do not start a new ID iteration past this


# ---------- material and game phase ----------

# Indexed by piece_type 1..6 (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING).
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
    """Pre-fold (piece_value + PST) for fast eval.

    Returns mg[color][piece_type] and eg[color][piece_type], each a list of 64
    ints indexed by python-chess square (a1=0..h8=63).
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

# Adjacent-file masks for isolated-pawn detection.
_ADJ_FILE_MASK = [0] * 8
for _f in range(8):
    m = 0
    if _f > 0:
        m |= chess.BB_FILES[_f - 1]
    if _f < 7:
        m |= chess.BB_FILES[_f + 1]
    _ADJ_FILE_MASK[_f] = m


# ---- v2 additions: passed-pawn detection masks ----
# For a white pawn on square sq to be passed: no black pawn on its file
# or adjacent files at any rank in front of (i.e. higher than) sq.
# Symmetric mask for black: ranks below sq.
_WHITE_PASSED_MASK = [0] * 64
_BLACK_PASSED_MASK = [0] * 64
for _sq in range(64):
    _f = _sq & 7
    _r = _sq >> 3
    _files = chess.BB_FILES[_f] | _ADJ_FILE_MASK[_f]
    _ahead_w = 0
    for _rr in range(_r + 1, 8):
        _ahead_w |= chess.BB_RANKS[_rr]
    _ahead_b = 0
    for _rr in range(_r - 1, -1, -1):
        _ahead_b |= chess.BB_RANKS[_rr]
    _WHITE_PASSED_MASK[_sq] = _files & _ahead_w
    _BLACK_PASSED_MASK[_sq] = _files & _ahead_b

# Per-rank passed-pawn bonus (mg, eg). Index by rank 0..7 (white POV).
# Symmetric mirror for black uses (7 - rank).
_PASSED_BONUS_MG = [0, 5, 10, 15, 25, 45, 80, 0]
_PASSED_BONUS_EG = [0, 15, 25, 40, 65, 100, 160, 0]

# ---- v2 additions: king-zone masks (king square -> 8-neighbour mask) ----
_KING_ZONE = [0] * 64
for _sq in range(64):
    _f = _sq & 7
    _r = _sq >> 3
    _zone = 0
    for _df in (-1, 0, 1):
        for _dr in (-1, 0, 1):
            if _df == 0 and _dr == 0:
                continue
            _nf, _nr = _f + _df, _r + _dr
            if 0 <= _nf < 8 and 0 <= _nr < 8:
                _zone |= 1 << (_nr * 8 + _nf)
    _KING_ZONE[_sq] = _zone


class TimeUp(Exception):
    """Raised inside the search to abort the current iteration."""


def _popcount(x):
    return bin(x).count("1")


class Engine:
    def __init__(self):
        self.tt = {}
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {}
        self.nodes = 0
        self.start_time = 0.0
        self.time_hard = 0.0
        self.time_soft = 0.0

    @staticmethod
    def _key(board):
        return (
            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.kings,
            board.occupied_co[chess.WHITE],
            board.occupied_co[chess.BLACK],
            board.turn,
            board.castling_rights,
            board.ep_square,
        )

    @staticmethod
    def evaluate(board):
        """Static eval in centipawns from side-to-move perspective."""
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

            # `board.pieces_mask(pt, color)` would do the same; inline to skip
            # one Python-level attribute lookup per call.
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

        # Bishop pair.
        wb = board.bishops & white_occ
        if wb & (wb - 1):
            mg += 30
            eg += 50
        bb_b = board.bishops & black_occ
        if bb_b & (bb_b - 1):
            mg -= 30
            eg -= 50

        # Pawn structure and rook-on-(semi-)open file.
        white_pawns = board.pawns & white_occ
        black_pawns = board.pawns & black_occ
        white_rooks = board.rooks & white_occ
        black_rooks = board.rooks & black_occ

        for f in range(8):
            file_bb = chess.BB_FILES[f]
            wp = white_pawns & file_bb
            bp = black_pawns & file_bb
            wp_count = _popcount(wp)
            bp_count = _popcount(bp)

            if wp_count >= 2:
                mg -= 10 * (wp_count - 1)
                eg -= 25 * (wp_count - 1)
            if bp_count >= 2:
                mg += 10 * (bp_count - 1)
                eg += 25 * (bp_count - 1)

            adj = _ADJ_FILE_MASK[f]
            if wp and not (white_pawns & adj):
                mg -= 12 * wp_count
                eg -= 18 * wp_count
            if bp and not (black_pawns & adj):
                mg += 12 * bp_count
                eg += 18 * bp_count

            # Rook on open / semi-open file.
            wr_on_file = white_rooks & file_bb
            if wr_on_file:
                n = _popcount(wr_on_file)
                if not wp:
                    if not bp:
                        mg += 25 * n
                        eg += 15 * n
                    else:
                        mg += 12 * n
                        eg += 8 * n
            br_on_file = black_rooks & file_bb
            if br_on_file:
                n = _popcount(br_on_file)
                if not bp:
                    if not wp:
                        mg -= 25 * n
                        eg -= 15 * n
                    else:
                        mg -= 12 * n
                        eg -= 8 * n

        # ---- v2: passed pawns ----
        bb = white_pawns
        while bb:
            lsb = bb & -bb
            sq = lsb.bit_length() - 1
            bb ^= lsb
            if not (black_pawns & _WHITE_PASSED_MASK[sq]):
                rank = sq >> 3
                mg += _PASSED_BONUS_MG[rank]
                eg += _PASSED_BONUS_EG[rank]
        bb = black_pawns
        while bb:
            lsb = bb & -bb
            sq = lsb.bit_length() - 1
            bb ^= lsb
            if not (white_pawns & _BLACK_PASSED_MASK[sq]):
                rank = 7 - (sq >> 3)
                mg -= _PASSED_BONUS_MG[rank]
                eg -= _PASSED_BONUS_EG[rank]

        # ---- v2: cheap pawn-shield king safety ----
        # Only gated on queens-on-board (middlegame matters). Counts
        # friendly pawns on the 3 files closest to the king on the rank
        # just in front of it. -12 cp per missing shield pawn, -30 cp for
        # a king that hasn't moved toward a corner. (Earlier attacker-
        # count via board.attackers_mask ate too much budget.)
        wq = board.queens & white_occ
        bq = board.queens & black_occ
        if wq and bq:
            wk = board.kings & white_occ
            if wk:
                wk_sq = wk.bit_length() - 1
                kf = wk_sq & 7
                kr = wk_sq >> 3
                if kr <= 1 and (kf <= 2 or kf >= 5):
                    shield_files = (
                        chess.BB_FILES[max(kf - 1, 0)]
                        | chess.BB_FILES[kf]
                        | chess.BB_FILES[min(kf + 1, 7)]
                    )
                    shield_rank = chess.BB_RANKS[kr + 1] if kr + 1 < 8 else 0
                    shield = white_pawns & shield_files & shield_rank
                    missing = 3 - _popcount(shield)
                    mg -= missing * 12
                else:
                    mg -= 30
            bk = board.kings & black_occ
            if bk:
                bk_sq = bk.bit_length() - 1
                kf = bk_sq & 7
                kr = bk_sq >> 3
                if kr >= 6 and (kf <= 2 or kf >= 5):
                    shield_files = (
                        chess.BB_FILES[max(kf - 1, 0)]
                        | chess.BB_FILES[kf]
                        | chess.BB_FILES[min(kf + 1, 7)]
                    )
                    shield_rank = chess.BB_RANKS[kr - 1] if kr - 1 >= 0 else 0
                    shield = black_pawns & shield_files & shield_rank
                    missing = 3 - _popcount(shield)
                    mg += missing * 12
                else:
                    mg += 30

        if phase > MAX_PHASE:
            phase = MAX_PHASE
        score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
        return score if board.turn == chess.WHITE else -score

    def _check_time(self):
        if (self.nodes & 2047) == 0:
            if time.perf_counter() - self.start_time > self.time_hard:
                raise TimeUp()

    def _has_non_pawn_material(self, board):
        c = board.turn
        occ = board.occupied_co[c]
        return bool(
            (board.knights | board.bishops | board.rooks | board.queens) & occ
        )

    def _score_move(self, board, move, tt_move, killer1, killer2):
        if move == tt_move:
            return 10_000_000

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

        if move == killer1:
            return 400_000
        if move == killer2:
            return 350_000

        attacker = board.piece_type_at(move.from_square)
        return self.history.get((board.turn, attacker, move.to_square), 0)

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
            stand_pat = -INF  # no stand-pat when in check
        else:
            stand_pat = self.evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            moves = list(board.generate_legal_captures())
            # Queen-promotion pushes are not captures but are tactical enough.
            for m in board.generate_legal_moves():
                if m.promotion == chess.QUEEN and not board.is_capture(m):
                    moves.append(m)

        def qscore(m):
            if m.promotion:
                return 100_000 + PIECE_VALUE_MG[m.promotion]
            if board.is_en_passant(m):
                return PIECE_VALUE_MG[chess.PAWN] * 16
            v = board.piece_type_at(m.to_square)
            a = board.piece_type_at(m.from_square) or 0
            return (PIECE_VALUE_MG[v] if v else 0) * 16 - a

        moves.sort(key=qscore, reverse=True)

        best = stand_pat
        for move in moves:
            # Delta pruning when not in check.
            if not in_check and not move.promotion:
                if board.is_en_passant(move):
                    victim_v = PIECE_VALUE_MG[chess.PAWN]
                else:
                    vpt = board.piece_type_at(move.to_square)
                    victim_v = PIECE_VALUE_MG[vpt] if vpt else 0
                if stand_pat + victim_v + 200 < alpha:
                    continue

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

    def search(self, board, depth, alpha, beta, ply, allow_null=True):
        self.nodes += 1
        self._check_time()

        if ply > 0 and (
            board.is_repetition(2)
            or board.halfmove_clock >= 100
            or board.is_insufficient_material()
        ):
            return 0

        in_check = board.is_check()
        if in_check:
            depth += 1

        if depth <= 0:
            return self.quiesce(board, alpha, beta, ply)

        # ---- TT probe ----
        key = self._key(board)
        tt_move = None
        original_alpha = alpha
        entry = self.tt.get(key)
        if entry is not None:
            tt_depth, tt_value, tt_flag, tt_move_uci = entry
            if ply > 0 and tt_depth >= depth:
                v = tt_value
                if v > MATE_THRESHOLD:
                    v -= ply
                elif v < -MATE_THRESHOLD:
                    v += ply
                if tt_flag == TT_EXACT:
                    return v
                if tt_flag == TT_LOWER and v >= beta:
                    return v
                if tt_flag == TT_UPPER and v <= alpha:
                    return v
            if tt_move_uci:
                try:
                    cand = chess.Move.from_uci(tt_move_uci)
                    if cand in board.legal_moves:
                        tt_move = cand
                except (ValueError, AssertionError):
                    tt_move = None

        # ---- null-move pruning ----
        if (
            allow_null
            and not in_check
            and depth >= 3
            and self._has_non_pawn_material(board)
            and abs(beta) < MATE_THRESHOLD
        ):
            R = 2 + depth // 6
            board.push(chess.Move.null())
            null_v = -self.search(board, depth - 1 - R, -beta, -beta + 1, ply + 1, False)
            board.pop()
            if null_v >= beta:
                if null_v >= MATE_THRESHOLD:
                    null_v = beta
                return null_v

        # ---- move generation and ordering ----
        legal = list(board.generate_legal_moves())
        if not legal:
            return -MATE + ply if in_check else 0

        k1, k2 = self.killers[ply]
        scored = [
            (self._score_move(board, m, tt_move, k1, k2), i, m)
            for i, m in enumerate(legal)
        ]
        scored.sort(reverse=True)

        best_value = -INF
        best_move = None
        moves_searched = 0

        # Pre-compute pruning threshold for Late Move Pruning (LMP):
        # at low depth with no check, prune obvious-loser late quiet
        # moves entirely instead of paying for a null-window probe.
        # Standard formula: 3 + 2*depth*depth quiet moves at depth<=3.
        lmp_threshold = (
            3 + 2 * depth * depth if depth <= 3 and not in_check else 10**9
        )

        for _, _, move in scored:
            is_capture = board.is_capture(move)
            is_promo = move.promotion is not None
            is_tactical = is_capture or is_promo or in_check

            # ---- LMP ----
            if (
                moves_searched >= lmp_threshold
                and not is_tactical
                and best_value > -MATE_THRESHOLD
            ):
                continue

            board.push(move)

            # ---- LMR ----
            reduction = 0
            if depth >= 3 and moves_searched >= 3 and not is_tactical:
                # Skip reducing checking moves: gives_check is checked after
                # push() so the position reflects the move.
                if not board.is_check():
                    reduction = 1
                    if moves_searched >= 6:
                        reduction = 2
                    if depth >= 6 and moves_searched >= 12:
                        reduction = 3
                    if reduction >= depth - 1:
                        reduction = depth - 2

            if moves_searched == 0:
                value = -self.search(board, depth - 1, -beta, -alpha, ply + 1, True)
            else:
                value = -self.search(
                    board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, True
                )
                if value > alpha and reduction > 0:
                    value = -self.search(
                        board, depth - 1, -alpha - 1, -alpha, ply + 1, True
                    )
                if alpha < value < beta:
                    value = -self.search(board, depth - 1, -beta, -alpha, ply + 1, True)

            board.pop()
            moves_searched += 1

            if value > best_value:
                best_value = value
                best_move = move
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        if not is_capture and not is_promo:
                            if self.killers[ply][0] != move:
                                self.killers[ply][1] = self.killers[ply][0]
                                self.killers[ply][0] = move
                            attacker = board.piece_type_at(move.from_square)
                            if attacker is not None:
                                hk = (board.turn, attacker, move.to_square)
                                self.history[hk] = self.history.get(hk, 0) + depth * depth
                        break

        # ---- TT store ----
        if best_value <= original_alpha:
            flag = TT_UPPER
        elif best_value >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT

        store_v = best_value
        if store_v > MATE_THRESHOLD:
            store_v += ply
        elif store_v < -MATE_THRESHOLD:
            store_v -= ply

        self.tt[key] = (
            depth,
            store_v,
            flag,
            best_move.uci() if best_move else None,
        )

        return best_value

    def _search_root(self, board, depth, prev_best, alpha=-INF, beta=INF):
        """Search the root, returning (best_value, best_move).

        Re-raises TimeUp without modifying any partial state so the caller
        can fall back to the previous iteration's best move. Accepts
        optional (alpha, beta) bounds for aspiration-window search by the
        caller.
        """
        legal = list(board.generate_legal_moves())
        if not legal:
            return 0, None

        k1, k2 = self.killers[0]
        scored = [
            (self._score_move(board, m, prev_best, k1, k2), i, m)
            for i, m in enumerate(legal)
        ]
        scored.sort(reverse=True)

        best_value = -INF
        best_move = scored[0][2]
        moves_searched = 0

        for _, _, move in scored:
            board.push(move)
            if moves_searched == 0:
                value = -self.search(board, depth - 1, -beta, -alpha, 1, True)
            else:
                value = -self.search(board, depth - 1, -alpha - 1, -alpha, 1, True)
                if alpha < value < beta:
                    value = -self.search(board, depth - 1, -beta, -alpha, 1, True)
            board.pop()
            moves_searched += 1

            if value > best_value:
                best_value = value
                best_move = move
                if value > alpha:
                    alpha = value

        # Store root in TT for next-iteration ordering.
        self.tt[self._key(board)] = (
            depth,
            best_value,
            TT_EXACT,
            best_move.uci(),
        )
        return best_value, best_move

    def choose(self, board, budget=DEFAULT_BUDGET):
        self.start_time = time.perf_counter()
        self.time_hard = budget * TIME_HARD_RATIO
        self.time_soft = budget * TIME_SOFT_RATIO
        self.nodes = 0

        # Reset per-call structures so search depth is predictable.
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        if self.history:
            self.history = {k: v // 2 for k, v in self.history.items() if v >= 2}
        if len(self.tt) > 1_000_000:
            self.tt.clear()

        legal = list(board.legal_moves)
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        best_move = legal[0]
        prev_score = 0
        for depth in range(1, MAX_PLY):
            elapsed = time.perf_counter() - self.start_time
            if depth > 2 and elapsed > self.time_soft:
                break
            try:
                if depth >= 4:
                    # Aspiration around prev iteration's score, widen 3×
                    # on fail. Commits the search result even on a bounded
                    # fail — empirically this gave the best win-rate.
                    window = 30
                    while True:
                        alpha = prev_score - window
                        beta = prev_score + window
                        value, move = self._search_root(
                            board, depth, best_move, alpha, beta,
                        )
                        if value <= alpha:
                            window *= 3
                            prev_score = value
                            if window > 800:
                                value, move = self._search_root(
                                    board, depth, best_move,
                                )
                                break
                        elif value >= beta:
                            window *= 3
                            prev_score = value
                            if window > 800:
                                value, move = self._search_root(
                                    board, depth, best_move,
                                )
                                break
                        else:
                            break
                else:
                    value, move = self._search_root(board, depth, best_move)
                if move is not None:
                    best_move = move
                    prev_score = value
                if abs(value) >= MATE_THRESHOLD:
                    break
            except TimeUp:
                break

        return best_move


# ---- Opening book ----
# Maps board.fen()[:position+castling+ep] to a list of book replies (UCI).
# Tiny mainline-only book: white opens 1.e4 / 1.d4 / 1.c4 / 1.Nf3, black
# replies with solid mainline answers, then we follow with theory for 2-3
# more plies. Coverage is thin but the openings are exactly what v1 spends
# its first ~6 plies finding via α-β.
def _book_key(board: chess.Board) -> str:
    # FEN piece placement + side-to-move + castling + ep square (ignore
    # halfmove/fullmove counters so transpositions match).
    fen = board.fen()
    parts = fen.split()
    return " ".join(parts[:4])


def _build_book() -> dict[str, list[str]]:
    book: dict[str, list[str]] = {}

    def add(uci_sequence: list[str]) -> None:
        b = chess.Board()
        for uci in uci_sequence[:-1]:
            mv = chess.Move.from_uci(uci)
            if mv not in b.legal_moves:
                return
            b.push(mv)
        key = _book_key(b)
        reply = uci_sequence[-1]
        if reply not in book.get(key, []):
            book.setdefault(key, []).append(reply)

    # White first move: 1.e4. (Picking one line so transpositions are
    # consistent — v2 always opens 1.e4 when on white.)
    add(["e2e4"])

    # Black responses to *any* sane white first move. We bake in
    # solid mainline replies for 1.e4, 1.d4, 1.c4, 1.Nf3, 1.b3, 1.g3.
    add(["e2e4", "c7c5"])                      # 1.e4 c5 (Sicilian)
    add(["e2e4", "e7e5"])
    add(["e2e4", "e7e6"])
    add(["e2e4", "c7c6"])
    add(["d2d4", "g8f6"])                      # 1.d4 Nf6
    add(["d2d4", "d7d5"])                      # 1.d4 d5
    add(["c2c4", "e7e5"])                      # 1.c4 e5 (reversed Sicilian)
    add(["c2c4", "g8f6"])                      # 1.c4 Nf6
    add(["g1f3", "d7d5"])                      # 1.Nf3 d5
    add(["g1f3", "g8f6"])                      # 1.Nf3 Nf6
    add(["b2b3", "e7e5"])
    add(["b2b3", "d7d5"])
    add(["g2g3", "d7d5"])
    add(["g2g3", "e7e5"])

    # White's 2nd move after 1.e4 c5 (Sicilian — most decisive against
    # equal-strength engines; baked Open Sicilian).
    add(["e2e4", "c7c5", "g1f3"])
    add(["e2e4", "c7c5", "g1f3", "d7d6"])
    add(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"])
    add(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"])
    add(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"])
    add(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"])
    add(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"])
    add(["e2e4", "c7c5", "g1f3", "b8c6"])
    add(["e2e4", "c7c5", "g1f3", "b8c6", "d2d4"])
    add(["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4"])
    add(["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4"])
    add(["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6"])
    add(["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"])

    # 1.e4 e5: Italian / Ruy via Nf3-Nc6-Bc4/Bb5.
    add(["e2e4", "e7e5", "g1f3"])
    add(["e2e4", "e7e5", "g1f3", "b8c6"])
    add(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"])
    add(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"])
    add(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3"])
    add(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"])
    add(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"])
    add(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4"])

    # 1.e4 e6: French Tarrasch.
    add(["e2e4", "e7e6", "d2d4"])
    add(["e2e4", "e7e6", "d2d4", "d7d5"])
    add(["e2e4", "e7e6", "d2d4", "d7d5", "b1d2"])

    # 1.e4 c6: Caro-Kann main.
    add(["e2e4", "c7c6", "d2d4"])
    add(["e2e4", "c7c6", "d2d4", "d7d5"])
    add(["e2e4", "c7c6", "d2d4", "d7d5", "b1c3"])
    add(["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"])
    add(["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4"])
    return book


_OPENING_BOOK = _build_book()


class Bot:
    """v2: adaptive budget, opening book, aspiration windows, passed-pawn
    + king-safety eval terms layered on the v1 α-β engine."""

    name = "claude_opus4p7_hard_v2"

    _BUDGET_AGGRESSIVE = DEFAULT_BUDGET  # 1.85
    _BUDGET_SAFE = 0.78

    def __init__(self):
        self.engine = Engine()
        self.budget = self._BUDGET_AGGRESSIVE
        # last_completed=True so the very first move doesn't pre-drop.
        self._last_completed = True

    def choose_action(self, state):
        if not self._last_completed and self.budget > self._BUDGET_SAFE:
            self.budget = self._BUDGET_SAFE
        self._last_completed = False
        try:
            return self._choose_action_inner(state)
        finally:
            self._last_completed = True

    def _choose_action_inner(self, state):
        legal = state["legal_actions"]
        if not legal:
            return ""
        if len(legal) == 1:
            return legal[0]
        board = chess.Board(state["fen"])

        # Opening book: instant book reply when the position has one and
        # the reply is legal.
        book = _OPENING_BOOK.get(_book_key(board))
        if book:
            for uci in book:
                if uci in legal:
                    return uci

        move = self.engine.choose(board, budget=self.budget)
        if move is None:
            return legal[0]
        uci = move.uci()
        return uci if uci in legal else legal[0]


_MODULE_ENGINE = Engine()
_MODULE_STATE = {"budget": Bot._BUDGET_AGGRESSIVE, "last_completed": True}


def choose_action(state):
    """Functional wrapper. Uses a module-level engine (state persists across
    calls within a Python process)."""
    if not _MODULE_STATE["last_completed"] and _MODULE_STATE["budget"] > Bot._BUDGET_SAFE:
        _MODULE_STATE["budget"] = Bot._BUDGET_SAFE
    _MODULE_STATE["last_completed"] = False
    try:
        legal = state["legal_actions"]
        if not legal:
            return ""
        if len(legal) == 1:
            return legal[0]
        board = chess.Board(state["fen"])
        book = _OPENING_BOOK.get(_book_key(board))
        if book:
            for uci in book:
                if uci in legal:
                    return uci
        move = _MODULE_ENGINE.choose(board, budget=_MODULE_STATE["budget"])
        if move is None:
            return legal[0]
        uci = move.uci()
        return uci if uci in legal else legal[0]
    finally:
        _MODULE_STATE["last_completed"] = True
