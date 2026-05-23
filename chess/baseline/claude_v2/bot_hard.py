"""claude_v2 / bot_hard.py

Hand-written chess engine. No code shared with the other baseline bots.

Search
  - Principal Variation Search (PVS) inside iterative deepening.
  - Alpha-beta with a transposition table keyed by python-chess's internal
    transposition key (fast, no zobrist hash recomputation per node).
  - Null-move pruning (R = 2 + depth // 4) when we have non-pawn material.
  - Late-move reductions on quiet non-check non-capture moves.
  - Check extensions of one ply.
  - Quiescence search over captures and queen promotions (and all legal
    moves when in check) with stand-pat and MVV-LVA ordering.
  - Mate-distance pruning at the root of every internal call.

Move ordering
  - TT move first, then captures by MVV-LVA, then promotions, then killer
    moves, then a history heuristic indexed by (from, to).

Evaluation
  - Tapered (middlegame / endgame) piece-square tables, PESTO weights.
  - Bishop pair, simple king pawn-shield, passed-pawn bonus, rook on open
    file, doubled-pawn penalty, side-to-move tempo.
"""

from __future__ import annotations

import sys
import time

import chess


# Search uses moderately deep recursion (check extensions + qsearch chains).
# Give Python enough head-room; we still cap ply ourselves inside _pvs.
if sys.getrecursionlimit() < 4000:
    sys.setrecursionlimit(4000)


# ---------------------------------------------------------------------------
# Evaluation tables
# ---------------------------------------------------------------------------

INF = 10_000_000
MATE = 1_000_000
MATE_IN_MAX = MATE - 1024

# PESTO-style tapered piece values.
MG_VALUE = {
    chess.PAWN: 82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK: 477,
    chess.QUEEN: 1025,
    chess.KING: 0,
}
EG_VALUE = {
    chess.PAWN: 94,
    chess.KNIGHT: 281,
    chess.BISHOP: 297,
    chess.ROOK: 512,
    chess.QUEEN: 936,
    chess.KING: 0,
}

# Game-phase increments — total at the start of the game is 24.
PHASE_INC = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
MAX_PHASE = 24

# All PSTs are written from white's perspective using the python-chess
# square index where 0 = a1 and 63 = h8 (rank-then-file). For black, we
# look up the mirrored square (sq XOR 56).

PAWN_MG = [
      0,   0,   0,   0,   0,   0,   0,   0,
    -35,  -1, -20, -23, -15,  24,  38, -22,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -14,  13,   6,  21,  23,  12,  17, -23,
     -6,   7,  26,  31,  65,  56,  25, -20,
     98, 134,  61,  95,  68, 126,  34, -11,
      0,   0,   0,   0,   0,   0,   0,   0,
]
PAWN_EG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     13,   8,   8,  10,  13,   0,   2,  -7,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
     32,  24,  13,   5,  -2,   4,  17,  17,
     94, 100,  85,  67,  56,  53,  82,  84,
    178, 173, 158, 134, 147, 132, 165, 187,
      0,   0,   0,   0,   0,   0,   0,   0,
]
KNIGHT_MG = [
   -105, -21, -58, -33, -17, -28, -19, -23,
    -29, -53, -12,  -3,  -1,  18, -14, -19,
    -23,  -9,  12,  10,  19,  17,  25, -16,
    -13,   4,  16,  13,  28,  19,  21,  -8,
     -9,  17,  19,  53,  37,  69,  18,  22,
    -47,  60,  37,  65,  84, 129,  73,  44,
    -73, -41,  72,  36,  23,  62,   7, -17,
   -167, -89, -34, -49,  61, -97, -15, -107,
]
KNIGHT_EG = [
    -29, -51, -23, -15, -22, -18, -50, -64,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -58, -38, -13, -28, -31, -27, -63, -99,
]
BISHOP_MG = [
    -33,  -3, -14, -21, -13, -12, -39, -21,
      4,  15,  16,   0,   7,  21,  33,   1,
      0,  15,  15,  15,  14,  27,  18,  10,
     -6,  13,  13,  26,  34,  12,  10,   4,
     -4,   5,  19,  50,  37,  37,   7,  -2,
    -16,  37,  43,  40,  35,  50,  37,  -2,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -29,   4, -82, -37, -25, -42,   7,  -8,
]
BISHOP_EG = [
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
     -3,   9,  12,   9,  14,  10,   3,   2,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
]
ROOK_MG = [
    -19, -13,   1,  17,  16,   7, -37, -26,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -24, -11,   7,  26,  24,  35,  -8, -20,
     -5,  19,  26,  36,  17,  45,  61,  16,
     27,  32,  58,  62,  80,  67,  26,  44,
     32,  42,  32,  51,  63,   9,  31,  43,
]
ROOK_EG = [
     -9,   2,   3,  -1,  -5, -13,   4, -20,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
      4,   3,  13,   1,   2,   1,  -1,   2,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
     11,  13,  13,  11,  -3,   3,   8,   3,
     13,  10,  18,  15,  12,  12,   8,   5,
]
QUEEN_MG = [
     -1, -18,  -9,  10, -15, -25, -31, -50,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -28,   0,  29,  12,  59,  44,  43,  45,
]
QUEEN_EG = [
    -33, -28, -22, -43,  -5, -32, -20, -41,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -18,  28,  19,  47,  31,  34,  39,  23,
      3,  22,  24,  45,  57,  40,  57,  36,
    -20,   6,   9,  49,  47,  35,  19,   9,
    -17,  20,  32,  41,  58,  25,  30,   0,
     -9,  22,  22,  27,  27,  19,  10,  20,
]
KING_MG = [
    -15,  36,  12, -54,   8, -28,  24,  14,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -14, -14, -22, -46, -44, -30, -15, -27,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -17, -20, -12, -27, -30, -25, -14, -36,
     -9,  24,   2, -16, -20,   6,  22, -22,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
    -65,  23,  16, -15, -56, -34,   2,  13,
]
KING_EG = [
    -53, -34, -21, -11, -28, -14, -24, -43,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -18,  -4,  21,  24,  27,  23,   9, -11,
     -8,  22,  24,  27,  26,  33,  26,   3,
     10,  17,  23,  15,  20,  45,  44,  13,
    -12,  17,  14,  17,  17,  38,  23,  11,
    -74, -35, -18, -18, -11,  15,   4, -17,
]

PST_MG = {
    chess.PAWN: PAWN_MG,
    chess.KNIGHT: KNIGHT_MG,
    chess.BISHOP: BISHOP_MG,
    chess.ROOK: ROOK_MG,
    chess.QUEEN: QUEEN_MG,
    chess.KING: KING_MG,
}
PST_EG = {
    chess.PAWN: PAWN_EG,
    chess.KNIGHT: KNIGHT_EG,
    chess.BISHOP: BISHOP_EG,
    chess.ROOK: ROOK_EG,
    chess.QUEEN: QUEEN_EG,
    chess.KING: KING_EG,
}

FILE_OF = [chess.square_file(sq) for sq in range(64)]
RANK_OF = [chess.square_rank(sq) for sq in range(64)]

# Bitmasks for pawn structure: file mask and "passed pawn" front-spans.
FILE_BB = [chess.BB_FILES[f] for f in range(8)]


def _adjacent_files(f: int) -> int:
    mask = 0
    if f > 0:
        mask |= FILE_BB[f - 1]
    if f < 7:
        mask |= FILE_BB[f + 1]
    return mask


ADJACENT_FILES_BB = [_adjacent_files(f) for f in range(8)]


def _passed_mask(color: bool, sq: int) -> int:
    f = FILE_OF[sq]
    r = RANK_OF[sq]
    mask = FILE_BB[f] | ADJACENT_FILES_BB[f]
    if color == chess.WHITE:
        # All ranks strictly in front (toward rank 8).
        ahead = 0
        for rr in range(r + 1, 8):
            ahead |= chess.BB_RANKS[rr]
    else:
        ahead = 0
        for rr in range(0, r):
            ahead |= chess.BB_RANKS[rr]
    return mask & ahead


PASSED_W = [_passed_mask(chess.WHITE, sq) for sq in range(64)]
PASSED_B = [_passed_mask(chess.BLACK, sq) for sq in range(64)]

PASSED_BONUS_MG = [0, 5, 10, 20, 35, 60, 100, 0]   # by rank from own side
PASSED_BONUS_EG = [0, 10, 20, 35, 60, 100, 160, 0]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Weight per attacker type when scoring pressure on the king zone.
_KZ_WEIGHT = {
    chess.PAWN: 0,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 5,
    chess.KING: 0,
}
# Lookup table that turns total weighted attacker count into a penalty.
# Calibrated to top out around two pawns of penalty for an overwhelming
# attack — any more and the engine becomes too risk-averse to open lines.
_KZ_TABLE = [
      0,   0,   1,   2,   4,   7,  11,  16,
     22,  30,  40,  50,  60,  72,  84,  96,
    108, 120, 132, 144, 156, 168, 178, 188,
    196, 202, 206, 210, 212, 214, 215, 215,
]


def _king_zone_pressure(board: chess.Board, color: bool, king_sq: int) -> int:
    """Return a non-negative penalty representing pressure on `color`'s king."""
    # King zone = king's square plus the 8 adjacent squares.
    zone = chess.BB_KING_ATTACKS[king_sq] | chess.BB_SQUARES[king_sq]
    enemy = not color
    weight = 0
    occ_enemy = board.occupied_co[enemy]
    if not occ_enemy:
        return 0
    # For every enemy non-pawn piece, see how many king-zone squares it
    # attacks. Each attacked square adds the piece's weight.
    pieces = occ_enemy & ~board.pawns
    sq_iter = chess.scan_forward(pieces)
    for sq in sq_iter:
        attacks = board.attacks_mask(sq) & zone
        if attacks:
            piece = board.piece_at(sq)
            if piece is not None:
                weight += _KZ_WEIGHT[piece.piece_type] * chess.popcount(attacks)
    if weight <= 0:
        return 0
    if weight >= len(_KZ_TABLE):
        weight = len(_KZ_TABLE) - 1
    return _KZ_TABLE[weight]


def evaluate(board: chess.Board) -> int:
    """Static evaluation, returned from the side-to-move's perspective."""
    mg = 0
    eg = 0
    phase = 0
    wb = 0
    bb = 0

    white_pawns = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns = board.pawns & board.occupied_co[chess.BLACK]

    for sq, piece in board.piece_map().items():
        pt = piece.piece_type
        if piece.color == chess.WHITE:
            mg += MG_VALUE[pt] + PST_MG[pt][sq]
            eg += EG_VALUE[pt] + PST_EG[pt][sq]
            if pt == chess.BISHOP:
                wb += 1
            elif pt == chess.PAWN:
                if (PASSED_W[sq] & black_pawns) == 0:
                    r = RANK_OF[sq]
                    mg += PASSED_BONUS_MG[r]
                    eg += PASSED_BONUS_EG[r]
            elif pt == chess.ROOK:
                f = FILE_OF[sq]
                file_mask = FILE_BB[f]
                if (file_mask & board.pawns) == 0:
                    mg += 25
                    eg += 15
                elif (file_mask & white_pawns) == 0:
                    mg += 12
                    eg += 8
        else:
            m = sq ^ 56
            mg -= MG_VALUE[pt] + PST_MG[pt][m]
            eg -= EG_VALUE[pt] + PST_EG[pt][m]
            if pt == chess.BISHOP:
                bb += 1
            elif pt == chess.PAWN:
                if (PASSED_B[sq] & white_pawns) == 0:
                    r = 7 - RANK_OF[sq]
                    mg -= PASSED_BONUS_MG[r]
                    eg -= PASSED_BONUS_EG[r]
            elif pt == chess.ROOK:
                f = FILE_OF[sq]
                file_mask = FILE_BB[f]
                if (file_mask & board.pawns) == 0:
                    mg -= 25
                    eg -= 15
                elif (file_mask & black_pawns) == 0:
                    mg -= 12
                    eg -= 8
        phase += PHASE_INC[pt]

    # Bishop pair.
    if wb >= 2:
        mg += 30
        eg += 50
    if bb >= 2:
        mg -= 30
        eg -= 50

    # Doubled pawns (count pairs per file).
    for f in range(8):
        wcnt = chess.popcount(FILE_BB[f] & white_pawns)
        if wcnt >= 2:
            mg -= 12 * (wcnt - 1)
            eg -= 18 * (wcnt - 1)
        bcnt = chess.popcount(FILE_BB[f] & black_pawns)
        if bcnt >= 2:
            mg += 12 * (bcnt - 1)
            eg += 18 * (bcnt - 1)

    # King safety: pawn shield + attacker-zone pressure (middlegame only).
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is not None:
        shield_mask = chess.BB_KING_ATTACKS[wk] & white_pawns
        mg += 8 * chess.popcount(shield_mask)
        mg -= _king_zone_pressure(board, chess.WHITE, wk)
    if bk is not None:
        shield_mask = chess.BB_KING_ATTACKS[bk] & black_pawns
        mg -= 8 * chess.popcount(shield_mask)
        mg += _king_zone_pressure(board, chess.BLACK, bk)

    if phase > MAX_PHASE:
        phase = MAX_PHASE
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE

    # Side-to-move tempo.
    score += 14 if board.turn == chess.WHITE else -14

    return score if board.turn == chess.WHITE else -score


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


class _Stop(Exception):
    """Raised internally to abort search when the deadline expires."""


class Search:
    """Stateful searcher reused for one root call (iterative deepening)."""

    QS_MAX_DEPTH = 24
    MAX_PLY = 96
    # We treat repetition / 50-move draws as worse than equal for the side
    # we're moving for at the root, so the engine prefers playing on when
    # the position is roughly balanced rather than shuffling into a draw.
    CONTEMPT = 25

    def __init__(self, board: chess.Board, deadline: float) -> None:
        self.board = board
        self.deadline = deadline
        # tt key -> (depth, score, flag, move, age)
        self.tt: dict = {}
        self.killers: list[list] = []
        self.history: dict = {}
        self.nodes = 0
        self.stopped = False
        # Root halfmove counter so we can produce mate-in-N scores correctly.
        self._root_ply = board.ply()
        # The side we are searching for at the root — used for contempt.
        self._root_color = board.turn

    def _killers_at(self, ply: int) -> list:
        while len(self.killers) <= ply:
            self.killers.append([None, None])
        return self.killers[ply]

    def _draw_score(self) -> int:
        # Score is returned from the current side-to-move's POV.
        if self.board.turn == self._root_color:
            return -self.CONTEMPT
        return self.CONTEMPT

    # -- timing ------------------------------------------------------------

    def _check_time(self) -> None:
        # Sample the clock once every 1024 nodes to keep overhead small.
        if (self.nodes & 1023) == 0 and time.monotonic() > self.deadline:
            self.stopped = True
            raise _Stop()
        if self.stopped:
            raise _Stop()

    # -- move ordering -----------------------------------------------------

    def _score_move(self, m: chess.Move, ply: int, tt_move) -> int:
        b = self.board
        if tt_move is not None and m == tt_move:
            return 1_000_000
        if b.is_capture(m):
            victim = b.piece_at(m.to_square)
            if victim is None:  # en passant
                v_val = MG_VALUE[chess.PAWN]
            else:
                v_val = MG_VALUE[victim.piece_type]
            attacker = b.piece_at(m.from_square)
            a_val = MG_VALUE[attacker.piece_type] if attacker else 0
            base = 800_000 if v_val >= a_val else 200_000
            return base + v_val * 16 - a_val
        if m.promotion:
            return 700_000 + MG_VALUE.get(m.promotion, 0)
        if ply < len(self.killers):
            kl = self.killers[ply]
            if kl[0] == m:
                return 600_000
            if kl[1] == m:
                return 500_000
        return self.history.get((m.from_square, m.to_square), 0)

    def _order(self, moves, ply, tt_move):
        moves.sort(key=lambda mv: -self._score_move(mv, ply, tt_move))
        return moves

    # -- quiescence --------------------------------------------------------

    def _qsearch(self, alpha: int, beta: int, ply: int, qdepth: int = 0) -> int:
        self.nodes += 1
        self._check_time()

        b = self.board
        in_check = b.is_check()
        # Hard cap on quiescence depth — pathological capture chains can
        # otherwise blow the Python recursion stack.
        if qdepth >= self.QS_MAX_DEPTH and not in_check:
            return evaluate(b)
        if not in_check:
            stand = evaluate(b)
            if stand >= beta:
                return beta
            if stand > alpha:
                alpha = stand
            # Generate captures and queen promotions only.
            moves = []
            for m in b.legal_moves:
                if b.is_capture(m) or (m.promotion == chess.QUEEN):
                    moves.append(m)
            if not moves:
                return alpha
        else:
            moves = list(b.legal_moves)
            if not moves:
                # mated in qsearch
                return -MATE + (b.ply() - self._root_ply)

        self._order(moves, ply, None)
        for m in moves:
            # Skip clearly losing captures using a quick MVV/LVA prune.
            if not in_check and b.is_capture(m):
                victim = b.piece_at(m.to_square)
                attacker = b.piece_at(m.from_square)
                v = MG_VALUE[chess.PAWN] if victim is None else MG_VALUE[victim.piece_type]
                a = MG_VALUE[attacker.piece_type] if attacker else 0
                # Delta-pruning: if even taking this piece for free can't
                # raise eval past alpha, skip it (very rough, but fast).
                if v + 200 < alpha and a > v:
                    continue
            b.push(m)
            score = -self._qsearch(-beta, -alpha, ply + 1, qdepth + 1)
            b.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # -- non-pawn material check (for null-move legality) ------------------

    def _has_non_pawn(self, color: bool) -> bool:
        b = self.board
        occ = b.occupied_co[color]
        return bool(occ & ~(b.pawns | b.kings))

    # -- main alpha-beta ---------------------------------------------------

    def _pvs(self, depth: int, alpha: int, beta: int, ply: int, allow_null: bool) -> int:
        self.nodes += 1
        self._check_time()

        b = self.board

        # Draw detection. Avoid stepping into repetitions when we'd rather
        # play on — the contempt term penalises draws for the side to move
        # at the root and rewards them for the opponent.
        if ply > 0:
            if b.halfmove_clock >= 100 or b.is_insufficient_material():
                return self._draw_score()
            if b.halfmove_clock >= 4 and b.is_repetition(2):
                return self._draw_score()

        # Mate-distance pruning.
        mating = MATE - ply
        if mating < beta:
            beta = mating
            if alpha >= mating:
                return mating
        mated = -MATE + ply
        if mated > alpha:
            alpha = mated
            if beta <= mated:
                return mated

        if depth <= 0 or ply >= self.MAX_PLY:
            return self._qsearch(alpha, beta, ply)

        in_check = b.is_check()
        is_pv = (beta - alpha) > 1
        key = b._transposition_key()
        tt_move = None
        ent = self.tt.get(key)
        if ent is not None:
            tt_depth, tt_score, tt_flag, tt_move_e = ent
            if ply > 0 and tt_depth >= depth and not is_pv:
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == TT_UPPER and tt_score <= alpha:
                    return tt_score
            tt_move = tt_move_e

        # Internal iterative reduction: at high depth with no TT move,
        # shrink depth by one so this iteration can populate a TT move
        # for a better-ordered re-search at the next iterative-deepening
        # pass. Cheap, safe, worth a small amount of strength.
        if depth >= 5 and tt_move is None and not in_check:
            depth -= 1

        # Lazy static eval shared by reverse-futility, razoring, NMP, and
        # forward futility pruning below.
        static_eval = None

        # Forward pruning gates only fire at non-PV, non-check nodes.
        if not in_check and not is_pv:
            static_eval = evaluate(b)

            # Reverse-futility / static null-move pruning. At low depth a
            # large static surplus is unlikely to be reeled back, so we
            # can short-circuit.
            if depth <= 6 and static_eval - 90 * depth >= beta and abs(beta) < MATE_IN_MAX:
                return static_eval

            # Razoring at depth 1: if even adding a fat margin keeps us
            # below alpha, drop straight to quiescence.
            if depth <= 2 and static_eval + 150 * depth <= alpha:
                qscore = self._qsearch(alpha, beta, ply)
                if qscore <= alpha:
                    return qscore

            # Null-move pruning gated by static eval >= beta.
            if (
                allow_null
                and depth >= 3
                and self._has_non_pawn(b.turn)
                and static_eval >= beta
            ):
                R = 2 + depth // 4
                if static_eval - beta >= 200:
                    R += 1
                b.push(chess.Move.null())
                score = -self._pvs(depth - 1 - R, -beta, -beta + 1, ply + 1, allow_null=False)
                b.pop()
                if score >= beta:
                    if score >= MATE_IN_MAX:
                        score = beta
                    return score

        moves = list(b.legal_moves)
        if not moves:
            if in_check:
                return -MATE + ply
            return self._draw_score()
        self._order(moves, ply, tt_move)

        # Futility pruning eligibility (skip quiet moves at the frontier
        # when the static eval is far below alpha).
        futility = (
            not in_check
            and not is_pv
            and depth <= 3
            and (static_eval if static_eval is not None else evaluate(b)) + 100 * depth + 50 <= alpha
            and abs(alpha) < MATE_IN_MAX
        )

        best_score = -INF
        best_move = None
        flag = TT_UPPER
        searched = 0
        original_alpha = alpha

        for m in moves:
            is_capture = b.is_capture(m)
            is_promo = m.promotion is not None
            gives_check = b.gives_check(m)
            is_quiet = not is_capture and not is_promo and not gives_check

            # Late-move pruning: at low depth after enough quiet moves
            # already failed to raise alpha, skip the rest of the quiets.
            if (
                not in_check
                and not is_pv
                and depth <= 4
                and is_quiet
                and best_score > -MATE_IN_MAX
                and searched >= 3 + depth * depth
            ):
                continue

            # Frontier futility pruning on quiet moves.
            if futility and is_quiet and searched > 0:
                continue

            # One-ply check extension, but only when there is plenty of
            # head-room left in this branch. Otherwise long forcing lines
            # can chain extensions and keep depth from ever decreasing.
            extension = 1 if in_check and ply < 24 else 0
            new_depth = depth - 1 + extension

            b.push(m)
            try:
                if searched == 0:
                    score = -self._pvs(new_depth, -beta, -alpha, ply + 1, True)
                else:
                    # LMR: reduce late quiet moves.
                    reduce = 0
                    if (
                        depth >= 3
                        and searched >= 3
                        and is_quiet
                        and not in_check
                        and extension == 0
                    ):
                        # Logarithmic-ish formula: deeper or later -> reduce more.
                        reduce = 1
                        if searched >= 6:
                            reduce = 2
                        if depth >= 6 and searched >= 10:
                            reduce = 3
                    score = -self._pvs(new_depth - reduce, -alpha - 1, -alpha, ply + 1, True)
                    if score > alpha and reduce > 0:
                        score = -self._pvs(new_depth, -alpha - 1, -alpha, ply + 1, True)
                    if score > alpha and score < beta:
                        score = -self._pvs(new_depth, -beta, -alpha, ply + 1, True)
            finally:
                b.pop()
            searched += 1

            if score > best_score:
                best_score = score
                best_move = m
            if score > alpha:
                alpha = score
                flag = TT_EXACT
            if alpha >= beta:
                # Beta cutoff. Reward this move in killer / history tables.
                if not is_capture and not is_promo:
                    kl = self._killers_at(ply)
                    if kl[0] != m:
                        kl[1] = kl[0]
                        kl[0] = m
                    k = (m.from_square, m.to_square)
                    self.history[k] = self.history.get(k, 0) + depth * depth
                flag = TT_LOWER
                break

        if best_score <= original_alpha:
            flag = TT_UPPER

        # Replace-on-deeper TT policy.
        cur = self.tt.get(key)
        if cur is None or cur[0] <= depth:
            self.tt[key] = (depth, best_score, flag, best_move)

        return best_score

    # -- root --------------------------------------------------------------

    def search_root(self, max_depth: int = 64):
        b = self.board
        legal = list(b.legal_moves)
        if not legal:
            return None, 0

        best_move = legal[0]
        best_score = 0
        ordered = legal

        for depth in range(1, max_depth + 1):
            try:
                alpha = -INF
                beta = INF
                local_best = None
                local_best_score = -INF
                if best_move in ordered:
                    ordered = [best_move] + [m for m in ordered if m != best_move]
                searched = 0
                in_check_root = b.is_check()
                for m in ordered:
                    extension = 1 if in_check_root else 0
                    b.push(m)
                    try:
                        if searched == 0:
                            score = -self._pvs(depth - 1 + extension, -beta, -alpha, 1, True)
                        else:
                            score = -self._pvs(depth - 1 + extension, -alpha - 1, -alpha, 1, True)
                            if score > alpha:
                                score = -self._pvs(depth - 1 + extension, -beta, -alpha, 1, True)
                    finally:
                        b.pop()
                    searched += 1
                    if score > local_best_score:
                        local_best_score = score
                        local_best = m
                    if score > alpha:
                        alpha = score
                # Iteration completed fully.
                best_move = local_best
                best_score = local_best_score
                if best_score >= MATE_IN_MAX or best_score <= -MATE_IN_MAX:
                    break
            except _Stop:
                break

        return best_move, best_score


# ---------------------------------------------------------------------------
# Bot adapter
# ---------------------------------------------------------------------------

class Bot:
    name = "claude_v2_hard"

    # Reserve this much of the deadline for the rest of the bot machinery
    # (state copy, return path, queue handoff in the env's timer thread).
    SAFETY_MARGIN = 0.18
    DEFAULT_BUDGET = 1.8

    def choose_action(self, state) -> str:
        legal = state["legal_actions"]
        if not legal:
            raise ValueError("no legal actions")
        if len(legal) == 1:
            return legal[0]

        budget = state.get("decision_timeout")
        if budget is None:
            budget = self.DEFAULT_BUDGET
        budget = max(0.05, float(budget) - self.SAFETY_MARGIN)

        board = chess.Board(state["fen"])
        deadline = time.monotonic() + budget
        search = Search(board, deadline)
        move, _ = search.search_root(max_depth=64)
        if move is None:
            return legal[0]
        uci = move.uci()
        if uci in legal:
            return uci
        # If something subtle went wrong (e.g., promotion piece mismatch),
        # fall back to a same-square move from the legal list.
        for action in legal:
            if action[:4] == uci[:4]:
                return action
        return legal[0]
