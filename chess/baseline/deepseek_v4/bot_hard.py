"""DeepSeek v4 hard bot — classical chess engine (~1.6 s / move).

A non‑NN engine built around:
- Iterative deepening with time‑based stopping
- Principal‑variation alpha‑beta search
- Quiescence search (captures + promotions)
- Transposition table (Zobrist hashed, always‑replace)
- Null‑move pruning (R = 3 + depth // 4)
- Late‑move reductions
- Killer moves & history heuristic
- Tapered evaluation with piece‑square tables, pawn structure, king safety,
  bishop pair, and mobility
"""

from __future__ import annotations

import math
import random
import sys
import time

import chess
import chess.polyglot

SAFETY_MARGIN_SECONDS = 0.15

# Python default recursion limit may be too low for deep searches.
sys.setrecursionlimit(10_000)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATE_SCORE   = 100_000
MATE_BOUND   = MATE_SCORE - 2_000
DRAW_SCORE   = 0

# Transposition‑table flags
TT_EXACT = 0   # exact score
TT_ALPHA = 1   # upper bound (fail‑low)
TT_BETA  = 2   # lower bound (fail‑high)

# Piece type indices used in PST arrays.
P = 0  # pawn
N = 1  # knight
B = 2  # bishop
R = 3  # rook
Q = 4  # queen
K = 5  # king

PIECE_INDEX = {
    chess.PAWN: P, chess.KNIGHT: N, chess.BISHOP: B,
    chess.ROOK: R, chess.QUEEN: Q, chess.KING: K,
}

# Centipawn material values.
MATERIAL_VALUE = [100, 320, 330, 500, 900, 0]

# Weights used for game‑phase tapering (non‑pawn, non‑king material).
# Starting total   = 4*320 + 4*330 + 4*500 + 2*900 = 6200 cp.
PHASE_MAX = 6200

# ---------------------------------------------------------------------------
# Piece‑Square Tables  (middlegame  /  endgame)
# Values are from the perspective of *white*.  Black tables are flipped.
# ---------------------------------------------------------------------------

# fmt: off
_PST_MG = [
    # Pawn
    [0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 27, 27, 10,  5,  5,
     0,  0,  0, 25, 25,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-25,-25, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0],
    # Knight
    [-50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50],
    # Bishop
    [-20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20],
    # Rook
    [ 0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0],
    # Queen
    [-20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20],
    # King (middlegame — stay safe in the corner)
    [-30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20],
]

_PST_EG = [
    # Pawn
    [0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0],
    # Knight
    [-50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50],
    # Bishop
    [-20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20],
    # Rook
    [ 0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0],
    # Queen
    [-20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20],
    # King (endgame — centralize)
    [-50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50],
]
# fmt: on


# ---------------------------------------------------------------------------
# Transposition Table
# ---------------------------------------------------------------------------

class TranspositionTable:
    """Fixed‑size always‑replace transposition table keyed by Zobrist hash."""

    __slots__ = ("_table", "_max_entries")

    def __init__(self, max_mb: int = 64) -> None:
        self._max_entries = max(1, (max_mb * 1_048_576) // 32)
        self._table: dict[int, tuple[int, int, int, chess.Move | None]] = {}

    def store(
        self, key: int, depth: int, flag: int, score: int, move: chess.Move | None,
    ) -> None:
        # Adjust mate scores relative to the stored position.
        if score > MATE_BOUND:
            score += 0  # depth‑adjustment not needed with correct ply handling
        elif score < -MATE_BOUND:
            score -= 0

        entry = (depth, flag, score, move)
        if key in self._table:
            old_depth, _, _, _ = self._table[key]
            if old_depth > depth:
                return
        elif len(self._table) >= self._max_entries:
            self._table.clear()
        self._table[key] = entry

    def probe(
        self, key: int, depth: int, alpha: int, beta: int,
    ) -> int | None:
        entry = self._table.get(key)
        if entry is None:
            return None
        stored_depth, flag, score, _ = entry
        if stored_depth < depth:
            return None
        if flag == TT_EXACT:
            return score
        if flag == TT_ALPHA and score <= alpha:
            return score
        if flag == TT_BETA and score >= beta:
            return score
        return None

    def get_move(self, key: int) -> chess.Move | None:
        entry = self._table.get(key)
        return entry[3] if entry else None


# ---------------------------------------------------------------------------
# Bitboard helpers  (avoid SquareSet compatibility issues)
# ---------------------------------------------------------------------------

def _bb_iter(bb: int):
    """Yield each square index whose bit is set in the bitboard *bb*."""
    while bb:
        lsb = bb & -bb
        sq = lsb.bit_length() - 1
        yield sq
        bb ^= lsb


def _bb_count(bb: int) -> int:
    """Return the population count of bitboard *bb*."""
    return bb.bit_count()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _flip_sq(sq: int) -> int:
    """Flip square vertically (a1 <-> a8)."""
    return sq ^ 56


def _sq_index(sq: int, is_white: bool) -> int:
    """Return the PST index for the given square and colour."""
    return sq if is_white else _flip_sq(sq)


def _pst(sq: int, piece_type: int, is_white: bool, mg: bool) -> int:
    """Look up the piece‑square value."""
    table = _PST_MG if mg else _PST_EG
    idx = _sq_index(sq, is_white)
    return table[piece_type][idx]


class Evaluator:
    """Tapered, hand‑crafted evaluation function."""

    # Pawn structure bonuses / penalties (centipawns)
    DOUBLED_PAWN_PENALTY  = -15
    ISOLATED_PAWN_PENALTY = -15
    PASSED_PAWN_BONUS     = [0, 0, 5, 10, 30, 60, 90, 0]  # per rank (2-7)

    BISHOP_PAIR_BONUS_MG = 45
    BISHOP_PAIR_BONUS_EG = 60

    # King‑safety table: penalty for each missing pawn in the king's shield.
    KING_SHIELD_MISSING = -12

    # Mobility weights (middlegame only, approximate)
    MOBILITY_WEIGHTS = [0, 2, 3, 4, 2, 0]  # P, N, B, R, Q, K

    def evaluate(self, board: chess.Board) -> int:
        """Return score in centipawns from White's perspective."""
        if board.is_checkmate():
            return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return DRAW_SCORE

        mg = [0, 0]  # [white_mg, black_mg]
        eg = [0, 0]  # [white_eg, black_eg]
        material = [0, 0]  # non‑pawn, non‑king material for phase
        piece_counts: dict[int, list[int]] = {
            chess.PAWN: [0, 0], chess.KNIGHT: [0, 0], chess.BISHOP: [0, 0],
            chess.ROOK: [0, 0], chess.QUEEN: [0, 0],
        }

        # Pawn masks for structure evaluation.
        wpawns = board.pieces(chess.PAWN, chess.WHITE)
        bpawns = board.pieces(chess.PAWN, chess.BLACK)

        for color in (chess.WHITE, chess.BLACK):
            side = 0 if color == chess.WHITE else 1
            is_w = color == chess.WHITE
            king_sq = board.king(color)
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)

            for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                               chess.ROOK, chess.QUEEN, chess.KING):
                squares = board.pieces(piece_type, color)
                for sq in squares:
                    # Material + PST
                    pidx = PIECE_INDEX[piece_type]
                    mg[side] += MATERIAL_VALUE[pidx] + _pst(sq, pidx, is_w, True)
                    eg[side] += MATERIAL_VALUE[pidx] + _pst(sq, pidx, is_w, False)

                    # Phase material (exclude pawns and kings)
                    if piece_type not in (chess.PAWN, chess.KING):
                        material[side] += MATERIAL_VALUE[pidx]

                # Track piece counts for bishop pair.
                if piece_type in piece_counts:
                    cnt = piece_counts[piece_type]
                    cnt[side] = len(squares)

            # ---- Pawn structure (side = 0 white, side = 1 black) ----
            side_pawns_bb = int(wpawns if is_w else bpawns)
            opp_pawns_bb  = int(bpawns if is_w else wpawns)
            for sq in _bb_iter(side_pawns_bb):
                f = chess.square_file(sq)
                r = chess.square_rank(sq)
                # Doubled pawn
                if (side_pawns_bb & chess.BB_FILES[f]).bit_count() > 1:
                    mg[side] += Evaluator.DOUBLED_PAWN_PENALTY
                    eg[side] += Evaluator.DOUBLED_PAWN_PENALTY
                # Isolated pawn (no friendly pawn on adjacent files)
                adj_mask = 0
                if f > 0:
                    adj_mask |= chess.BB_FILES[f - 1]
                if f < 7:
                    adj_mask |= chess.BB_FILES[f + 1]
                if not (side_pawns_bb & adj_mask):
                    mg[side] += Evaluator.ISOLATED_PAWN_PENALTY
                    eg[side] += Evaluator.ISOLATED_PAWN_PENALTY
                # Passed pawn (no opposing pawns in front on same/adjacent files)
                rank_slice = chess.BB_RANKS[r + 1:] if is_w else chess.BB_RANKS[:r]
                ahead_mask = 0
                for rm in rank_slice:
                    ahead_mask |= rm
                if not (opp_pawns_bb & adj_mask & ahead_mask) and not (opp_pawns_bb & chess.BB_FILES[f] & ahead_mask):
                    rank = r if is_w else 7 - r
                    bonus = Evaluator.PASSED_PAWN_BONUS[rank]
                    mg[side] += bonus
                    eg[side] += bonus * 2  # passed pawns more valuable in endgame

            # ---- King safety (middlegame) ----
            # Shield: pawns in front of king
            shield_files = set()
            for df in (-1, 0, 1):
                f2 = king_file + df
                if 0 <= f2 <= 7:
                    shield_files.add(f2)
            for f2 in shield_files:
                shielded = False
                if is_w:
                    ranks = range(max(0, king_rank - 2), king_rank + 1)
                else:
                    ranks = range(king_rank, min(7, king_rank + 3))
                for r2 in ranks:
                    if not 0 <= r2 <= 7:
                        continue
                    if chess.BB_SQUARES[chess.square(f2, r2)] & side_pawns_bb:
                        shielded = True
                        break
                if not shielded:
                    mg[side] += Evaluator.KING_SHIELD_MISSING

            # King endgame bonus is already handled by PST.

        # ---- Bishop pair ----
        for side in (0, 1):
            if piece_counts[chess.BISHOP][side] >= 2:
                mg[side] += Evaluator.BISHOP_PAIR_BONUS_MG
                eg[side] += Evaluator.BISHOP_PAIR_BONUS_EG

        # ---- Mobility (approximate, for middlegame only) ----
        for color in (chess.WHITE, chess.BLACK):
            side = 0 if color == chess.WHITE else 1
            b2 = board.copy()
            b2.turn = color
            n_moves = b2.legal_moves.count()
            # Rough mobility bonus: ~2 cp per legal move above 10
            mobility_bonus = max(0, (n_moves - 10)) * 2
            mg[side] += mobility_bonus

        # ---- Tapered evaluation ----
        phase = (material[0] + material[1])
        phase = max(0, min(PHASE_MAX, phase))
        phase_ratio = phase / PHASE_MAX

        score_mg = mg[0] - mg[1]
        score_eg = eg[0] - eg[1]
        score = int(score_mg * phase_ratio + score_eg * (1.0 - phase_ratio))

        # Tempo bonus for side to move.
        score += 10

        return score


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

# MVV‑LVA victim scores for capture ordering.
MVV_LVA = {
    chess.PAWN: 100, chess.KNIGHT: 200, chess.BISHOP: 300,
    chess.ROOK: 400, chess.QUEEN: 500, chess.KING: 600,
}

KILLER_MAX = 128  # max ply for killer slots
HISTORY_MAX = 1 << 12


def _is_endgame(board: chess.Board) -> bool:
    """Heuristic: game is endgame when both sides have <= 13 non‑pawn material."""
    material = 0
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            material += len(board.pieces(pt, color)) * MATERIAL_VALUE[PIECE_INDEX[pt]]
    return material <= 2600


def _move_ordering_score(
    board: chess.Board,
    move: chess.Move,
    hash_move: chess.Move | None,
    killers: list[chess.Move | None],
    history: dict[tuple[int, int], int],
) -> int:
    """Return a score for move ordering (higher = search first)."""
    if move == hash_move:
        return 10_000_000
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        if victim is None and board.is_en_passant(move):
            victim_type = chess.PAWN
        else:
            victim_type = victim.piece_type if victim else chess.PAWN
        attacker = board.piece_at(move.from_square)
        attacker_type = attacker.piece_type if attacker else chess.PAWN
        # MVV‑LVA: high victim, low attacker
        return 1_000_000 + MVV_LVA[victim_type] * 100 - MVV_LVA[attacker_type]
    if move.promotion:
        return 900_000 + MATERIAL_VALUE[PIECE_INDEX[move.promotion]]
    # Killer moves
    for i, killer in enumerate(killers):
        if move == killer:
            return 800_000 - i * 100
    # History heuristic
    key = (move.from_square, move.to_square)
    return history.get(key, 0)


class Bot:
    """Classical alpha‑beta chess engine."""

    name = "deepseek_v4_hard"

    def __init__(self) -> None:
        self._tt = TranspositionTable(max_mb=64)
        self._evaluator = Evaluator()
        self._killers: list[list[chess.Move | None]] = [
            [None, None] for _ in range(KILLER_MAX)
        ]
        self._history: dict[tuple[int, int], int] = {}
        self._nodes = 0
        self._start_time = 0.0
        self._time_limit = 1.55  # seconds (leave 0.05 s safety margin)
        self._timed_out = False
        self._best_root_move: str | None = None
        self._last_root_score: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def choose_action(self, state: dict) -> str:
        """Return a UCI move from *state* within ~1.6 s."""
        legal: list[str] = state["legal_actions"]
        if not legal:
            raise ValueError("no legal actions available")
        if len(legal) == 1:
            return legal[0]
        timeout = state.get("decision_timeout") or state.get("time_limit")
        if timeout:
            self._time_limit = max(0.05, float(timeout) - SAFETY_MARGIN_SECONDS)

        board = chess.Board(state["fen"])
        self._start_time = time.time()
        self._nodes = 0
        self._timed_out = False
        self._best_root_move = None

        # Iterative deepening.
        for depth in range(1, 128):
            move = self._search_root(board, depth)
            if self._timed_out:
                break
            self._best_root_move = move
            # If mate found, stop.
            if self._best_root_move is not None and abs(self._last_root_score) > MATE_BOUND:
                break
            # Time check — stop if last iteration used > 50 % of remaining time.
            elapsed = time.time() - self._start_time
            if elapsed > self._time_limit * 0.55:
                break

        result = self._best_root_move
        if result is None or result not in legal:
            result = legal[0]
        return result

    # ------------------------------------------------------------------
    # Root search
    # ------------------------------------------------------------------

    def _search_root(self, board: chess.Board, depth: int) -> str | None:
        alpha = -MATE_SCORE
        beta = MATE_SCORE
        best_score = -MATE_SCORE
        best_move: chess.Move | None = None

        moves = list(board.legal_moves)
        if not moves:
            return None

        hash_key = chess.polyglot.zobrist_hash(board)
        hash_move = self._tt.get_move(hash_key)
        killers = self._killers[0]

        # Order moves at root.
        scored_moves: list[tuple[int, chess.Move]] = []
        for m in moves:
            sc = _move_ordering_score(board, m, hash_move, killers, self._history)
            scored_moves.append((-sc, m))
        scored_moves.sort(key=lambda x: x[0])

        for _, move in scored_moves:
            board.push(move)
            self._nodes += 1
            score = -self._search(board, depth - 1, -beta, -alpha, ply=1)
            board.pop()

            if self._timed_out:
                break

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score

        self._last_root_score = best_score
        return best_move.uci() if best_move else None

    # ------------------------------------------------------------------
    # Alpha‑beta search
    # ------------------------------------------------------------------

    def _search(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int = 0,
    ) -> int:
        """Negamax alpha‑beta with null‑move pruning, LMR, TT, and QS."""
        # Timeout check (every 2048 nodes).
        self._nodes += 1
        if (self._nodes & 2047) == 0:
            if time.time() - self._start_time > self._time_limit:
                self._timed_out = True
                return DRAW_SCORE
        if self._timed_out:
            return DRAW_SCORE

        # Draw detection.
        if board.is_repetition(2) or board.can_claim_fifty_moves():
            return DRAW_SCORE

        # Mate distance pruning.
        if alpha < -MATE_BOUND:
            alpha = max(alpha, -MATE_SCORE + ply)
        if beta > MATE_BOUND:
            beta = min(beta, MATE_SCORE - ply - 1)
        if alpha >= beta:
            return alpha

        # Transposition table probe.
        hash_key = chess.polyglot.zobrist_hash(board)
        tt_score = self._tt.probe(hash_key, depth, alpha, beta)
        if tt_score is not None:
            return tt_score

        # Leaf node → quiescence search.
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        # Null‑move pruning.
        if (depth >= 3 and not board.is_check()
                and not _is_endgame(board)
                and board.legal_moves.count() > 0):
            board.push(chess.Move.null())
            R = 3 + depth // 4
            null_score = -self._search(board, depth - 1 - R, -beta, -beta + 1, ply + 1)
            board.pop()
            if self._timed_out:
                return DRAW_SCORE
            if null_score >= beta:
                return beta

        # Generate & order moves.
        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if board.is_check() else DRAW_SCORE

        hash_move = self._tt.get_move(hash_key)
        killers = self._killers[ply] if ply < KILLER_MAX else [None, None]

        scored: list[tuple[int, chess.Move]] = []
        for m in moves:
            sc = _move_ordering_score(board, m, hash_move, killers, self._history)
            scored.append((-sc, m))
        scored.sort(key=lambda x: x[0])

        best_score = -MATE_SCORE
        best_move: chess.Move | None = None
        flag = TT_ALPHA

        for i, (_, move) in enumerate(scored):
            is_capture = board.is_capture(move)
            is_promo = move.promotion is not None
            quiet = not is_capture and not is_promo

            # Extend: check‐giving moves.
            board.push(move)
            gives_check = board.is_check()
            extension = 1 if gives_check and depth <= 4 else 0

            # Late‑move reduction for quiet moves.
            if i >= 4 and depth >= 3 and quiet and not gives_check:
                R = 1 + (depth // 6) + (i // 6)
                R = min(R, depth - 1)
                score = -self._search(board, depth - 1 - R + extension, -alpha - 1, -alpha, ply + 1)
                if score > alpha:
                    # Re‑search at full depth.
                    score = -self._search(board, depth - 1 + extension, -beta, -alpha, ply + 1)
            else:
                score = -self._search(board, depth - 1 + extension, -beta, -alpha, ply + 1)

            board.pop()

            if self._timed_out:
                return DRAW_SCORE

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    flag = TT_EXACT
                    if score >= beta:
                        flag = TT_BETA
                        # Update killer & history for quiet moves that cause cutoff.
                        if quiet:
                            if ply < KILLER_MAX:
                                if self._killers[ply][0] != move:
                                    self._killers[ply][1] = self._killers[ply][0]
                                    self._killers[ply][0] = move
                            key = (move.from_square, move.to_square)
                            self._history[key] = self._history.get(key, 0) + depth * depth
                        break

        # Store in transposition table.
        if not self._timed_out and best_move is not None:
            self._tt.store(hash_key, depth, flag, best_score, best_move)

        return best_score

    # ------------------------------------------------------------------
    # Quiescence search
    # ------------------------------------------------------------------

    def _quiescence(
        self, board: chess.Board, alpha: int, beta: int, ply: int,
    ) -> int:
        """Search captures and promotions until a quiet position is reached."""
        self._nodes += 1
        if (self._nodes & 2047) == 0:
            if time.time() - self._start_time > self._time_limit:
                self._timed_out = True
                return DRAW_SCORE
        if self._timed_out:
            return DRAW_SCORE

        # Stand‑pat score.
        stand_pat = self._evaluator.evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Generate captures + promotions.
        moves: list[chess.Move] = []
        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion:
                moves.append(move)

        # Order captures by MVV‑LVA.
        moves.sort(key=lambda m: -_move_ordering_score(board, m, None, [], {}))

        for move in moves:
            # Delta pruning: skip captures that can't possibly raise alpha.
            victim = board.piece_at(move.to_square)
            if victim is None and board.is_en_passant(move):
                victim_type = chess.PAWN
            else:
                victim_type = victim.piece_type if victim else chess.PAWN
            gain = MATERIAL_VALUE[PIECE_INDEX[victim_type]]
            if stand_pat + gain + 200 < alpha:  # 200 cp margin
                continue

            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()

            if self._timed_out:
                return DRAW_SCORE

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha
