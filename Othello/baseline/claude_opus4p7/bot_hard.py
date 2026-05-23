"""Strong Othello bot. Bitboard + iterative-deepening alpha-beta + PVS + TT.

Time budget: ~1.7 s per move (the env's convention is a 2 s decision timeout).

Components:
  - 64-bit bitboard representation (one mask per side); square index follows
    a1 = bit 0 ... h8 = bit 63.
  - Move generation via the standard shift-cascade in 8 directions
    (5 propagation iterations cover any line length on an 8x8 board).
  - Iterative deepening α-β with principal variation search.
  - Transposition table keyed on (me, opp) with depth-preferred replacement
    and mate-distance separation (huge multiplier on disc differential keeps
    solved leaves clearly above midgame eval values).
  - Move ordering: TT move > static positional ordering (corner first,
    edges next, C/X-squares last).
  - Evaluation: PST + mobility + frontier-disc count + phase-tapered
    disc-count term (negative early, positive in the endgame).
  - When both sides have no moves the search returns the exact disc
    differential, so deep iterations naturally produce perfect endgame
    solutions whenever the search depth reaches the empty-square count.
"""

from __future__ import annotations

import time


# ---------- bitboard constants ----------

BB_MASK = 0xFFFFFFFFFFFFFFFF
NOT_A = 0xFEFEFEFEFEFEFEFE
NOT_H = 0x7F7F7F7F7F7F7F7F

CORNERS = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)

# Static positional ordering scores (a1 = index 0 ... h8 = 63).
# Corners are great, X-squares (b2/g2/b7/g7) and C-squares
# (a2/b1/h2/g1/a7/b8/h7/g8) are terrible until the corner is yours.
ORDER_SCORE = (
    100, -50,  10,   2,   2,  10, -50, 100,
    -50, -80,  -4,  -4,  -4,  -4, -80, -50,
     10,  -4,   0,   0,   0,   0,  -4,  10,
      2,  -4,   0,   0,   0,   0,  -4,   2,
      2,  -4,   0,   0,   0,   0,  -4,   2,
     10,  -4,   0,   0,   0,   0,  -4,  10,
    -50, -80,  -4,  -4,  -4,  -4, -80, -50,
    100, -50,  10,   2,   2,  10, -50, 100,
)

# Evaluation PST. Same shape, sharper values used for static eval.
PST = (
    100, -20,  10,   5,   5,  10, -20, 100,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
      5,  -2,  -1,  -1,  -1,  -1,  -2,   5,
     10,  -2,  -1,  -1,  -1,  -1,  -2,  10,
    -20, -50,  -2,  -2,  -2,  -2, -50, -20,
    100, -20,  10,   5,   5,  10, -20, 100,
)

DISC_VALUE = 1_000_000       # multiplier on exact endgame disc diff
SOLVED_THRESHOLD = 500_000   # any |score| above this signals a solved leaf

INF = 10**12
MAX_DEPTH = 64

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

TIME_BUDGET = 1.7
SAFETY_MARGIN_SECONDS = 0.15
TIME_SOFT_RATIO = 0.55


# ---------- bitboard primitives ----------

def shift_n(b):  return (b << 8) & BB_MASK
def shift_s(b):  return b >> 8
def shift_e(b):  return (b << 1) & NOT_A
def shift_w(b):  return (b >> 1) & NOT_H
def shift_ne(b): return (b << 9) & NOT_A
def shift_nw(b): return (b << 7) & NOT_H
def shift_se(b): return (b >> 7) & NOT_A
def shift_sw(b): return (b >> 9) & NOT_H


_SHIFTS = (shift_n, shift_s, shift_e, shift_w,
           shift_ne, shift_nw, shift_se, shift_sw)


def popcount(b):
    return bin(b).count("1")


def generate_moves(me, opp):
    """Bitboard of all empty squares where `me` can legally play."""
    empty = ~(me | opp) & BB_MASK
    moves = 0
    for sf in _SHIFTS:
        flank = sf(me) & opp
        flank |= sf(flank) & opp
        flank |= sf(flank) & opp
        flank |= sf(flank) & opp
        flank |= sf(flank) & opp
        flank |= sf(flank) & opp
        moves |= sf(flank) & empty
    return moves


def compute_flips(move_bit, me, opp):
    """Bitboard of opponent discs flipped if `me` plays at move_bit."""
    flips = 0
    for sf in _SHIFTS:
        line = 0
        cur = sf(move_bit)
        while cur and (cur & opp):
            line |= cur
            cur = sf(cur)
        if cur & me:
            flips |= line
    return flips


def frontier_mask(empty):
    """Squares adjacent (orthogonally + diagonally) to any empty square."""
    f = 0
    for sf in _SHIFTS:
        f |= sf(empty)
    return f


# ---------- square <-> name ----------

def sq_to_name(sq):
    return f"{chr(ord('a') + (sq & 7))}{(sq >> 3) + 1}"


def name_to_sq(name):
    return ((int(name[1]) - 1) << 3) | (ord(name[0]) - ord("a"))


# ---------- state parsing ----------

def parse_state(state):
    """Return (me, opp) bitboards from the side-to-move's perspective."""
    actor = state.get("actor", state.get("player_id", 0))
    rows = state["board"]  # rows[0] is rank 8
    black = white = 0
    for i, row in enumerate(rows):
        r = 7 - i
        for c, ch in enumerate(row):
            if ch == "B":
                black |= 1 << (r * 8 + c)
            elif ch == "W":
                white |= 1 << (r * 8 + c)
    if actor == 0:
        return black, white
    return white, black


# ---------- search engine ----------

class TimeUp(Exception):
    pass


class Engine:
    def __init__(self):
        self.tt = {}
        self.deadline = 0.0
        self.deadline_soft = 0.0
        self.start = 0.0
        self.nodes = 0
        self.last_root_score = 0

    # ---- evaluation ----

    @staticmethod
    def evaluate(me, opp):
        """Static eval from side-to-move (`me`) perspective. Bounded
        well below SOLVED_THRESHOLD so solved leaves dominate."""
        empty = ~(me | opp) & BB_MASK
        empty_count = popcount(empty)

        # PST
        pst = 0
        bb = me
        while bb:
            lsb = bb & -bb
            pst += PST[lsb.bit_length() - 1]
            bb ^= lsb
        bb = opp
        while bb:
            lsb = bb & -bb
            pst -= PST[lsb.bit_length() - 1]
            bb ^= lsb

        # Mobility
        me_mob_bb = generate_moves(me, opp)
        opp_mob_bb = generate_moves(opp, me)
        me_mob = popcount(me_mob_bb)
        opp_mob = popcount(opp_mob_bb)
        mob = me_mob - opp_mob

        # Frontier discs (lower is better for the owner)
        fm = frontier_mask(empty)
        me_front = popcount(me & fm)
        opp_front = popcount(opp & fm)
        front = opp_front - me_front

        # Disc count (phase-tapered: negative early, positive late)
        me_count = popcount(me)
        opp_count = popcount(opp)
        disc = me_count - opp_count

        if empty_count > 40:           # opening
            return pst * 3 + mob * 30 + front * 8 + (-disc) * 1
        if empty_count > 20:           # midgame
            return pst * 2 + mob * 22 + front * 12 + disc * 1
        if empty_count > 8:            # late midgame
            return pst * 1 + mob * 14 + front * 8 + disc * 4
        # Sub-endgame buffer (only reached when search didn't go deep enough)
        return pst * 1 + mob * 8 + front * 4 + disc * 18

    # ---- time control ----

    def _check_time(self):
        if (self.nodes & 1023) == 0:
            if time.perf_counter() > self.deadline:
                raise TimeUp()

    # ---- main search ----

    def _search(self, me, opp, depth, alpha, beta, ply, pass_count):
        self.nodes += 1
        self._check_time()

        moves_bb = generate_moves(me, opp)
        if moves_bb == 0:
            if pass_count >= 1:
                # Both sides have just passed → game over, exact result.
                return (popcount(me) - popcount(opp)) * DISC_VALUE
            # Forced pass.
            return -self._search(opp, me, depth - 1, -beta, -alpha,
                                 ply + 1, pass_count + 1)

        if depth <= 0:
            return self.evaluate(me, opp)

        # ---- TT probe ----
        key = (me, opp)
        tt_move = -1
        original_alpha = alpha
        entry = self.tt.get(key)
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_depth >= depth and ply > 0:
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == TT_UPPER and tt_score <= alpha:
                    return tt_score

        # ---- generate + order moves ----
        moves = []
        bb = moves_bb
        while bb:
            lsb = bb & -bb
            sq = lsb.bit_length() - 1
            bb ^= lsb
            score = ORDER_SCORE[sq]
            if sq == tt_move:
                score += 100_000
            moves.append((-score, sq))  # sort ascending = best first
        moves.sort()

        best_score = -INF
        best_move = -1
        searched = 0

        for _, sq in moves:
            move_bit = 1 << sq
            flips = compute_flips(move_bit, me, opp)
            if flips == 0:
                # Shouldn't happen — generate_moves only returns squares
                # whose play actually flanks at least one disc.
                continue
            new_me = me | move_bit | flips
            new_opp = opp ^ flips

            if searched == 0:
                score = -self._search(new_opp, new_me, depth - 1,
                                      -beta, -alpha, ply + 1, 0)
            else:
                score = -self._search(new_opp, new_me, depth - 1,
                                      -alpha - 1, -alpha, ply + 1, 0)
                if alpha < score < beta:
                    score = -self._search(new_opp, new_me, depth - 1,
                                          -beta, -alpha, ply + 1, 0)
            searched += 1

            if score > best_score:
                best_score = score
                best_move = sq
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break

        if best_score <= original_alpha:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        self.tt[key] = (depth, best_score, flag, best_move)
        return best_score

    def _search_root(self, me, opp, depth, prev_best):
        moves_bb = generate_moves(me, opp)
        if moves_bb == 0:
            return 0, -1

        moves = []
        bb = moves_bb
        while bb:
            lsb = bb & -bb
            sq = lsb.bit_length() - 1
            bb ^= lsb
            score = ORDER_SCORE[sq]
            if sq == prev_best:
                score += 200_000
            moves.append((-score, sq))
        moves.sort()

        alpha = -INF
        beta = INF
        best_score = -INF
        best_move = moves[0][1]
        searched = 0

        for _, sq in moves:
            move_bit = 1 << sq
            flips = compute_flips(move_bit, me, opp)
            new_me = me | move_bit | flips
            new_opp = opp ^ flips

            if searched == 0:
                score = -self._search(new_opp, new_me, depth - 1,
                                      -beta, -alpha, 1, 0)
            else:
                score = -self._search(new_opp, new_me, depth - 1,
                                      -alpha - 1, -alpha, 1, 0)
                if alpha < score < beta:
                    score = -self._search(new_opp, new_me, depth - 1,
                                          -beta, -alpha, 1, 0)
            searched += 1

            if score > best_score:
                best_score = score
                best_move = sq
                if score > alpha:
                    alpha = score

        return best_score, best_move

    # ---- public choose ----

    def choose(self, state, budget=TIME_BUDGET):
        legal = state["legal_actions"]
        if not legal:
            return ""
        if legal == ["PASS"]:
            return "PASS"
        if len(legal) == 1:
            return legal[0]

        budget = _time_budget(state, budget)
        self.start = time.perf_counter()
        self.deadline = self.start + budget
        self.deadline_soft = self.start + budget * TIME_SOFT_RATIO
        self.nodes = 0
        if len(self.tt) > 400_000:
            self.tt.clear()

        me, opp = parse_state(state)
        empty_count = popcount(~(me | opp) & BB_MASK)

        # Iterative deepening. When empty_count is small the search will
        # naturally reach game-over leaves with exact disc differentials.
        max_depth = empty_count if empty_count <= 16 else MAX_DEPTH

        best_move = -1
        best_score = 0
        for depth in range(1, max_depth + 1):
            if depth > 2 and time.perf_counter() > self.deadline_soft:
                # In a clear endgame we still let depth grow even past soft;
                # the hard deadline still bounds us.
                if empty_count > 14:
                    break
            try:
                score, sq = self._search_root(me, opp, depth, best_move)
            except TimeUp:
                break
            if sq >= 0:
                best_move = sq
                best_score = score
            # Solved (exact endgame result reached).
            if abs(best_score) > SOLVED_THRESHOLD:
                break

        legal_set = set(legal)
        if best_move >= 0:
            name = sq_to_name(best_move)
            if name in legal_set:
                self.last_root_score = best_score
                return name

        # Fallbacks: prefer corners, then any non-PASS legal action.
        for a in legal:
            if a in ("a1", "a8", "h1", "h8"):
                return a
        for a in legal:
            if a != "PASS":
                return a
        return legal[0]


# ---------- public Bot class ----------

class Bot:
    name = "claude_opus4p7_hard"

    def __init__(self):
        self.engine = Engine()

    def choose_action(self, state):
        return self.engine.choose(state)


_MODULE_ENGINE = Engine()


def choose_action(state):
    return _MODULE_ENGINE.choose(state)


def _time_budget(state, fallback):
    timeout = state.get("decision_timeout") or state.get("time_limit")
    if timeout:
        return max(0.05, float(timeout) - SAFETY_MARGIN_SECONDS)
    return fallback
