#!/usr/bin/env python3
"""
DeepSeek v4 Hard Gomoku Bot — pure CPU, no neural networks.

Strategy:
  - Iterative-deepening principal-variation search (PVS) with TT + killer.
  - Full pattern-based eval (5-cell + 6-cell windows, open-end scoring).
  - Quiescence search on forcing moves (wins, fours, four-blocks).
  - Threat-based search extensions (four-creation → +1 depth).
  - Candidate pruning to radius-2 from stones; aggressive branch limits.
  - Time budget: 1.8 s within a 2.0 s decision_timeout.
"""

from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

# Ensure the gomoku package root is on sys.path so that `from env.gomoku_env import ...` works
# regardless of how the bot module was loaded (e.g. via importlib).
_ROOT = Path(__file__).resolve().parent.parent.parent  # BoardArena/gomoku/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.gomoku_env import (
    BLACK, WHITE, EMPTY, SYMBOLS, BOARD_SIZE, DIRECTIONS,
    board_from_rows, square_to_coords, coords_to_square,
    classify_move, on_board, opponent, count_fours_and_open_threes,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Eval tables
# ═══════════════════════════════════════════════════════════════════════════

_WIN       = 100_000_000
_OPEN_4    =   8_000_000
_HALF_4    =     500_000
_DEAD_4    =      40_000
_OPEN_3    =     300_000
_HALF_3    =      20_000
_DEAD_3    =       1_500
_OPEN_2    =       5_000
_HALF_2    =         300
_OPEN_1    =          40

_S5 = {
    (5, 2): _WIN,    (5, 1): _WIN,    (5, 0): _WIN,
    (4, 2): _OPEN_4, (4, 1): _HALF_4, (4, 0): _DEAD_4,
    (3, 2): _OPEN_3, (3, 1): _HALF_3, (3, 0): _DEAD_3,
    (2, 2): _OPEN_2, (2, 1): _HALF_2, (2, 0): 0,
    (1, 2): _OPEN_1, (1, 1): 0,       (1, 0): 0,
}

_OP3 = frozenset({".XXX..", "..XXX.", ".X.XX.", ".XX.X."})
_OP3S = _OPEN_3


# ═══════════════════════════════════════════════════════════════════════════
#  Full static evaluation from `player`'s perspective
# ═══════════════════════════════════════════════════════════════════════════

def _eval(board, player):
    m = SYMBOLS[player]
    o = SYMBOLS[opponent(player)]
    total = 0
    seen5 = set()
    seen6 = set()

    # Only iterate over occupied cells for efficiency
    for r in range(BOARD_SIZE):
        row = board[r]
        for c in range(BOARD_SIZE):
            if row[c] == EMPTY:
                continue
            for dr, dc in DIRECTIONS:
                # 5-cell windows through (r,c)
                for off in range(-4, 1):
                    wr, wc = r + off * dr, c + off * dc
                    er, ec = wr + 4 * dr, wc + 4 * dc
                    if not on_board(wr, wc) or not on_board(er, ec):
                        continue
                    k = (wr, wc, dr, dc)
                    if k in seen5:
                        continue
                    seen5.add(k)
                    mc = oc = 0
                    for i in range(5):
                        cell = board[wr + i * dr][wc + i * dc]
                        if cell == m:      mc += 1
                        elif cell == o:    oc += 1
                    if mc > 0 and oc == 0:
                        oe = 0
                        br, bc = wr - dr, wc - dc
                        if not on_board(br, bc) or board[br][bc] == EMPTY: oe += 1
                        ar, ac = er + dr, ec + dc
                        if not on_board(ar, ac) or board[ar][ac] == EMPTY: oe += 1
                        total += _S5.get((mc, oe), 0)
                    elif oc > 0 and mc == 0:
                        oe = 0
                        br, bc = wr - dr, wc - dc
                        if not on_board(br, bc) or board[br][bc] == EMPTY: oe += 1
                        ar, ac = er + dr, ec + dc
                        if not on_board(ar, ac) or board[ar][ac] == EMPTY: oe += 1
                        total -= _S5.get((oc, oe), 0)

                # 6-cell open-three windows through (r,c)
                for off in range(-5, 1):
                    wr, wc = r + off * dr, c + off * dc
                    er, ec = wr + 5 * dr, wc + 5 * dc
                    if not on_board(wr, wc) or not on_board(er, ec):
                        continue
                    k = (wr, wc, dr, dc)
                    if k in seen6:
                        continue
                    seen6.add(k)
                    w = []
                    for i in range(6):
                        cell = board[wr + i * dr][wc + i * dc]
                        w.append("X" if cell == m else ("O" if cell == o else "."))
                    w = "".join(w)
                    if w in _OP3:
                        total += _OP3S
                    ow = w.replace("X", "T").replace("O", "X").replace("T", "O")
                    if ow in _OP3:
                        total -= _OP3S

    return total


# ═══════════════════════════════════════════════════════════════════════════
#  Candidate moves (radius 2 from stones)
# ═══════════════════════════════════════════════════════════════════════════

def _cands(board):
    cells = set()
    for r in range(BOARD_SIZE):
        row = board[r]
        for c in range(BOARD_SIZE):
            if row[c] == EMPTY:
                continue
            for dr in range(-2, 3):
                nr = r + dr
                if nr < 0 or nr >= BOARD_SIZE:
                    continue
                for dc in range(-2, 3):
                    nc = c + dc
                    if 0 <= nc < BOARD_SIZE and board[nr][nc] == EMPTY:
                        cells.add((nr, nc))
    return cells


# ═══════════════════════════════════════════════════════════════════════════
#  TT
# ═══════════════════════════════════════════════════════════════════════════

_EXACT, _LOWER, _UPPER = 0, 1, 2
_NEG_INF = -300_000_000
_POS_INF =  300_000_000


class TT:
    def __init__(self, maxsize=300_000):
        self._d = OrderedDict()
        self._max = maxsize

    def store(self, key, depth, flag, value, best):
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = (depth, flag, value, best)
        if len(self._d) > self._max:
            self._d.popitem(last=False)

    def probe(self, key, depth):
        e = self._d.get(key)
        if e is None:
            return None
        d, flag, value, best = e
        if d >= depth:
            self._d.move_to_end(key)
            return (flag, value, best)
        return None

    def clear(self):
        self._d.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _upk(k, depth, move):
    if depth in k:
        k1, k2 = k[depth]
        if move != k1:
            k[depth] = (move, k1)
    else:
        k[depth] = (move, None)


def _prio(board, r, c, player, tbest, k1, k2):
    s = 0
    a = coords_to_square(r, c)
    if a == tbest: s += 10_000_000
    if a == k1:    s +=  5_000_000
    if a == k2:    s +=  4_000_000
    cx = BOARD_SIZE // 2
    s += max(0, 15 - abs(r - cx) - abs(c - cx))
    return s


def _thr(board, r, c, player):
    """(is_win, n_fours, n_threes)."""
    st = classify_move(board, r, c, player)
    if st in ("win", "white_win_overline"):
        return (True, 0, 0)
    if st.startswith("forbidden"):
        return (False, 0, 0)
    f, t = count_fours_and_open_threes(board, r, c, player)
    return (False, f, t)


# ═══════════════════════════════════════════════════════════════════════════
#  Quiescence
# ═══════════════════════════════════════════════════════════════════════════

def _qsearch(board, player, alpha, beta, deadline):
    """Quiescence: extend only on forcing moves."""
    if time.time() > deadline:
        raise TimeoutError()

    stand_pat = _eval(board, player)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    opp = opponent(player)
    cells = _cands(board)
    if not cells:
        return alpha

    forcing = []
    for r, c in cells:
        iw, n4, _ = _thr(board, r, c, player)
        if iw:
            forcing.append((30, r, c)); continue
        ow, o4, _ = _thr(board, r, c, opp)
        if ow:
            forcing.append((29, r, c)); continue
        if n4 >= 1:
            forcing.append((28, r, c)); continue
        if o4 >= 1:
            forcing.append((27, r, c))

    if not forcing:
        return alpha

    forcing.sort(key=lambda x: -x[0])
    forcing = forcing[:8]

    for _, r, c in forcing:
        if player == BLACK and classify_move(board, r, c, BLACK).startswith("forbidden"):
            continue
        board[r][c] = SYMBOLS[player]
        v = -_qsearch(board, opp, -beta, -alpha, deadline)
        board[r][c] = EMPTY
        if v >= beta:
            return beta
        if v > alpha:
            alpha = v
    return alpha


# ═══════════════════════════════════════════════════════════════════════════
#  Alpha-Beta PVS with threat extensions
# ═══════════════════════════════════════════════════════════════════════════

def _search(board, player, depth, alpha, beta, tt, killer, nodes, deadline):
    """Negamax PVS + quiescence at leaves."""
    if time.time() > deadline:
        raise TimeoutError()
    if nodes[0] <= 0:
        raise TimeoutError()
    nodes[0] -= 1

    key = tuple(tuple(r) for r in board)
    e = tt.probe(key, depth)
    tbest = None
    if e is not None:
        flag, tv, tb = e
        tbest = tb
        if flag == _EXACT:
            return tv, tb
        if flag == _LOWER and tv > alpha:
            alpha = tv
        elif flag == _UPPER and tv < beta:
            beta = tv
        if alpha >= beta:
            return tv, tb

    if depth <= 0:
        return _eval(board, player), None

    cells = list(_cands(board))
    if not cells:
        return _NEG_INF // 2, None

    opp = opponent(player)
    k1, k2 = killer.get(depth, (None, None))

    scored = []
    for r, c in cells:
        a = coords_to_square(r, c)
        s = _prio(board, r, c, player, tbest, k1, k2)
        iw, n4, n3 = _thr(board, r, c, player)
        if iw: s += 15_000_000
        ow, o4, o3 = _thr(board, r, c, opp)
        if ow: s += 14_000_000
        s += n4 * 1_000_000 + o4 * 800_000 + n3 * 12_000 + o3 * 10_000
        scored.append((s, r, c, a, iw, n4))

    scored.sort(key=lambda x: -x[0])
    BRANCH = 10 if depth >= 3 else 14
    scored = scored[:BRANCH]

    best_val = _NEG_INF
    best_move = None
    orig_alpha = alpha

    for i, (_, r, c, a, is_win, n4) in enumerate(scored):
        if player == BLACK and classify_move(board, r, c, BLACK).startswith("forbidden"):
            continue

        board[r][c] = SYMBOLS[player]

        if is_win:
            board[r][c] = EMPTY
            v = _POS_INF - 1
            tt.store(key, depth, _EXACT, v, a)
            _upk(killer, depth, a)
            return v, a

        ext = 1 if n4 >= 1 else 0

        if i == 0:
            ov, _ = _search(board, opp, depth - 1 + ext,
                            -beta, -alpha, tt, killer, nodes, deadline)
            v = -ov
        else:
            ov, _ = _search(board, opp, depth - 1 + ext,
                            -alpha - 1, -alpha, tt, killer, nodes, deadline)
            v = -ov
            if v > alpha and v < beta:
                ov, _ = _search(board, opp, depth - 1 + ext,
                                -beta, -alpha, tt, killer, nodes, deadline)
                v = -ov

        board[r][c] = EMPTY

        if v > best_val:
            best_val = v
            best_move = a
            if v > alpha:
                alpha = v
                if alpha >= beta:
                    _upk(killer, depth, a)
                    break

    if best_move is None:
        for _, r, c, a, _, _ in scored:
            if player == WHITE or not classify_move(board, r, c, BLACK).startswith("forbidden"):
                best_move = a
                break
        if best_move is None:
            return _NEG_INF // 2, None

    if best_val <= orig_alpha:
        tt.store(key, depth, _UPPER, best_val, best_move)
    elif best_val >= beta:
        tt.store(key, depth, _LOWER, best_val, best_move)
    else:
        tt.store(key, depth, _EXACT, best_val, best_move)

    return best_val, best_move


# ═══════════════════════════════════════════════════════════════════════════
#  Root
# ═══════════════════════════════════════════════════════════════════════════

def _root_order(board, legal, player):
    opp = opponent(player)
    out = []
    for a in legal:
        r, c = square_to_coords(a)
        s = 0
        iw, n4, n3 = _thr(board, r, c, player)
        if iw: s += 10_000_000
        ow, o4, o3 = _thr(board, r, c, opp)
        if ow: s += 9_000_000
        s += n4 * 1200 + o4 * 800 + n3 * 120 + o3 * 80
        cx = BOARD_SIZE // 2
        s += max(0, 20 - abs(r - cx) - abs(c - cx))
        out.append((s, r, c, a))
    out.sort(key=lambda x: -x[0])
    return out


def _fallback(board, player, legal):
    opp = opponent(player)
    ba, bs = legal[0], -999999
    for a in legal:
        r, c = square_to_coords(a)
        iw, n4, n3 = _thr(board, r, c, player)
        if iw: return a
        ow, o4, o3 = _thr(board, r, c, opp)
        if ow: ba = a; bs = 1000000; continue
        s = n4 * 1200 + o4 * 800 + n3 * 120 + o3 * 80
        cx = BOARD_SIZE // 2
        s += max(0, 20 - abs(r - cx) - abs(c - cx))
        if s > bs: bs = s; ba = a
    return ba


# ═══════════════════════════════════════════════════════════════════════════
#  Bot
# ═══════════════════════════════════════════════════════════════════════════

class Bot:
    """Iterative-deepening PVS Gomoku bot with quiescence."""

    name = "deepseek_v4_hard"

    def __init__(self):
        self._tt = TT()
        self._killer: dict = {}
        self._t0 = 0.0

    def choose_action(self, state):
        legal = state["legal_actions"]
        if not legal:
            raise RuntimeError("no legal actions")
        if len(legal) == 1:
            return legal[0]

        board = board_from_rows(state["board"])
        me = state["actor"]
        opp = opponent(me)
        plies = state.get("plies", 0)

        if plies == 0:
            return "h8"

        for a in legal:
            r, c = square_to_coords(a)
            if classify_move(board, r, c, me) in ("win", "white_win_overline"):
                return a

        opp_w = []
        for a in legal:
            r, c = square_to_coords(a)
            if classify_move(board, r, c, opp) in ("win", "white_win_overline"):
                opp_w.append(a)
        if len(opp_w) == 1:
            return opp_w[0]
        if len(opp_w) > 1:
            for a in legal:
                r, c = square_to_coords(a)
                f, _ = count_fours_and_open_threes(board, r, c, me)
                if f >= 1: return a
            return opp_w[0]

        if plies <= 2:
            cx = BOARD_SIZE // 2
            best, db = None, 999
            for a in legal:
                r, c = square_to_coords(a)
                d = abs(r - cx) + abs(c - cx)
                if d < db: db = d; best = a
            return best

        fb = _fallback(board, me, legal)
        self._t0 = time.time()
        DEADLINE = self._t0 + 1.80
        self._tt.clear()
        self._killer.clear()

        root = _root_order(board, legal, me)
        best = root[0][3] if root else fb

        nc = len(_cands(board))
        if nc <= 10:      md = 8
        elif nc <= 20:    md = 6
        elif nc <= 35:    md = 4
        else:             md = 2

        for d in range(2, md + 1, 2):
            if time.time() - self._t0 > 1.80 * 0.2:
                break
            try:
                a = self._sroot(board, me, d, root, DEADLINE)
                if a is not None:
                    best = a
            except TimeoutError:
                break

        if best is None:
            best = fb
        return best

    def _sroot(self, board, player, depth, root_scored, deadline):
        nodes = [999_999]
        best_val = _NEG_INF
        best_move = None
        n = min(len(root_scored), 15)

        for idx in range(n):
            if time.time() > deadline:
                raise TimeoutError()
            _, r, c, a = root_scored[idx]
            board[r][c] = SYMBOLS[player]
            st = classify_move(board, r, c, player)
            if st in ("win", "white_win_overline"):
                board[r][c] = EMPTY
                return a
            ov, _ = _search(board, opponent(player), depth - 1,
                            _NEG_INF, -best_val,
                            self._tt, self._killer, nodes, deadline)
            v = -ov
            board[r][c] = EMPTY
            if v > best_val:
                best_val = v
                best_move = a
        return best_move


_DEFAULT = None

def choose_action(state):
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Bot()
    return _DEFAULT.choose_action(state)
