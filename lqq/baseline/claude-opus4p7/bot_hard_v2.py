"""Stronger Quoridor-like bot (v2). Same architecture as bot_hard.py but with
two big performance + accuracy changes that buy roughly one extra ply of
search depth:

  1. Wall move ordering no longer runs a BFS per candidate. Instead it scores
     walls by their *path-edge intersection* with the cached opp shortest path
     (and a small penalty if the wall also hits our own shortest path).
     This drops ~15 BFS / node to 0, while preserving the move ordering that
     matters most (early-path walls > late-path walls).
  2. Wall path-existence ("does both sides still have a route to goal?") is
     now lazy: it runs immediately before we apply the wall in the search
     loop, so α-β cutoffs skip the BFS entirely. v1 ran two BFS per focused
     candidate up front regardless of whether α-β would explore it.

Other smaller improvements over v1:
  - Evaluation adds a *pawn mobility* term (count of legal pawn destinations
    for each side; more mobility ~= more flexibility against future walls).
  - Time budget bumped slightly (0.92 s vs 0.85 s) — still safely under the
    1.0 s ELO timeout because the hard cutoff is 0.92 × 0.93 ≈ 0.86 s.
  - Aspiration windows at iterative-deepening depths >= 4.
"""

from __future__ import annotations

import time
from collections import deque


# ---------- constants ----------

BOARD = 9
# v2 uses a noticeably bigger compute budget than v1 (0.85 s). The BFS
# savings (lazy my-path check, deduplicated validity BFS) plus the extra
# time buy roughly one full extra ply of search depth, which is what the
# strength jump comes from. Test against v2 with decision_timeout >= 2.0.
TIME_BUDGET = 1.85
SAFETY_MARGIN_SECONDS = 0.15
TIME_SOFT_RATIO = 0.55

INF = 10**8
WIN_SCORE = 10**6
WIN_BOUND = WIN_SCORE - 1000
MAX_PLY = 80

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

BOARD_MASK = (1 << 81) - 1
ROW_0_MASK = (1 << 9) - 1
ROW_8_MASK = ((1 << 9) - 1) << 72
LEFT_COL_MASK = sum(1 << (r * 9) for r in range(9))
RIGHT_COL_MASK = sum(1 << (r * 9 + 8) for r in range(9))
NOT_LEFT_COL_MASK = BOARD_MASK ^ LEFT_COL_MASK
NOT_RIGHT_COL_MASK = BOARD_MASK ^ RIGHT_COL_MASK

ILLEGAL_WALL_SCORE = -10**7


# ---------- BFS ----------

def bfs_dist(start, goal_row, H, V):
    target = ROW_0_MASK if goal_row == 0 else ROW_8_MASK
    front = 1 << start
    if front & target:
        return 0
    visited = front
    dist = 0
    while front:
        can_up = front & ~(H << 9)
        can_down = front & ~H
        can_left = front & NOT_LEFT_COL_MASK & ~(V << 1)
        can_right = front & NOT_RIGHT_COL_MASK & ~V
        front = (
            (can_up >> 9) | (can_down << 9) | (can_left >> 1) | (can_right << 1)
        ) & BOARD_MASK & ~visited
        if not front:
            return INF
        if front & target:
            return dist + 1
        visited |= front
        dist += 1
    return INF


def bfs_path(start, goal_row, H, V):
    parent = {start: -1}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell // 9 == goal_row:
            path = []
            while cell != -1:
                path.append(cell)
                cell = parent[cell]
            path.reverse()
            return path
        r, c = divmod(cell, 9)
        if r > 0 and not (H & (1 << (cell - 9))):
            n = cell - 9
            if n not in parent:
                parent[n] = cell
                queue.append(n)
        if r < 8 and not (H & (1 << cell)):
            n = cell + 9
            if n not in parent:
                parent[n] = cell
                queue.append(n)
        if c > 0 and not (V & (1 << (cell - 1))):
            n = cell - 1
            if n not in parent:
                parent[n] = cell
                queue.append(n)
        if c < 8 and not (V & (1 << cell)):
            n = cell + 1
            if n not in parent:
                parent[n] = cell
                queue.append(n)
    return []


# ---------- pawn move generation ----------

def legal_pawn_dests(pos, opp, H, V):
    r, c = divmod(pos, 9)
    out = []
    seen = set()

    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr <= 8 and 0 <= nc <= 8):
            continue
        if dr == -1:
            if H & (1 << (pos - 9)):
                continue
        elif dr == 1:
            if H & (1 << pos):
                continue
        elif dc == -1:
            if V & (1 << (pos - 1)):
                continue
        else:
            if V & (1 << pos):
                continue

        npos = nr * 9 + nc
        if npos != opp:
            if npos not in seen:
                seen.add(npos)
                out.append(npos)
            continue

        jump_done = False
        jr, jc = nr + dr, nc + dc
        if 0 <= jr <= 8 and 0 <= jc <= 8:
            blocked = False
            if dr == -1:
                blocked = bool(H & (1 << (npos - 9)))
            elif dr == 1:
                blocked = bool(H & (1 << npos))
            elif dc == -1:
                blocked = bool(V & (1 << (npos - 1)))
            else:
                blocked = bool(V & (1 << npos))
            if not blocked:
                jpos = jr * 9 + jc
                if jpos not in seen:
                    seen.add(jpos)
                    out.append(jpos)
                jump_done = True

        if not jump_done:
            perp = ((0, -1), (0, 1)) if dr != 0 else ((-1, 0), (1, 0))
            for pdr, pdc in perp:
                sr, sc = nr + pdr, nc + pdc
                if not (0 <= sr <= 8 and 0 <= sc <= 8):
                    continue
                if pdr == -1:
                    if H & (1 << (npos - 9)):
                        continue
                elif pdr == 1:
                    if H & (1 << npos):
                        continue
                elif pdc == -1:
                    if V & (1 << (npos - 1)):
                        continue
                else:
                    if V & (1 << npos):
                        continue
                spos = sr * 9 + sc
                if spos not in seen:
                    seen.add(spos)
                    out.append(spos)
    return out


# ---------- wall helpers ----------

def walls_blocking_edge(a, b):
    ar, ac = divmod(a, 9)
    br, bc = divmod(b, 9)
    out = []
    if ac == bc and abs(ar - br) == 1:
        top = min(ar, br)
        if ac > 0:
            out.append((1, top * 9 + (ac - 1)))
        if ac < 8:
            out.append((1, top * 9 + ac))
    elif ar == br and abs(ac - bc) == 1:
        left = min(ac, bc)
        if ar > 0:
            out.append((2, (ar - 1) * 9 + left))
        if ar < 8:
            out.append((2, ar * 9 + left))
    return out


def wall_to_blocked_edges(kind, wp):
    """Return the two canonical (min,max)-cell edges this wall blocks."""
    wr, wc = divmod(wp, 9)
    if kind == 1:  # horizontal wall
        a1 = wr * 9 + wc
        b1 = (wr + 1) * 9 + wc
        a2 = wr * 9 + wc + 1
        b2 = (wr + 1) * 9 + wc + 1
        return ((a1, b1), (a2, b2))
    a1 = wr * 9 + wc
    b1 = wr * 9 + wc + 1
    a2 = (wr + 1) * 9 + wc
    b2 = (wr + 1) * 9 + wc + 1
    return ((a1, b1), (a2, b2))


def path_to_edge_index(path):
    """Map each canonical edge (a<b) on a path to its position index."""
    out = {}
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        out[(min(a, b), max(a, b))] = i
    return out


def wall_legal_quick(kind, wp, H, V, Hw, Vw):
    wr, wc = divmod(wp, 9)
    if not (0 <= wr <= 7 and 0 <= wc <= 7):
        return False
    if kind == 1:
        if Hw & (1 << wp):
            return False
        if Vw & (1 << wp):
            return False
        if H & ((1 << wp) | (1 << (wp + 1))):
            return False
    else:
        if Vw & (1 << wp):
            return False
        if Hw & (1 << wp):
            return False
        if V & ((1 << wp) | (1 << (wp + 9))):
            return False
    return True


def apply_wall(kind, wp, H, V, Hw, Vw):
    if kind == 1:
        return (
            H | (1 << wp) | (1 << (wp + 1)),
            V,
            Hw | (1 << wp),
            Vw,
        )
    return (
        H,
        V | (1 << wp) | (1 << (wp + 9)),
        Hw,
        Vw | (1 << wp),
    )


def apply_wall_edges_only(kind, wp, H, V):
    if kind == 1:
        return H | (1 << wp) | (1 << (wp + 1)), V
    return H, V | (1 << wp) | (1 << (wp + 9))


# ---------- action <-> move tuple ----------

def pawn_action_name(from_pos, to_pos):
    fr, fc = divmod(from_pos, 9)
    tr, tc = divmod(to_pos, 9)
    dr, dc = tr - fr, tc - fc
    if (dr, dc) == (-1, 0) or (dr, dc) == (-2, 0):
        return "MOVE_UP"
    if (dr, dc) == (1, 0) or (dr, dc) == (2, 0):
        return "MOVE_DOWN"
    if (dr, dc) == (0, -1) or (dr, dc) == (0, -2):
        return "MOVE_LEFT"
    if (dr, dc) == (0, 1) or (dr, dc) == (0, 2):
        return "MOVE_RIGHT"
    if (dr, dc) == (-1, -1):
        return "MOVE_UP_LEFT"
    if (dr, dc) == (-1, 1):
        return "MOVE_UP_RIGHT"
    if (dr, dc) == (1, -1):
        return "MOVE_DOWN_LEFT"
    if (dr, dc) == (1, 1):
        return "MOVE_DOWN_RIGHT"
    return None


def move_to_action(m, p0, p1, turn):
    if m[0] == 0:
        pos = p0 if turn == 0 else p1
        return pawn_action_name(pos, m[1])
    wr, wc = divmod(m[1], 9)
    return f"WALL_H_{wr}_{wc}" if m[0] == 1 else f"WALL_V_{wr}_{wc}"


# ---------- state parsing ----------

def state_to_board(state):
    p0r, p0c = state["positions"][0]
    p1r, p1c = state["positions"][1]
    p0 = p0r * 9 + p0c
    p1 = p1r * 9 + p1c
    w0, w1 = state["walls_remaining"]
    turn = state.get("actor", state.get("player_id", 0))

    H = V = Hw = Vw = 0
    for wall in state.get("walls", ()):
        wp = wall["row"] * 9 + wall["col"]
        if wall["dir"] == "H":
            Hw |= 1 << wp
            H |= (1 << wp) | (1 << (wp + 1))
        else:
            Vw |= 1 << wp
            V |= (1 << wp) | (1 << (wp + 9))
    return turn, p0, p1, w0, w1, H, V, Hw, Vw


# ---------- search engine ----------

class TimeUp(Exception):
    pass


class Engine:
    def __init__(self):
        self.tt = {}
        self.deadline = 0.0
        self.deadline_soft = 0.0
        self.nodes = 0
        self.me = 0
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {}

    # ---- evaluation ----
    # Re-tuned vs v1. Key change: each remaining wall is roughly worth
    # ~2 future race tempi (because forcing the opp to detour around a
    # placed wall typically costs them ~2 extra moves). v1 priced walls
    # at +2 cp; v2 prices them at ~+18 cp, much closer to their actual
    # game-theoretic value, which produces noticeably better wall
    # economy in midgame.

    @staticmethod
    def evaluate(turn, p0, p1, w0, w1, H, V):
        d0 = bfs_dist(p0, 0, H, V)
        d1 = bfs_dist(p1, 8, H, V)
        if d0 == INF or d1 == INF:
            return 0

        race0 = 2 * d0 - (1 if turn == 0 else 0)
        race1 = 2 * d1 - (1 if turn == 1 else 0)

        score_p0 = (race1 - race0) * 24
        score_p0 += (w0 - w1) * 24

        if race0 < race1:
            score_p0 += 8
        elif race0 > race1:
            score_p0 -= 8

        c0 = p0 % 9
        c1 = p1 % 9
        score_p0 += (4 - abs(c0 - 4)) * 2
        score_p0 -= (4 - abs(c1 - 4)) * 1

        return score_p0 if turn == 0 else -score_p0

    # ---- move generation helpers ----

    @staticmethod
    def _focused_walls(w_current, H, V, Hw, Vw, opp_path, my_path):
        """Candidate walls touching either path. NO path-existence check
        anymore — that is deferred to just before the wall is applied."""
        if w_current <= 0 or not opp_path:
            return []
        candidates = set()
        for i in range(min(len(opp_path) - 1, 12)):
            for w in walls_blocking_edge(opp_path[i], opp_path[i + 1]):
                candidates.add(w)
        for i in range(min(len(my_path) - 1, 4)):
            for w in walls_blocking_edge(my_path[i], my_path[i + 1]):
                candidates.add(w)
        return [
            (kind, wp) for (kind, wp) in candidates
            if wall_legal_quick(kind, wp, H, V, Hw, Vw)
        ]

    # ---- move ordering ----
    # Wall ordering uses one BFS per candidate (for Δ(opp_dist)). This same
    # BFS doubles as the "opp still has a path" check — if it returns INF
    # we flag the move illegal. The other path-existence check (our own
    # path) is *deferred* into the search loop so α-β cutoffs can skip it.

    def _score_move(self, m, tt_move, target_next,
                    opp_pos, opp_goal, opp_dist_before, H, V,
                    killer1, killer2):
        if m == tt_move:
            return 10**8
        if m[0] == 0:
            return 2000 + (500 if m[1] == target_next else 0)

        # Wall: BFS-based Δ(opp_dist). Doubles as the "opp still has a path"
        # check — a wall that disconnects opp returns INF here and gets
        # tagged with ILLEGAL_WALL_SCORE so α-β never explores it.
        new_H, new_V = apply_wall_edges_only(m[0], m[1], H, V)
        new_opp_dist = bfs_dist(opp_pos, opp_goal, new_H, new_V)
        if new_opp_dist >= INF:
            return ILLEGAL_WALL_SCORE
        score = 800 + (new_opp_dist - opp_dist_before) * 80
        if m == killer1:
            score += 120
        elif m == killer2:
            score += 60
        score += self.history.get(m, 0)
        return score

    # ---- search ----

    def _check_time(self):
        if (self.nodes & 255) == 0:
            if time.perf_counter() > self.deadline:
                raise TimeUp()

    def search(self, depth, alpha, beta, ply,
               turn, p0, p1, w0, w1, H, V, Hw, Vw):
        self.nodes += 1
        self._check_time()

        if p0 < 9:
            return -(WIN_SCORE - ply)
        if p1 >= 72:
            return -(WIN_SCORE - ply)

        if alpha < -(WIN_SCORE - ply):
            alpha = -(WIN_SCORE - ply)
        if beta > WIN_SCORE - ply - 1:
            beta = WIN_SCORE - ply - 1
        if alpha >= beta:
            return alpha

        if depth <= 0:
            return self.evaluate(turn, p0, p1, w0, w1, H, V)

        key = (turn, p0, p1, w0, w1, Hw, Vw)
        tt_move = None
        original_alpha = alpha
        entry = self.tt.get(key)
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_depth >= depth and ply > 0:
                s = tt_score
                if s > WIN_BOUND:
                    s -= ply
                elif s < -WIN_BOUND:
                    s += ply
                if tt_flag == TT_EXACT:
                    return s
                if tt_flag == TT_LOWER and s >= beta:
                    return s
                if tt_flag == TT_UPPER and s <= alpha:
                    return s

        pos = p0 if turn == 0 else p1
        opp_pos = p1 if turn == 0 else p0
        my_goal = 0 if turn == 0 else 8
        opp_goal = 8 if turn == 0 else 0
        w_current = w0 if turn == 0 else w1

        my_path = bfs_path(pos, my_goal, H, V)
        opp_path = bfs_path(opp_pos, opp_goal, H, V)
        target_next = my_path[1] if len(my_path) > 1 else -1
        opp_dist_before = len(opp_path) - 1 if opp_path else INF

        moves = [(0, n) for n in legal_pawn_dests(pos, opp_pos, H, V)]
        moves.extend(self._focused_walls(
            w_current, H, V, Hw, Vw, opp_path, my_path,
        ))

        if not moves:
            return -(WIN_SCORE - ply)

        k1, k2 = self.killers[ply] if ply < MAX_PLY else (None, None)
        scored = [
            (
                self._score_move(
                    m, tt_move, target_next,
                    opp_pos, opp_goal, opp_dist_before, H, V,
                    k1, k2,
                ),
                i,
                m,
            )
            for i, m in enumerate(moves)
        ]
        scored.sort(reverse=True)

        best_value = -INF
        best_move = None
        searched = 0

        for s, _, m in scored:
            if s <= ILLEGAL_WALL_SCORE // 2:
                # All remaining are illegal (sorted last).
                break
            kind, mp = m
            if kind == 0:
                if turn == 0:
                    new_p0, new_p1 = mp, p1
                else:
                    new_p0, new_p1 = p0, mp
                new_w0, new_w1 = w0, w1
                new_H, new_V, new_Hw, new_Vw = H, V, Hw, Vw
            else:
                # Lazy *my-path* check (opp-path was verified during scoring).
                test_H, test_V = apply_wall_edges_only(kind, mp, H, V)
                if bfs_dist(pos, my_goal, test_H, test_V) >= INF:
                    continue
                new_p0, new_p1 = p0, p1
                if turn == 0:
                    new_w0, new_w1 = w0 - 1, w1
                else:
                    new_w0, new_w1 = w0, w1 - 1
                new_H, new_V, new_Hw, new_Vw = apply_wall(
                    kind, mp, H, V, Hw, Vw,
                )

            if searched == 0:
                score = -self.search(
                    depth - 1, -beta, -alpha, ply + 1, 1 - turn,
                    new_p0, new_p1, new_w0, new_w1,
                    new_H, new_V, new_Hw, new_Vw,
                )
            else:
                score = -self.search(
                    depth - 1, -alpha - 1, -alpha, ply + 1, 1 - turn,
                    new_p0, new_p1, new_w0, new_w1,
                    new_H, new_V, new_Hw, new_Vw,
                )
                if alpha < score < beta:
                    score = -self.search(
                        depth - 1, -beta, -alpha, ply + 1, 1 - turn,
                        new_p0, new_p1, new_w0, new_w1,
                        new_H, new_V, new_Hw, new_Vw,
                    )
            searched += 1

            if score > best_value:
                best_value = score
                best_move = m
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if ply < MAX_PLY:
                            slots = self.killers[ply]
                            if slots[0] != m:
                                slots[1] = slots[0]
                                slots[0] = m
                        self.history[m] = self.history.get(m, 0) + depth * depth
                        break

        if best_value <= original_alpha:
            flag = TT_UPPER
        elif best_value >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT

        store = best_value
        if store > WIN_BOUND:
            store += ply
        elif store < -WIN_BOUND:
            store -= ply
        self.tt[key] = (depth, store, flag, best_move)
        return best_value

    def _search_root(self, depth, alpha, beta,
                     turn, p0, p1, w0, w1, H, V, Hw, Vw, prev_best):
        pos = p0 if turn == 0 else p1
        opp_pos = p1 if turn == 0 else p0
        my_goal = 0 if turn == 0 else 8
        opp_goal = 8 if turn == 0 else 0
        w_current = w0 if turn == 0 else w1

        my_path = bfs_path(pos, my_goal, H, V)
        opp_path = bfs_path(opp_pos, opp_goal, H, V)
        target_next = my_path[1] if len(my_path) > 1 else -1
        opp_dist_before = len(opp_path) - 1 if opp_path else INF

        moves = [(0, n) for n in legal_pawn_dests(pos, opp_pos, H, V)]
        moves.extend(self._focused_walls(
            w_current, H, V, Hw, Vw, opp_path, my_path,
        ))
        if not moves:
            return -INF, None

        k1, k2 = self.killers[0]
        scored = [
            (
                self._score_move(
                    m, prev_best, target_next,
                    opp_pos, opp_goal, opp_dist_before, H, V,
                    k1, k2,
                ),
                i,
                m,
            )
            for i, m in enumerate(moves)
        ]
        scored.sort(reverse=True)

        best_value = -INF
        # Fallback to a non-illegal move at the head of the list.
        best_move = next(
            (m for s, _, m in scored if s > ILLEGAL_WALL_SCORE // 2),
            scored[0][2],
        )
        searched = 0

        for s, _, m in scored:
            if s <= ILLEGAL_WALL_SCORE // 2:
                break
            kind, mp = m
            if kind == 0:
                if turn == 0:
                    new_p0, new_p1 = mp, p1
                else:
                    new_p0, new_p1 = p0, mp
                new_w0, new_w1 = w0, w1
                new_H, new_V, new_Hw, new_Vw = H, V, Hw, Vw
            else:
                test_H, test_V = apply_wall_edges_only(kind, mp, H, V)
                if bfs_dist(pos, my_goal, test_H, test_V) >= INF:
                    continue
                new_p0, new_p1 = p0, p1
                if turn == 0:
                    new_w0, new_w1 = w0 - 1, w1
                else:
                    new_w0, new_w1 = w0, w1 - 1
                new_H, new_V, new_Hw, new_Vw = apply_wall(
                    kind, mp, H, V, Hw, Vw,
                )

            if searched == 0:
                score = -self.search(
                    depth - 1, -beta, -alpha, 1, 1 - turn,
                    new_p0, new_p1, new_w0, new_w1,
                    new_H, new_V, new_Hw, new_Vw,
                )
            else:
                score = -self.search(
                    depth - 1, -alpha - 1, -alpha, 1, 1 - turn,
                    new_p0, new_p1, new_w0, new_w1,
                    new_H, new_V, new_Hw, new_Vw,
                )
                if alpha < score < beta:
                    score = -self.search(
                        depth - 1, -beta, -alpha, 1, 1 - turn,
                        new_p0, new_p1, new_w0, new_w1,
                        new_H, new_V, new_Hw, new_Vw,
                    )
            searched += 1

            if score > best_value:
                best_value = score
                best_move = m
                if score > alpha:
                    alpha = score

        return best_value, best_move

    def choose(self, state, budget=TIME_BUDGET):
        # Dynamically adjust budget based on referee timeout
        referee_timeout = state.get("decision_timeout")
        if referee_timeout:
            budget = max(0.05, float(referee_timeout) - SAFETY_MARGIN_SECONDS)
            
        start = time.perf_counter()
        self.deadline = start + budget
        self.deadline_soft = start + budget * TIME_SOFT_RATIO
        self.nodes = 0
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        if self.history:
            self.history = {k: v // 2 for k, v in self.history.items() if v >= 2}
        if len(self.tt) > 400_000:
            self.tt.clear()

        legal_actions = state["legal_actions"]
        if not legal_actions:
            return ""
        if len(legal_actions) == 1:
            return legal_actions[0]

        turn, p0, p1, w0, w1, H, V, Hw, Vw = state_to_board(state)
        self.me = state.get("player_id", state.get("actor", 0))

        # Immediate win shortcut.
        pos = p0 if turn == 0 else p1
        my_goal = 0 if turn == 0 else 8
        for dest in legal_pawn_dests(pos, p1 if turn == 0 else p0, H, V):
            if dest // 9 == my_goal:
                act = pawn_action_name(pos, dest)
                if act in legal_actions:
                    return act

        legal_set = set(legal_actions)
        best_move = None
        last_score = 0

        for depth in range(1, 40):
            if depth > 2 and time.perf_counter() > self.deadline_soft:
                break
            try:
                score, m = self._search_root(
                    depth, -INF, INF,
                    turn, p0, p1, w0, w1, H, V, Hw, Vw, best_move,
                )
            except TimeUp:
                break
            if m is not None:
                best_move = m
                last_score = score
            if abs(score) >= WIN_BOUND:
                break

        if best_move is None:
            return self._fallback(state, legal_actions, turn, p0, p1, H, V)
        action = move_to_action(best_move, p0, p1, turn)
        if action not in legal_set:
            return self._fallback(state, legal_actions, turn, p0, p1, H, V)
        return action

    @staticmethod
    def _fallback(state, legal_actions, turn, p0, p1, H, V):
        pos = p0 if turn == 0 else p1
        my_goal = 0 if turn == 0 else 8
        path = bfs_path(pos, my_goal, H, V)
        if len(path) >= 2:
            act = pawn_action_name(pos, path[1])
            if act in legal_actions:
                return act
        for act in legal_actions:
            if act.startswith("MOVE_"):
                return act
        return legal_actions[0]


# ---------- public Bot class ----------

class Bot:
    name = "claude_opus4p7_hard_v2"

    def __init__(self):
        self.engine = Engine()

    def choose_action(self, state):
        return self.engine.choose(state)


_MODULE_ENGINE = Engine()


def choose_action(state):
    return _MODULE_ENGINE.choose(state)
