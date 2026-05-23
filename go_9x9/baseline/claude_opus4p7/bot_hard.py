"""Strong CPU-only 9x9 Go bot — MCTS with RAVE, focused branching, and
capture-priority rollouts.

Time budget: ~1.75 s per move (under the 2.0 s decision timeout convention).

Design choices that matter for strength inside pure-Python:
  1. **Focused branching**. We never consider more than ~22 candidate moves
     at the root or ~12 at an interior node. Candidates are ranked by a
     cheap heuristic: distance-to-centre, neighbours of any stone, last-move
     proximity. Spreading visits across all 81 cells starved the search
     of signal in our budget.
  2. **Capture-priority rollouts**. We scan all opponent groups once at the
     start of each rollout move and, if any are in atari, play the capture
     point with probability 0.85. Without this the rollouts give
     fundamentally noisy evaluations on small boards (you can't read tactics
     in random walks).
  3. **MCTS-UCT-RAVE blend** with the exploration term applied outside the
     RAVE/UCT convex combination. The textbook "blend then UCT-inside"
     formula causes a child with no RAVE data (eg PASS) to drown out
     genuinely better but newly-visited children.
  4. **Tree reuse**. We try to match the saved root against the new
     position one or two plies down; if the match succeeds we keep the
     subtree.
  5. **Eye-aware** (Bouzy 1-bad-diagonal interior / 0-bad-diagonal edge).
     Even random rollouts must avoid filling true eyes.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any


# ----------------------------- constants --------------------------------

BOARD_SIZE = 9
N = BOARD_SIZE * BOARD_SIZE  # 81
EMPTY = 0
BLACK = 1
WHITE = 2
PASS_MOVE = -1

FILES = "abcdefghi"
KOMI = 6.5

TIME_BUDGET = 1.75
SAFETY_MARGIN_SECONDS = 0.15
MAX_PLAYOUT_MOVES = 220
RAVE_EQUIV = 1500.0
UCB_C = 0.95
CAPTURE_PRIOR_PROB = 0.85
ROOT_BRANCH = 22
INNER_BRANCH = 12
PROGRESSIVE_BASE = 6  # initial branching at any node; grows with sqrt(visits)


# --------------------- precomputed neighbour tables --------------------


def _neighbours(p: int) -> tuple[int, ...]:
    r, c = divmod(p, BOARD_SIZE)
    out = []
    if r > 0:
        out.append(p - BOARD_SIZE)
    if r + 1 < BOARD_SIZE:
        out.append(p + BOARD_SIZE)
    if c > 0:
        out.append(p - 1)
    if c + 1 < BOARD_SIZE:
        out.append(p + 1)
    return tuple(out)


def _diagonals(p: int) -> tuple[int, ...]:
    r, c = divmod(p, BOARD_SIZE)
    out = []
    if r > 0 and c > 0:
        out.append(p - BOARD_SIZE - 1)
    if r > 0 and c + 1 < BOARD_SIZE:
        out.append(p - BOARD_SIZE + 1)
    if r + 1 < BOARD_SIZE and c > 0:
        out.append(p + BOARD_SIZE - 1)
    if r + 1 < BOARD_SIZE and c + 1 < BOARD_SIZE:
        out.append(p + BOARD_SIZE + 1)
    return tuple(out)


NEIGHBOURS: tuple[tuple[int, ...], ...] = tuple(_neighbours(p) for p in range(N))
DIAGONALS: tuple[tuple[int, ...], ...] = tuple(_diagonals(p) for p in range(N))


def point_to_action(p: int) -> str:
    if p == PASS_MOVE:
        return "PASS"
    r, c = divmod(p, BOARD_SIZE)
    return f"{FILES[c]}{r + 1}"


def action_to_point(action: str) -> int:
    if action == "PASS":
        return PASS_MOVE
    file = ord(action[0]) - ord("a")
    rank = int(action[1:])
    return (rank - 1) * BOARD_SIZE + file


# --------------------------------- Board --------------------------------


class Board:
    """Compact board. Group queries via on-demand BFS."""

    __slots__ = ("stones", "ko", "to_move", "last_move")

    def __init__(self) -> None:
        self.stones = bytearray(N)
        self.ko = -1
        self.to_move = BLACK
        self.last_move = -1

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.stones = bytearray(self.stones)
        b.ko = self.ko
        b.to_move = self.to_move
        b.last_move = self.last_move
        return b

    def from_rows(self, rows: list[str]) -> None:
        """Load from the env's state rows (rank 9 first)."""
        for display_index, row in enumerate(rows):
            internal_row = BOARD_SIZE - 1 - display_index
            for col, ch in enumerate(row):
                p = internal_row * BOARD_SIZE + col
                if ch == "B":
                    self.stones[p] = BLACK
                elif ch == "W":
                    self.stones[p] = WHITE
                else:
                    self.stones[p] = EMPTY

    # ---- group BFS ----

    def _group_and_libs(self, p: int) -> tuple[list[int], set[int]]:
        color = self.stones[p]
        group = [p]
        seen = {p}
        libs: set[int] = set()
        i = 0
        while i < len(group):
            q = group[i]
            i += 1
            for nq in NEIGHBOURS[q]:
                v = self.stones[nq]
                if v == EMPTY:
                    libs.add(nq)
                elif v == color and nq not in seen:
                    seen.add(nq)
                    group.append(nq)
        return group, libs

    def play_move(self, p: int, color: int) -> list[int]:
        """Place `color` at `p`. Assumes legal. Returns captured stones."""
        opp = WHITE if color == BLACK else BLACK
        self.stones[p] = color
        captured: list[int] = []
        for nq in NEIGHBOURS[p]:
            if self.stones[nq] == opp:
                grp, libs = self._group_and_libs(nq)
                if not libs:
                    for s in grp:
                        self.stones[s] = EMPTY
                        captured.append(s)
        # Ko detection.
        if len(captured) == 1:
            _, own_libs = self._group_and_libs(p)
            lone = all(self.stones[nq] != color for nq in NEIGHBOURS[p])
            if lone and len(own_libs) == 1 and next(iter(own_libs)) == captured[0]:
                self.ko = captured[0]
            else:
                self.ko = -1
        else:
            self.ko = -1
        self.to_move = opp
        self.last_move = p
        return captured

    def is_legal(self, p: int, color: int) -> bool:
        if p == PASS_MOVE:
            return True
        if self.stones[p] != EMPTY:
            return False
        if p == self.ko:
            return False
        # Fast path: any empty neighbour ⇒ legal.
        opp = WHITE if color == BLACK else BLACK
        for nq in NEIGHBOURS[p]:
            if self.stones[nq] == EMPTY:
                return True
        # Slow path: connect to friendly with extra liberty, or capture opp.
        for nq in NEIGHBOURS[p]:
            v = self.stones[nq]
            if v == color:
                _, libs = self._group_and_libs(nq)
                if len(libs) >= 2:
                    return True
            elif v == opp:
                _, libs = self._group_and_libs(nq)
                if len(libs) == 1:
                    return True
        return False

    def is_true_eye(self, p: int, color: int) -> bool:
        if self.stones[p] != EMPTY:
            return False
        for nq in NEIGHBOURS[p]:
            if self.stones[nq] != color:
                return False
        opp = WHITE if color == BLACK else BLACK
        bad = 0
        diags = DIAGONALS[p]
        for d in diags:
            if self.stones[d] == opp:
                bad += 1
        if len(diags) == 4:
            return bad <= 1
        return bad == 0

    def find_atari_capture(self, color: int) -> int:
        """Return any single-liberty opponent group's capture point, else -1.

        Uses a per-stone visited mask so each opponent group is BFS'd once.
        """
        opp = WHITE if color == BLACK else BLACK
        stones = self.stones
        seen = bytearray(N)
        for start in range(N):
            if stones[start] != opp or seen[start]:
                continue
            grp = [start]
            seen[start] = 1
            libs: list[int] = []
            i = 0
            while i < len(grp):
                q = grp[i]
                i += 1
                for nq in NEIGHBOURS[q]:
                    v = stones[nq]
                    if v == EMPTY:
                        if nq not in libs:
                            libs.append(nq)
                    elif v == opp and not seen[nq]:
                        seen[nq] = 1
                        grp.append(nq)
            if len(libs) == 1 and libs[0] != self.ko:
                return libs[0]
        return -1

    # ---- scoring ----

    def score(self, komi: float) -> tuple[float, float]:
        counts = [0, 0, 0]
        for v in self.stones:
            counts[v] += 1
        territory = [0, 0]
        visited = bytearray(N)
        for p in range(N):
            if visited[p] or self.stones[p] != EMPTY:
                continue
            region = [p]
            visited[p] = 1
            borders: set[int] = set()
            i = 0
            while i < len(region):
                q = region[i]
                i += 1
                for nq in NEIGHBOURS[q]:
                    v = self.stones[nq]
                    if v == EMPTY:
                        if not visited[nq]:
                            visited[nq] = 1
                            region.append(nq)
                    else:
                        borders.add(v)
            if len(borders) == 1:
                owner = next(iter(borders))
                territory[owner - 1] += len(region)
        black = counts[BLACK] + territory[0]
        white = counts[WHITE] + territory[1] + komi
        return float(black), float(white)


# ---------------- candidate heuristic + ordering ---------------------


def _centre_distance(p: int) -> int:
    r, c = divmod(p, BOARD_SIZE)
    cr = cc = BOARD_SIZE // 2
    return max(abs(r - cr), abs(c - cc))  # Chebyshev


def heuristic_score(board: Board, p: int, color: int) -> int:
    """Static priority for a candidate move (higher = better).

    Tactical bonuses dominate so that capturing or saving a group is
    always considered before pure positional play.
    """
    r, c = divmod(p, BOARD_SIZE)
    score = 0
    # Centre attraction (kept modest).
    score += (BOARD_SIZE // 2 - _centre_distance(p)) * 2
    # Avoid 1st line; mild boost for 3rd / 4th lines.
    if 0 in (r, c) or r == BOARD_SIZE - 1 or c == BOARD_SIZE - 1:
        score -= 5
    if 2 <= r <= BOARD_SIZE - 3 and 2 <= c <= BOARD_SIZE - 3:
        score += 1
    # Adjacent to stones (any colour).
    for nq in NEIGHBOURS[p]:
        if board.stones[nq] != EMPTY:
            score += 3
    for d in DIAGONALS[p]:
        if board.stones[d] != EMPTY:
            score += 1
    # Proximity to last move (highly relevant in Go).
    if board.last_move >= 0:
        lr, lc = divmod(board.last_move, BOARD_SIZE)
        dist = max(abs(r - lr), abs(c - lc))
        if dist <= 2:
            score += (3 - dist) * 3

    # ---- tactical bonuses ----
    opp = WHITE if color == BLACK else BLACK
    stones = board.stones
    # Bonus if this point captures an opp group in atari.
    capture_bonus = 0
    for nq in NEIGHBOURS[p]:
        if stones[nq] == opp:
            _, libs = board._group_and_libs(nq)
            if len(libs) == 1 and next(iter(libs)) == p:
                # Bigger bonus for bigger captures.
                capture_bonus = max(capture_bonus, 60)
    score += capture_bonus
    # Bonus if this point saves one of our groups currently in atari.
    save_bonus = 0
    for nq in NEIGHBOURS[p]:
        if stones[nq] == color:
            _, libs = board._group_and_libs(nq)
            if len(libs) == 1 and next(iter(libs)) == p:
                save_bonus = max(save_bonus, 50)
    score += save_bonus
    # Bonus if this point puts an opp group into atari (without capturing).
    # Detected by: an adjacent opp group has 2 liberties and we play in one.
    threat_bonus = 0
    seen_groups: set[int] = set()
    for nq in NEIGHBOURS[p]:
        if stones[nq] == opp and nq not in seen_groups:
            grp, libs = board._group_and_libs(nq)
            for s in grp:
                seen_groups.add(s)
            if len(libs) == 2 and p in libs:
                threat_bonus = max(threat_bonus, 15)
    score += threat_bonus
    return score


def candidate_moves(
    board: Board,
    color: int,
    *,
    limit: int,
    include_pass: bool = True,
) -> list[int]:
    """Top-`limit` legal non-eye moves by heuristic_score.

    PASS is appended last when `include_pass`.
    """
    scored: list[tuple[int, int]] = []
    for p in range(N):
        if board.stones[p] != EMPTY:
            continue
        if p == board.ko:
            continue
        if board.is_true_eye(p, color):
            continue
        if not board.is_legal(p, color):
            continue
        scored.append((heuristic_score(board, p, color), p))
    # Pick top-`limit` then shuffle ties so we don't lock in one specific
    # neighbouring point each time.
    scored.sort(reverse=True)
    top = scored[:limit]
    random.shuffle(top)
    moves = [p for _, p in top]
    if include_pass:
        moves.append(PASS_MOVE)
    return moves


# ----------------------------- playouts --------------------------------


def run_playout(
    board: Board,
    to_move: int,
    komi: float,
    moves_by_color: list[set[int]],
) -> int:
    """Capture-priority eye-aware random playout. Returns winner colour."""
    stones = board.stones
    NB = NEIGHBOURS
    consecutive_pass = 0

    for _ in range(MAX_PLAYOUT_MOVES):
        # 1. Capture priority.
        chosen = PASS_MOVE
        if random.random() < CAPTURE_PRIOR_PROB:
            cap = board.find_atari_capture(to_move)
            if cap >= 0 and not board.is_true_eye(cap, to_move):
                chosen = cap
        # 2. Random eye-aware fallback.
        if chosen == PASS_MOVE:
            empties: list[int] = []
            for p in range(N):
                if stones[p] == EMPTY:
                    empties.append(p)
            random.shuffle(empties)
            ko = board.ko
            for p in empties:
                if p == ko:
                    continue
                # Cheap eye precondition.
                surround = True
                for nq in NB[p]:
                    if stones[nq] != to_move:
                        surround = False
                        break
                if surround and board.is_true_eye(p, to_move):
                    continue
                # Fast legality.
                has_lib = False
                for nq in NB[p]:
                    if stones[nq] == EMPTY:
                        has_lib = True
                        break
                if not has_lib:
                    if not board.is_legal(p, to_move):
                        continue
                chosen = p
                break

        if chosen == PASS_MOVE:
            consecutive_pass += 1
            if consecutive_pass >= 2:
                break
        else:
            consecutive_pass = 0
            board.play_move(chosen, to_move)
            moves_by_color[to_move].add(chosen)
        to_move = WHITE if to_move == BLACK else BLACK

    black, white = board.score(komi)
    return BLACK if black > white else WHITE


# ----------------------------- MCTS node --------------------------------


class Node:
    __slots__ = (
        "parent",
        "move",
        "color_to_move",
        "children",
        "visits",
        "wins",
        "rave_visits",
        "rave_wins",
        "untried",
        "is_terminal",
    )

    def __init__(self, parent: "Node | None", move: int, color_to_move: int) -> None:
        self.parent = parent
        self.move = move
        self.color_to_move = color_to_move
        self.children: dict[int, Node] = {}
        self.visits = 0
        self.wins = 0.0
        self.rave_visits: dict[int, int] = {}
        self.rave_wins: dict[int, float] = {}
        self.untried: list[int] | None = None  # candidate moves not yet expanded
        self.is_terminal = False


def _ucb_rave_value(
    parent_visits: int, child: Node, rave_visits: int, rave_wins: float,
) -> float:
    if child.visits == 0:
        return 1e9
    win_rate = child.wins / child.visits
    if rave_visits > 0:
        rave_rate = rave_wins / rave_visits
        beta = math.sqrt(RAVE_EQUIV / (3 * child.visits + RAVE_EQUIV))
        value = (1 - beta) * win_rate + beta * rave_rate
    else:
        value = win_rate
    exploration = UCB_C * math.sqrt(math.log(parent_visits + 1) / child.visits)
    return value + exploration


def select_child(node: Node) -> Node:
    best = None
    best_val = -1.0
    parent_visits = node.visits
    rv_map = node.rave_visits
    rw_map = node.rave_wins
    for move, child in node.children.items():
        rv = rv_map.get(move, 0)
        rw = rw_map.get(move, 0.0)
        val = _ucb_rave_value(parent_visits, child, rv, rw)
        if val > best_val:
            best_val = val
            best = child
    assert best is not None
    return best


# ------------------------------ Engine ----------------------------------


class Engine:
    def __init__(self, komi: float = KOMI) -> None:
        self.komi = komi
        self.root: Node | None = None
        self.root_board: Board | None = None
        self.root_key: bytes | None = None

    def _board_key(self, board: Board, to_move: int) -> bytes:
        # Include ko in the key so we don't reuse a tree across a ko reset.
        return bytes(board.stones) + bytes((to_move, board.ko & 0xFF))

    def _maybe_reuse_tree(self, board: Board, to_move: int) -> None:
        key = self._board_key(board, to_move)
        if self.root is not None and self.root_key == key:
            return
        # Try 1- and 2-ply matches against saved tree.
        if self.root is not None and self.root_board is not None:
            for _, c1 in self.root.children.items():
                b1 = self._reconstruct(self.root_board, c1.move, self.root.color_to_move)
                if b1 is not None and self._board_key(b1, c1.color_to_move) == key:
                    self._adopt(c1, b1, key)
                    return
                for _, c2 in c1.children.items():
                    if b1 is None:
                        break
                    b2 = self._reconstruct(b1, c2.move, c1.color_to_move)
                    if b2 is not None and self._board_key(b2, c2.color_to_move) == key:
                        self._adopt(c2, b2, key)
                        return
        # Fresh root.
        self.root = Node(parent=None, move=PASS_MOVE, color_to_move=to_move)
        self.root_board = board.copy()
        self.root_key = key

    def _adopt(self, node: Node, board: Board, key: bytes) -> None:
        node.parent = None
        self.root = node
        self.root_board = board
        self.root_key = key

    @staticmethod
    def _reconstruct(parent_board: Board, move: int, color: int) -> Board | None:
        b = parent_board.copy()
        if move == PASS_MOVE:
            b.to_move = WHITE if color == BLACK else BLACK
            b.ko = -1
            return b
        if not b.is_legal(move, color):
            return None
        b.play_move(move, color)
        return b

    # ---- public choose ----

    def choose(
        self,
        board: Board,
        color: int,
        *,
        budget: float = TIME_BUDGET,
        opp_passed: bool = False,
    ) -> int:
        start = time.perf_counter()
        deadline = start + budget

        self._maybe_reuse_tree(board, color)
        root = self.root
        assert root is not None
        if self.root_board is None:
            self.root_board = board.copy()

        # Decide PASS-allowed AT THE ROOT. We do this even when the root is
        # being reused from a previous iteration's subtree — because that
        # subtree was created with PASS as a candidate (rollouts pass), and
        # we must strip PASS out before searching, otherwise the bot will
        # converge on PASS on a near-empty board (random rollouts plus
        # komi make PASS look great for white).
        empty_count = sum(1 for s in self.root_board.stones if s == EMPTY)
        pass_allowed = opp_passed or empty_count <= 25
        if not pass_allowed:
            if root.untried is not None:
                root.untried = [m for m in root.untried if m != PASS_MOVE]
            if PASS_MOVE in root.children:
                del root.children[PASS_MOVE]
            root.rave_visits.pop(PASS_MOVE, None)
            root.rave_wins.pop(PASS_MOVE, None)

        # Seed the root candidate list (if not already populated by reuse).
        if root.untried is None:
            root.untried = candidate_moves(
                self.root_board, color, limit=ROOT_BRANCH,
                include_pass=pass_allowed,
            )

        iterations = 0
        while time.perf_counter() < deadline:
            self._iterate(root)
            iterations += 1
            # Cheap early-out: if a single child is overwhelmingly visited and
            # winning, no point continuing.
            if iterations % 64 == 0 and root.visits > 200:
                items = sorted(
                    root.children.items(), key=lambda kv: kv[1].visits, reverse=True,
                )
                if items:
                    top = items[0][1]
                    if top.visits > 0 and top.wins / top.visits > 0.97:
                        break

        # Final selection: most visited; break ties by win rate.
        best_move = PASS_MOVE
        best_score = (-1, -1.0)
        for move, child in root.children.items():
            visits = child.visits
            wr = (child.wins / visits) if visits > 0 else 0.0
            score = (visits, wr)
            if score > best_score:
                best_score = score
                best_move = move

        # Resign-via-pass (winning side ends a settled game quickly).
        # Only when the game is almost over AND we lead by a clear
        # territorial margin (komi alone isn't enough — random rollouts on
        # near-empty boards bake in komi).
        pass_child = root.children.get(PASS_MOVE)
        if pass_child and pass_child.visits >= 50:
            black, white = self.root_board.score(self.komi)
            stones_on_board = sum(
                1 for s in self.root_board.stones if s != EMPTY
            )
            if color == BLACK:
                lead_no_komi = black - (white - self.komi)
                we_lead = lead_no_komi > 4 and black > white + 2.0
            else:
                lead_no_komi = (white - self.komi) - black
                we_lead = lead_no_komi > 4 and white > black + 2.0
            late_game = stones_on_board >= 55 or opp_passed
            if (
                we_lead
                and late_game
                and pass_child.wins / pass_child.visits >= 0.95
            ):
                best_move = PASS_MOVE

        # ---- Resignation when hopelessly behind ----
        # Two-tier resignation. Once enough stones are placed and the
        # search has run a reasonable amount, PASS if:
        #   (a) MCTS reports best-move win rate < 0.15 AND we already trail
        #       in area by more than ~8 points, OR
        #   (b) we trail in area by more than ~25 points regardless of the
        #       win-rate estimate (lopsided positions where MCTS hasn't
        #       converged but the static eval is unambiguous).
        # Both tiers require enough stones on the board so that the static
        # score is meaningful — early-game scoring is dominated by single-
        # colour neutral regions that flip wildly.
        if best_move != PASS_MOVE and root.visits >= 200:
            best_child = root.children.get(best_move)
            stones_on_board = sum(
                1 for s in self.root_board.stones if s != EMPTY
            )
            if (
                best_child is not None
                and best_child.visits >= 30
                and stones_on_board >= 25
            ):
                best_wr = best_child.wins / best_child.visits
                black, white = self.root_board.score(self.komi)
                if color == BLACK:
                    behind_by = white - black
                else:
                    behind_by = black - white
                if (
                    (best_wr < 0.15 and behind_by > 8)
                    or behind_by > 25
                ):
                    return PASS_MOVE

        return best_move

    # ---- single iteration ----

    def _iterate(self, root: Node) -> None:
        board = self.root_board.copy()  # type: ignore[union-attr]
        node = root

        # ---- SELECTION ----
        while node.untried is not None and not node.untried and node.children:
            child = select_child(node)
            if child.move == PASS_MOVE:
                board.to_move = WHITE if node.color_to_move == BLACK else BLACK
                board.ko = -1
            else:
                board.play_move(child.move, node.color_to_move)
            node = child

        # ---- EXPANSION ----
        if not node.is_terminal:
            if node.untried is None:
                limit = ROOT_BRANCH if node is root else INNER_BRANCH
                node.untried = candidate_moves(board, node.color_to_move, limit=limit)
            if node.untried:
                move = node.untried.pop(0)
                next_color = WHITE if node.color_to_move == BLACK else BLACK
                if move == PASS_MOVE:
                    board.to_move = next_color
                    board.ko = -1
                else:
                    board.play_move(move, node.color_to_move)
                child = Node(parent=node, move=move, color_to_move=next_color)
                node.children[move] = child
                node = child

        # ---- SIMULATION ----
        moves_by_color: list[set[int]] = [set(), set(), set()]
        if node.parent is not None and node.move != PASS_MOVE and node.move >= 0:
            moves_by_color[node.parent.color_to_move].add(node.move)
        winner = run_playout(board, board.to_move, self.komi, moves_by_color)

        # ---- BACKPROP with RAVE ----
        cur: Node | None = node
        while cur is not None:
            cur.visits += 1
            if cur.parent is not None:
                moved_color = cur.parent.color_to_move
                if winner == moved_color:
                    cur.wins += 1.0
            c = cur.color_to_move
            played_set = moves_by_color[c]
            if played_set:
                for move, child in cur.children.items():
                    if move in played_set:
                        cur.rave_visits[move] = cur.rave_visits.get(move, 0) + 1
                        if winner == c:
                            cur.rave_wins[move] = cur.rave_wins.get(move, 0.0) + 1.0
            cur = cur.parent


# ----------------------------- public Bot -------------------------------


class Bot:
    name = "claude_opus4p7_go9_hard"

    def __init__(self) -> None:
        self._engine = Engine(komi=KOMI)

    def choose_action(self, state: dict[str, Any]) -> str:
        legal = state["legal_actions"]
        if not legal:
            return "PASS"
        if len(legal) == 1:
            return legal[0]

        board = Board()
        board.from_rows(state["board"])
        color = state.get("actor", state.get("player_id", 0)) + 1
        board.to_move = color
        # We don't have play history to set `last_move` precisely; best effort:
        last_move_str = state.get("last_move")
        opp_passed = last_move_str == "PASS"
        if isinstance(last_move_str, str) and last_move_str != "PASS":
            board.last_move = action_to_point(last_move_str)
        # Ko: env exposes only `ko_active` (a flag), not the ko square. Without
        # the square we leave ko = -1 and rely on `legal_actions` (filtered by
        # the env) to gate the root move. Inside the tree the engine tracks
        # its own ko via play_move.
        komi = float(state.get("komi", KOMI))
        self._engine.komi = komi

        point = self._engine.choose(
            board, color, budget=_time_budget(state, TIME_BUDGET), opp_passed=opp_passed,
        )
        action = point_to_action(point)
        legal_set = set(legal)
        if action in legal_set:
            return action
        # Fallback: most-visited *legal* root child.
        if self._engine.root is not None:
            ranked = sorted(
                self._engine.root.children.items(),
                key=lambda kv: kv[1].visits,
                reverse=True,
            )
            for move, _ in ranked:
                a = point_to_action(move)
                if a in legal_set:
                    return a
        if "PASS" in legal_set:
            return "PASS"
        return legal[0]


_MODULE_ENGINE: Engine | None = None


def choose_action(state: dict[str, Any]) -> str:
    global _MODULE_ENGINE
    if _MODULE_ENGINE is None:
        _MODULE_ENGINE = Engine(komi=KOMI)
    bot = Bot()
    bot._engine = _MODULE_ENGINE
    return bot.choose_action(state)


def _time_budget(state: dict[str, Any], fallback: float) -> float:
    timeout = state.get("decision_timeout") or state.get("time_limit")
    if timeout:
        return max(0.05, float(timeout) - SAFETY_MARGIN_SECONDS)
    return fallback
