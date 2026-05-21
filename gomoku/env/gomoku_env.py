#!/usr/bin/env python3
"""Gym-style 15x15 Gomoku (Renju) environment and bot battle API.

The rule engine is dependency-free. It implements standard Renju rules:

  - Standard 15x15 board, black moves first.
  - Five-in-a-row wins immediately for the player who just moved.
  - Renju forbidden moves apply only to black: a black move is illegal if
    it (a) creates an overline (6+ in a row) without simultaneously
    creating an exact 5-in-a-row, (b) creates a "double four" (the move
    creates two or more distinct fours), or (c) creates a "double three"
    (the move creates two or more distinct open threes).
    Exception: a black move that creates an exact five always wins and
    bypasses the forbidden-move checks.
  - White has no restrictions; white wins on 5-in-a-row, including overline.

The implementation purposely avoids the deepest renju subtleties (it does
not chain "is the resulting open-four legal" recursively to decide if an
open-three is "fake"). For ordinary play these rules match competitive
renju behavior.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import random
import sys
import threading
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


BOARD_SIZE = 15
SUPPORTED_PLAYERS = 2
BLACK = 0
WHITE = 1
EMPTY = "."
DEFAULT_MAX_PLIES = BOARD_SIZE * BOARD_SIZE  # 225

FILES = "abcdefghijklmno"  # 15 columns: a..o
SYMBOLS = ("B", "W")
COLORS = ("black", "white")
COLOR_NAMES = ("黑方", "白方")

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

_MATCH_LOGS: dict[str, list[str]] = {}


class IllegalActionError(ValueError):
    """Raised when an action is not legal in the current position."""


class BotTimeoutError(TimeoutError):
    """Raised when a bot does not return an action before the deadline."""


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def as_tuple(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self.observation, self.reward, self.terminated, self.truncated, self.info


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------


def opponent(player: int) -> int:
    return 1 - player


def player_symbol(player: int) -> str:
    return SYMBOLS[player]


def new_board() -> list[list[str]]:
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def square_to_coords(square: str) -> tuple[int, int] | None:
    if not isinstance(square, str) or len(square) < 2 or len(square) > 3:
        return None
    file_char = square[0].lower()
    rank_text = square[1:]
    if file_char not in FILES:
        return None
    try:
        rank = int(rank_text)
    except ValueError:
        return None
    if not (1 <= rank <= BOARD_SIZE):
        return None
    return rank - 1, FILES.index(file_char)


def coords_to_square(row: int, col: int) -> str:
    return f"{FILES[col]}{row + 1}"


def on_board(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def board_from_rows(rows: list[str]) -> list[list[str]]:
    if len(rows) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in rows):
        raise ValueError(f"board rows must be {BOARD_SIZE} strings of length {BOARD_SIZE}")

    board = new_board()
    # Display order: row[0] = rank 15 (top), row[-1] = rank 1 (bottom).
    for display_index, row_text in enumerate(rows):
        internal_row = BOARD_SIZE - 1 - display_index
        for col, value in enumerate(row_text):
            if value not in {EMPTY, SYMBOLS[BLACK], SYMBOLS[WHITE]}:
                raise ValueError(f"invalid board symbol: {value!r}")
            board[internal_row][col] = value
    return board


def board_to_rows(board: list[list[str]]) -> list[str]:
    return ["".join(board[row]) for row in range(BOARD_SIZE - 1, -1, -1)]


def board_key(board: list[list[str]]) -> str:
    return "".join("".join(row) for row in board)


def copy_board(board: list[list[str]]) -> list[list[str]]:
    return [list(row) for row in board]


def stone_counts(board: list[list[str]]) -> list[int]:
    return [
        sum(cell == SYMBOLS[BLACK] for row in board for cell in row),
        sum(cell == SYMBOLS[WHITE] for row in board for cell in row),
    ]


def empty_count(board: list[list[str]]) -> int:
    return sum(cell == EMPTY for row in board for cell in row)


# ---------------------------------------------------------------------------
# Run length and Renju pattern detection
# ---------------------------------------------------------------------------


def run_length(board: list[list[str]], row: int, col: int, dr: int, dc: int) -> int:
    """Return the length of the consecutive same-color run through (row, col)."""
    target = board[row][col]
    if target == EMPTY:
        return 0
    count = 1
    r, c = row + dr, col + dc
    while on_board(r, c) and board[r][c] == target:
        count += 1
        r += dr
        c += dc
    r, c = row - dr, col - dc
    while on_board(r, c) and board[r][c] == target:
        count += 1
        r -= dr
        c -= dc
    return count


def max_run_through(board: list[list[str]], row: int, col: int) -> int:
    """Maximum run length in any of the 4 directions through (row, col)."""
    return max(run_length(board, row, col, dr, dc) for dr, dc in DIRECTIONS)


def line_window(
    board: list[list[str]],
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: int,
    half: int = 5,
) -> str:
    """Build a 2*half+1 character window centered on (row, col).

    The window uses 'X' for the player's own stones, 'O' for the opponent's,
    '.' for empty and '|' for off-board.
    """
    sym = SYMBOLS[player]
    chars: list[str] = []
    for i in range(-half, half + 1):
        nr = row + i * dr
        nc = col + i * dc
        if not on_board(nr, nc):
            chars.append("|")
        elif board[nr][nc] == sym:
            chars.append("X")
        elif board[nr][nc] == EMPTY:
            chars.append(".")
        else:
            chars.append("O")
    return "".join(chars)


def _count_fours_in_window(window: str, center: int) -> int:
    """Number of distinct 4-X subsets through center that can complete to a 5.

    Each "four" is identified by its set of 4 X positions; the open four
    `.XXXX.` therefore counts as ONE four (not two), even though it has two
    valid completion cells. We also reject "fours" that would extend to a
    6+-run, since those completions yield overlines, not 5-exact.
    """
    n = len(window)
    found: set[frozenset[int]] = set()
    for start in range(n - 4):
        sub = window[start:start + 5]
        if sub.count("X") != 4 or sub.count(".") != 1:
            continue
        x_positions = frozenset(
            start + i for i in range(5) if window[start + i] == "X"
        )
        # The four must include the placed stone at `center`.
        if center not in x_positions:
            continue
        # Reject completions that produce 6+ runs.
        if start > 0 and window[start - 1] == "X":
            continue
        if start + 5 < n and window[start + 5] == "X":
            continue
        found.add(x_positions)
    return len(found)


def _count_open_threes_in_window(window: str, center: int) -> int:
    """Number of distinct 3-X open-three patterns through center.

    Three pattern templates are detected (each match contributes one open
    three): consecutive `..XXX..`, gap-left `.X.XX.`, gap-right `.XX.X.`.
    Patterns are deduplicated by their X-position set, so the same three
    is not double counted by overlapping window scans.
    """
    n = len(window)
    found: set[frozenset[int]] = set()

    for start in range(n - 6):
        if window[start:start + 7] == "..XXX..":
            xs = frozenset({start + 2, start + 3, start + 4})
            if center in xs:
                found.add(xs)

    for start in range(n - 5):
        sub = window[start:start + 6]
        if sub == ".X.XX.":
            xs = frozenset({start + 1, start + 3, start + 4})
            if center in xs:
                found.add(xs)
        elif sub == ".XX.X.":
            xs = frozenset({start + 1, start + 2, start + 4})
            if center in xs:
                found.add(xs)

    return len(found)


def count_fours_and_open_threes(
    board: list[list[str]], row: int, col: int, player: int,
) -> tuple[int, int]:
    """Sum fours and open threes through (row, col) across all 4 directions."""
    fours = 0
    threes = 0
    for dr, dc in DIRECTIONS:
        window = line_window(board, row, col, dr, dc, player, half=5)
        fours += _count_fours_in_window(window, 5)
        threes += _count_open_threes_in_window(window, 5)
    return fours, threes


def classify_move(
    board: list[list[str]], row: int, col: int, player: int,
) -> str:
    """Return one of:
    - "win" : the move creates an exact 5-in-a-row (winning).
    - "white_win_overline" : white-only, 6+ run is a win for white.
    - "forbidden_overline" : black-only, 6+ run without an exact 5 elsewhere.
    - "forbidden_double_four" : black-only, move creates >= 2 fours.
    - "forbidden_double_three" : black-only, move creates >= 2 open threes.
    - "ok" : otherwise.
    """
    if not on_board(row, col) or board[row][col] != EMPTY:
        return "illegal_square"

    sym = SYMBOLS[player]
    board[row][col] = sym
    try:
        max_run = 0
        has_overline = False
        has_exact_five = False
        for dr, dc in DIRECTIONS:
            run = run_length(board, row, col, dr, dc)
            if run > max_run:
                max_run = run
            if run == 5:
                has_exact_five = True
            if run >= 6:
                has_overline = True

        if has_exact_five:
            return "win"

        if has_overline:
            if player == WHITE:
                return "white_win_overline"
            return "forbidden_overline"

        if player == BLACK:
            fours, threes = count_fours_and_open_threes(board, row, col, BLACK)
            if fours >= 2:
                return "forbidden_double_four"
            if threes >= 2:
                return "forbidden_double_three"
        return "ok"
    finally:
        board[row][col] = EMPTY


def is_winning_move(board: list[list[str]], row: int, col: int, player: int) -> bool:
    """True iff playing `player` at (row, col) would win the game."""
    status = classify_move(board, row, col, player)
    return status == "win" or status == "white_win_overline"


def legal_actions_for(board: list[list[str]], player: int) -> list[str]:
    """List of legal action strings for `player`, sorted by square name."""
    actions: list[str] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != EMPTY:
                continue
            status = classify_move(board, row, col, player)
            if status.startswith("forbidden"):
                continue
            if status == "illegal_square":
                continue
            actions.append(coords_to_square(row, col))
    actions.sort()
    return actions


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class GomokuEnv:
    """A small alternating-turn Gym-style environment for 15x15 Renju."""

    metadata = {"render_modes": ["ansi"], "players": SUPPORTED_PLAYERS}

    def __init__(
        self,
        *,
        seed: int | None = None,
        max_plies: int | None = DEFAULT_MAX_PLIES,
        board_rows: list[str] | None = None,
        actor: int = BLACK,
    ) -> None:
        if actor not in (BLACK, WHITE):
            raise ValueError("actor must be 0 (black) or 1 (white)")
        self.max_plies = max_plies
        self.rng = random.Random(seed)
        self.initial_rows = list(board_rows) if board_rows is not None else None
        self.initial_actor = actor
        self.board = (
            board_from_rows(board_rows) if board_rows is not None else new_board()
        )
        self.actor = actor
        self.plies = 0
        self.last_move: str | None = None
        self.winner: int | None = None
        self.terminal_status: str | None = None
        self.history: list[dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        board_rows: list[str] | None = None,
        actor: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        if actor is not None and actor not in (BLACK, WHITE):
            raise ValueError("actor must be 0 (black) or 1 (white)")

        rows = board_rows if board_rows is not None else self.initial_rows
        self.board = board_from_rows(rows) if rows is not None else new_board()
        self.actor = self.initial_actor if actor is None else actor
        self.plies = 0
        self.last_move = None
        self.winner = None
        self.terminal_status = None
        self.history = []
        return self.state(), {
            "actor": self.actor,
            "board": board_to_rows(self.board),
        }

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.is_done():
            return self.state(), 0.0, True, False, self._outcome_info()

        legal = self.legal_actions()
        if action not in legal:
            raise IllegalActionError(f"{action!r} is not legal in the current position")

        actor = self.actor
        coords = square_to_coords(action)
        if coords is None:
            raise IllegalActionError(f"{action!r} is not a valid coordinate")
        row, col = coords
        status = classify_move(self.board, row, col, actor)
        # `legal` already filters forbidden moves; status here is "win",
        # "white_win_overline", or "ok".

        self.board[row][col] = SYMBOLS[actor]
        self.last_move = action
        self.history.append({
            "ply": self.plies + 1,
            "player": actor,
            "action": action,
        })
        self.plies += 1

        if status in ("win", "white_win_overline"):
            self.winner = actor
            self.terminal_status = "five_in_row" if status == "win" else "overline_win"
        elif empty_count(self.board) == 0:
            self.terminal_status = "board_full"
        else:
            next_actor = opponent(actor)
            if not legal_actions_for(self.board, next_actor):
                # Opponent has no legal moves — they lose.
                self.winner = actor
                self.terminal_status = "no_legal_actions"

        self.actor = opponent(actor)

        terminated = self.is_done()
        truncated = bool(
            self.max_plies is not None
            and self.plies >= self.max_plies
            and not terminated
        )
        info = self._outcome_info() if terminated or truncated else {}
        reward = self._reward_for(actor, info)
        return self.state(), reward, terminated, truncated, info

    def legal_actions(self) -> list[str]:
        if self.is_done():
            return []
        return legal_actions_for(self.board, self.actor)

    def is_done(self) -> bool:
        return self.winner is not None or self.terminal_status is not None

    def state(self, player_id: int | None = None) -> dict[str, Any]:
        done = self.is_done()
        legal = [] if done else self.legal_actions()
        counts = stone_counts(self.board)

        last_player: int | None = None
        if self.last_move is not None and self.history:
            last_player = self.history[-1]["player"]

        return {
            "player_id": self.actor if player_id is None else player_id,
            "num_players": SUPPORTED_PLAYERS,
            "phase": "game_over" if done else "turn",
            "actor": self.actor,
            "turn": COLORS[self.actor],
            "legal_actions": legal,
            "board": board_to_rows(self.board),
            "board_size": BOARD_SIZE,
            "pieces": self._pieces(),
            "stone_counts": {"black": counts[BLACK], "white": counts[WHITE]},
            "scores": counts,
            "empty_count": empty_count(self.board),
            "plies": self.plies,
            "last_move": self.last_move,
            "last_player": last_player,
            "history": list(self.history),
            "winner": self.winner,
            "status": self.terminal_status,
            "result": _result_for_winner(self.winner) if done else "*",
        }

    def render(self) -> str:
        header = "    " + " ".join(FILES)
        lines = [header]
        for row in range(BOARD_SIZE - 1, -1, -1):
            cells = " ".join(self.board[row])
            lines.append(f"{row + 1:>3} {cells} {row + 1}")
        lines.append(header)
        counts = stone_counts(self.board)
        lines.append(
            f"black={counts[BLACK]} white={counts[WHITE]} turn={COLORS[self.actor]} "
            f"plies={self.plies} status={self.terminal_status}"
        )
        return "\n".join(lines)

    def _pieces(self) -> list[dict[str, str]]:
        pieces: list[dict[str, str]] = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                value = self.board[row][col]
                if value == EMPTY:
                    continue
                player = BLACK if value == SYMBOLS[BLACK] else WHITE
                pieces.append({
                    "square": coords_to_square(row, col),
                    "color": COLORS[player],
                    "symbol": value,
                })
        return sorted(pieces, key=lambda item: item["square"])

    def _outcome_info(self) -> dict[str, Any]:
        if (
            self.max_plies is not None
            and self.plies >= self.max_plies
            and not self.is_done()
        ):
            return {
                "status": "turn_limit",
                "winner": None,
                "result": "*",
                "termination": "turn_limit",
            }
        return {
            "status": self.terminal_status,
            "winner": self.winner,
            "result": _result_for_winner(self.winner),
            "termination": self.terminal_status,
        }

    @staticmethod
    def _reward_for(actor: int, info: dict[str, Any]) -> float:
        winner = info.get("winner")
        if winner is None:
            return 0.0
        return 1.0 if winner == actor else -1.0


# ---------------------------------------------------------------------------
# Battle infrastructure
# ---------------------------------------------------------------------------


class SystemBot:
    """A simple opponent: take any winning move, block any opponent winning
    move, otherwise play near recent moves with a preference for the center."""

    name = "system"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_action(self, state: dict[str, Any]) -> str:
        legal = state["legal_actions"]
        if not legal:
            raise ValueError("system bot received no legal actions")
        if len(legal) == 1:
            return legal[0]

        board = board_from_rows(state["board"])
        me = state["actor"]
        opp = opponent(me)

        for action in legal:
            row, col = square_to_coords(action)  # type: ignore[misc]
            status = classify_move(board, row, col, me)
            if status in ("win", "white_win_overline"):
                return action

        for action in legal:
            row, col = square_to_coords(action)  # type: ignore[misc]
            if board[row][col] != EMPTY:
                continue
            status = classify_move(board, row, col, opp)
            if status in ("win", "white_win_overline"):
                return action

        # Otherwise prefer cells near existing stones; fall back to center.
        center = BOARD_SIZE // 2
        scored: list[tuple[float, str]] = []
        for action in legal:
            row, col = square_to_coords(action)  # type: ignore[misc]
            near = 0
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if on_board(nr, nc) and board[nr][nc] != EMPTY:
                        near += 1
            dist_to_center = abs(row - center) + abs(col - center)
            scored.append((-near * 4 + dist_to_center, action))
        scored.sort()
        best = scored[0][0]
        choices = [a for s, a in scored if s == best]
        return self.rng.choice(choices)


class CallableBot:
    def __init__(self, choose_action: Callable[[dict[str, Any]], str], name: str) -> None:
        self.choose_action = choose_action
        self.name = name


def choose_action_with_timeout(
    bot: Any,
    state: dict[str, Any],
    decision_timeout: float | None,
) -> Any:
    if decision_timeout is None:
        return bot.choose_action(state)

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            result_queue.put((True, bot.choose_action(state)))
        except BaseException as exc:  # noqa: BLE001
            result_queue.put((False, exc))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(decision_timeout)
    if thread.is_alive():
        raise BotTimeoutError(f"choose_action exceeded {decision_timeout:g} seconds")

    ok, value = result_queue.get_nowait()
    if ok:
        return value
    raise value


def validate_decision_timeout(decision_timeout: float | None) -> None:
    if decision_timeout is not None and decision_timeout <= 0:
        raise ValueError("decision_timeout must be positive seconds or None")


def load_bot(bot_path: str | Path) -> Any:
    path = Path(bot_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"bot file not found: {path}")

    module_name = f"gomoku_user_bot_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import bot file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, "Bot"):
        bot = module.Bot()
        if not hasattr(bot, "choose_action"):
            raise AttributeError("Bot must define choose_action(state)")
        if not hasattr(bot, "name"):
            bot.name = path.stem
        return bot

    if hasattr(module, "choose_action"):
        return CallableBot(module.choose_action, getattr(module, "name", path.stem))

    raise AttributeError("bot.py must define choose_action(state) or class Bot")


def battle_once(
    bot_path: str | Path,
    *,
    players: int = SUPPORTED_PLAYERS,
    seat: int = BLACK,
    seed: int | None = None,
    keep_log: bool = True,
    max_plies: int = DEFAULT_MAX_PLIES,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("gomoku only supports players=2")
    if seat not in (BLACK, WHITE):
        raise ValueError("seat must be 0 or 1")
    validate_decision_timeout(decision_timeout)

    game_seed = seed if seed is not None else random.randrange(1 << 30)
    rng = random.Random(game_seed)
    game_id = uuid.uuid4().hex[:12]
    log: list[str] = [f"G:{game_id}:N2:SEED{game_seed}:GAME:GOMOKU"]

    developer_bot = load_bot(bot_path)
    bots: list[Any] = [SystemBot(rng), SystemBot(rng)]
    bots[seat] = developer_bot
    bot_names = [getattr(bot, "name", f"bot_{i}") for i, bot in enumerate(bots)]

    env = GomokuEnv(seed=game_seed, max_plies=max_plies)
    status = "ok"
    error: str | None = None

    while True:
        state = env.state(env.actor)
        actor = state["actor"]
        legal_actions = state["legal_actions"]

        if state["phase"] == "game_over":
            break
        if env.max_plies is not None and env.plies >= env.max_plies:
            status = "turn_limit"
            break
        if not legal_actions:
            status = "no_legal_actions"
            break

        try:
            action = choose_action_with_timeout(bots[actor], state, decision_timeout)
        except BotTimeoutError as exc:
            status = "timeout"
            error = f"player {actor} timed out: {exc}"
            log.append(f"T{env.plies}:ERR:P{actor}:TIMEOUT")
            break
        except Exception as exc:  # noqa: BLE001
            status = "bot_exception"
            error = f"{type(exc).__name__}: {exc}"
            log.append(f"T{env.plies}:ERR:P{actor}:EXCEPTION:{type(exc).__name__}")
            break

        if not isinstance(action, str) or action not in legal_actions:
            status = "invalid_action"
            error = f"invalid action from player {actor}: {action!r}"
            log.append(f"T{env.plies}:ERR:P{actor}:INVALID:{action}")
            break

        _, _, terminated, truncated, info = env.step(action)
        log.append(
            f"T{env.plies}:P{actor}:MOVE:{action}:BOARD:{'/'.join(board_to_rows(env.board))}"
        )
        if terminated or truncated:
            status = "turn_limit" if truncated else info.get("status", "ok")
            break

    final_state = env.state()
    winner = final_state["winner"]
    result = final_state["result"]
    if status in {"bot_exception", "invalid_action", "timeout"}:
        loser = env.actor
        winner = opponent(loser)
        result = _result_for_winner(winner)
    elif status == "turn_limit":
        winner = None
        result = "*"
    elif status == "no_legal_actions":
        winner = env.winner
        result = _result_for_winner(winner)

    scores = final_state["scores"]
    log.append(
        "END:"
        f"{status}:WINNER:{winner}:RESULT:{result}:PLIES:{env.plies}:"
        f"SCORE:{scores[BLACK]}-{scores[WHITE]}:BOARD:{'/'.join(board_to_rows(env.board))}"
    )

    if keep_log:
        _MATCH_LOGS[game_id] = log

    developer_win = winner == seat
    return {
        "game_id": game_id,
        "winner": winner,
        "status": status,
        "result": result,
        "plies": env.plies,
        "scores": scores,
        "board": board_to_rows(env.board),
        "bot_names": bot_names,
        "developer_seat": seat,
        "developer_win": developer_win,
        "error": error,
    }


def battle_many(
    bot_path: str | Path,
    *,
    games: int = 100,
    players: int = SUPPORTED_PLAYERS,
    seat: int = BLACK,
    seed: int | None = None,
    alternate_seats: bool = True,
    keep_logs: bool = False,
    max_plies: int = DEFAULT_MAX_PLIES,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("gomoku only supports players=2")
    if games < 1:
        raise ValueError("games must be >= 1")
    validate_decision_timeout(decision_timeout)

    wins_by_seat = [0, 0]
    statuses: Counter[str] = Counter()
    developer_wins = 0
    developer_losses = 0
    draws = 0
    game_ids: list[str] = []

    for index in range(games):
        game_seat = (seat + index) % 2 if alternate_seats else seat
        game_seed = None if seed is None else seed + index
        result = battle_once(
            bot_path,
            players=players,
            seat=game_seat,
            seed=game_seed,
            keep_log=keep_logs,
            max_plies=max_plies,
            decision_timeout=decision_timeout,
        )
        statuses[result["status"]] += 1
        if result["winner"] in (BLACK, WHITE):
            wins_by_seat[result["winner"]] += 1
        else:
            draws += 1
        if result["developer_win"]:
            developer_wins += 1
        else:
            developer_losses += 1
        if keep_logs:
            game_ids.append(result["game_id"])

    summary = {
        "games": games,
        "players": players,
        "wins_by_seat": wins_by_seat,
        "draws": draws,
        "developer_wins": developer_wins,
        "developer_losses": developer_losses,
        "developer_win_rate": developer_wins / games,
        "statuses": dict(statuses),
    }
    if keep_logs:
        summary["game_ids"] = game_ids
    return summary


def battle_bots_once(
    bot0_path: str | Path,
    bot1_path: str | Path,
    *,
    bot0_seat: int = BLACK,
    seed: int | None = None,
    keep_log: bool = True,
    max_plies: int = DEFAULT_MAX_PLIES,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if bot0_seat not in (BLACK, WHITE):
        raise ValueError("bot0_seat must be 0 or 1")
    validate_decision_timeout(decision_timeout)

    game_seed = seed if seed is not None else random.randrange(1 << 30)
    game_id = uuid.uuid4().hex[:12]
    log: list[str] = [f"G:{game_id}:N2:SEED{game_seed}:GAME:GOMOKU"]

    bot0 = load_bot(bot0_path)
    bot1 = load_bot(bot1_path)
    bots: list[Any] = [None, None]
    bots[bot0_seat] = bot0
    bots[1 - bot0_seat] = bot1
    bot_paths = [str(bot0_path), str(bot1_path)]
    bot_names = [getattr(bot, "name", f"bot_{i}") for i, bot in enumerate(bots)]
    bot_seats = [bot0_seat, 1 - bot0_seat]

    env = GomokuEnv(seed=game_seed, max_plies=max_plies)
    status = "ok"
    error: str | None = None

    while True:
        state = env.state(env.actor)
        actor = state["actor"]
        legal_actions = state["legal_actions"]

        if state["phase"] == "game_over":
            break
        if env.max_plies is not None and env.plies >= env.max_plies:
            status = "turn_limit"
            break
        if not legal_actions:
            status = "no_legal_actions"
            break

        try:
            action = choose_action_with_timeout(bots[actor], state, decision_timeout)
        except BotTimeoutError as exc:
            status = "timeout"
            error = f"player {actor} timed out: {exc}"
            log.append(f"T{env.plies}:ERR:P{actor}:TIMEOUT")
            break
        except Exception as exc:  # noqa: BLE001
            status = "bot_exception"
            error = f"{type(exc).__name__}: {exc}"
            log.append(f"T{env.plies}:ERR:P{actor}:EXCEPTION:{type(exc).__name__}")
            break

        if not isinstance(action, str) or action not in legal_actions:
            status = "invalid_action"
            error = f"invalid action from player {actor}: {action!r}"
            log.append(f"T{env.plies}:ERR:P{actor}:INVALID:{action}")
            break

        _, _, terminated, truncated, info = env.step(action)
        log.append(
            f"T{env.plies}:P{actor}:MOVE:{action}:BOARD:{'/'.join(board_to_rows(env.board))}"
        )
        if terminated or truncated:
            status = "turn_limit" if truncated else info.get("status", "ok")
            break

    final_state = env.state()
    winner = final_state["winner"]
    result = final_state["result"]
    if status in {"bot_exception", "invalid_action", "timeout"}:
        loser = env.actor
        winner = opponent(loser)
        result = _result_for_winner(winner)
    elif status == "turn_limit":
        winner = None
        result = "*"
    elif status == "no_legal_actions":
        winner = env.winner
        result = _result_for_winner(winner)

    winner_bot: int | None
    if winner is None:
        winner_bot = None
    elif winner == bot0_seat:
        winner_bot = 0
    else:
        winner_bot = 1

    scores = final_state["scores"]
    log.append(
        "END:"
        f"{status}:WINNER:{winner}:RESULT:{result}:PLIES:{env.plies}:"
        f"SCORE:{scores[BLACK]}-{scores[WHITE]}:BOARD:{'/'.join(board_to_rows(env.board))}"
    )

    if keep_log:
        _MATCH_LOGS[game_id] = log

    return {
        "game_id": game_id,
        "winner": winner,
        "winner_bot": winner_bot,
        "status": status,
        "result": result,
        "plies": env.plies,
        "scores": scores,
        "board": board_to_rows(env.board),
        "bot_names": bot_names,
        "bot_seats": bot_seats,
        "bot_paths": bot_paths,
        "error": error,
    }


def battle_bots_many(
    bot0_path: str | Path,
    bot1_path: str | Path,
    *,
    games: int = 50,
    seed: int | None = None,
    alternate_seats: bool = True,
    keep_logs: bool = False,
    max_plies: int = DEFAULT_MAX_PLIES,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if games < 1:
        raise ValueError("games must be >= 1")
    validate_decision_timeout(decision_timeout)

    wins_by_bot = [0, 0]
    wins_by_seat = [0, 0]
    statuses: Counter[str] = Counter()
    game_ids: list[str] = []

    for index in range(games):
        bot0_seat = (BLACK + index) % 2 if alternate_seats else BLACK
        game_seed = None if seed is None else seed + index
        result = battle_bots_once(
            bot0_path,
            bot1_path,
            bot0_seat=bot0_seat,
            seed=game_seed,
            keep_log=keep_logs,
            max_plies=max_plies,
            decision_timeout=decision_timeout,
        )
        statuses[result["status"]] += 1
        if result["winner"] in (BLACK, WHITE):
            wins_by_seat[result["winner"]] += 1
        if result["winner_bot"] is not None:
            wins_by_bot[result["winner_bot"]] += 1
        if keep_logs:
            game_ids.append(result["game_id"])

    summary = {
        "games": games,
        "players": SUPPORTED_PLAYERS,
        "bot_paths": [str(bot0_path), str(bot1_path)],
        "wins_by_bot": wins_by_bot,
        "wins_by_seat": wins_by_seat,
        "bot0_win_rate": wins_by_bot[0] / games,
        "bot1_win_rate": wins_by_bot[1] / games,
        "statuses": dict(statuses),
    }
    if keep_logs:
        summary["game_ids"] = game_ids
    return summary


def get_match_log(game_id: str) -> list[str]:
    return list(_MATCH_LOGS.get(game_id, []))


def write_sample_bot(path: str | Path = "bot.py") -> None:
    sample = '''def choose_action(state):
    legal = state["legal_actions"]
    # Play the center if available, otherwise the first legal move.
    if "h8" in legal:
        return "h8"
    return legal[0]
'''
    Path(path).write_text(sample, encoding="utf-8")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _result_for_winner(winner: int | None) -> str:
    if winner == BLACK:
        return "1-0"
    if winner == WHITE:
        return "0-1"
    return "1/2-1/2"


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="15x15 Gomoku/Renju Gym-style env and bot battle referee")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample-bot", help="write a sample bot.py")
    sample_parser.add_argument("--output", default="bot.py")

    battle_parser = subparsers.add_parser("battle", help="run one or many games")
    battle_parser.add_argument("--bot", required=True)
    battle_parser.add_argument("--players", type=int, default=SUPPORTED_PLAYERS)
    battle_parser.add_argument("--games", type=int, default=1)
    battle_parser.add_argument("--seat", type=int, default=BLACK)
    battle_parser.add_argument("--seed", type=int, default=None)
    battle_parser.add_argument("--keep-logs", action="store_true")
    battle_parser.add_argument("--fixed-seat", action="store_true")
    battle_parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    battle_parser.add_argument("--decision-timeout", type=float, default=None)

    duel_parser = subparsers.add_parser("duel", help="run two arbitrary bot paths against each other")
    duel_parser.add_argument("--bot0", required=True)
    duel_parser.add_argument("--bot1", required=True)
    duel_parser.add_argument("--games", type=int, default=1)
    duel_parser.add_argument("--seed", type=int, default=None)
    duel_parser.add_argument("--keep-logs", action="store_true")
    duel_parser.add_argument("--fixed-seats", action="store_true")
    duel_parser.add_argument("--bot0-seat", type=int, default=BLACK)
    duel_parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    duel_parser.add_argument("--decision-timeout", type=float, default=None)

    args = parser.parse_args(argv)

    try:
        if args.command == "sample-bot":
            write_sample_bot(args.output)
            print(f"wrote {args.output}")
            return 0

        if args.command == "battle":
            if args.games == 1:
                _print_json(
                    battle_once(
                        args.bot,
                        players=args.players,
                        seat=args.seat,
                        seed=args.seed,
                        keep_log=args.keep_logs,
                        max_plies=args.max_plies,
                        decision_timeout=args.decision_timeout,
                    )
                )
            else:
                _print_json(
                    battle_many(
                        args.bot,
                        games=args.games,
                        players=args.players,
                        seat=args.seat,
                        seed=args.seed,
                        alternate_seats=not args.fixed_seat,
                        keep_logs=args.keep_logs,
                        max_plies=args.max_plies,
                        decision_timeout=args.decision_timeout,
                    )
                )
            return 0

        if args.command == "duel":
            if args.games == 1:
                _print_json(
                    battle_bots_once(
                        args.bot0,
                        args.bot1,
                        bot0_seat=args.bot0_seat,
                        seed=args.seed,
                        keep_log=args.keep_logs,
                        max_plies=args.max_plies,
                        decision_timeout=args.decision_timeout,
                    )
                )
            else:
                _print_json(
                    battle_bots_many(
                        args.bot0,
                        args.bot1,
                        games=args.games,
                        seed=args.seed,
                        alternate_seats=not args.fixed_seats,
                        keep_logs=args.keep_logs,
                        max_plies=args.max_plies,
                        decision_timeout=args.decision_timeout,
                    )
                )
            return 0

        return 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
