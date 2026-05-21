#!/usr/bin/env python3
"""Gym-style 9x9 Go environment and bot battle API.

The rule engine is dependency-free and intentionally small. It supports legal
placements, captures, suicide prevention, simple ko, pass/pass ending, and
Chinese-style area scoring with komi.
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
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


BOARD_SIZE = 9
SUPPORTED_PLAYERS = 2
BLACK = 0
WHITE = 1
EMPTY = "."
PASS_ACTION = "PASS"
DEFAULT_KOMI = 6.5
DEFAULT_MAX_PLIES = 256

FILES = "abcdefghi"
RANKS = "123456789"
SYMBOLS = ("B", "W")
COLORS = ("black", "white")
COLOR_NAMES = ("黑方", "白方")

_MATCH_LOGS: dict[str, list[str]] = {}


class IllegalActionError(ValueError):
    """Raised when an action is not legal in the current Go position."""


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


def opponent(player: int) -> int:
    return 1 - player


def player_symbol(player: int) -> str:
    return SYMBOLS[player]


def new_board() -> list[list[str]]:
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def square_to_coords(square: str) -> tuple[int, int] | None:
    if len(square) not in (2, 3):
        return None
    file_char = square[0].lower()
    rank_text = square[1:]
    if file_char not in FILES or rank_text not in RANKS:
        return None
    return int(rank_text) - 1, FILES.index(file_char)


def coords_to_square(row: int, col: int) -> str:
    return f"{FILES[col]}{row + 1}"


def on_board(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def neighbors(row: int, col: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    if row > 0:
        result.append((row - 1, col))
    if row + 1 < BOARD_SIZE:
        result.append((row + 1, col))
    if col > 0:
        result.append((row, col - 1))
    if col + 1 < BOARD_SIZE:
        result.append((row, col + 1))
    return result


def board_from_rows(rows: list[str]) -> list[list[str]]:
    if len(rows) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in rows):
        raise ValueError("board rows must be 9 strings of length 9")

    board = new_board()
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


def collect_group(
    board: list[list[str]],
    row: int,
    col: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    color = board[row][col]
    if color == EMPTY:
        return set(), set()

    group: set[tuple[int, int]] = set()
    liberties: set[tuple[int, int]] = set()
    todo: deque[tuple[int, int]] = deque([(row, col)])
    group.add((row, col))
    while todo:
        current_row, current_col = todo.popleft()
        for next_row, next_col in neighbors(current_row, current_col):
            value = board[next_row][next_col]
            if value == EMPTY:
                liberties.add((next_row, next_col))
            elif value == color and (next_row, next_col) not in group:
                group.add((next_row, next_col))
                todo.append((next_row, next_col))
    return group, liberties


def play_on_board(
    board: list[list[str]],
    player: int,
    action: str,
    *,
    previous_board_key: str | None = None,
) -> tuple[list[list[str]], list[str]] | None:
    if action == PASS_ACTION:
        return copy_board(board), []

    coords = square_to_coords(action)
    if coords is None:
        return None
    row, col = coords
    if board[row][col] != EMPTY:
        return None

    mine = player_symbol(player)
    theirs = player_symbol(opponent(player))
    next_board = copy_board(board)
    next_board[row][col] = mine
    captured: list[str] = []

    for next_row, next_col in neighbors(row, col):
        if next_board[next_row][next_col] != theirs:
            continue
        group, liberties = collect_group(next_board, next_row, next_col)
        if liberties:
            continue
        for stone_row, stone_col in group:
            next_board[stone_row][stone_col] = EMPTY
            captured.append(coords_to_square(stone_row, stone_col))

    own_group, own_liberties = collect_group(next_board, row, col)
    if not own_group or not own_liberties:
        return None
    if previous_board_key is not None and board_key(next_board) == previous_board_key:
        return None
    return next_board, sorted(captured)


def legal_actions_for(
    board: list[list[str]],
    player: int,
    previous_board_key: str | None = None,
) -> list[str]:
    actions: list[str] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != EMPTY:
                continue
            action = coords_to_square(row, col)
            if play_on_board(board, player, action, previous_board_key=previous_board_key) is not None:
                actions.append(action)
    actions.sort()
    actions.append(PASS_ACTION)
    return actions


def score_position(board: list[list[str]], komi: float = DEFAULT_KOMI) -> dict[str, Any]:
    counts = stone_counts(board)
    territory = [0, 0]
    neutral = 0
    visited: set[tuple[int, int]] = set()

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != EMPTY or (row, col) in visited:
                continue

            region: set[tuple[int, int]] = set()
            borders: set[int] = set()
            todo: deque[tuple[int, int]] = deque([(row, col)])
            visited.add((row, col))
            while todo:
                current_row, current_col = todo.popleft()
                region.add((current_row, current_col))
                for next_row, next_col in neighbors(current_row, current_col):
                    value = board[next_row][next_col]
                    if value == EMPTY and (next_row, next_col) not in visited:
                        visited.add((next_row, next_col))
                        todo.append((next_row, next_col))
                    elif value == SYMBOLS[BLACK]:
                        borders.add(BLACK)
                    elif value == SYMBOLS[WHITE]:
                        borders.add(WHITE)

            if len(borders) == 1:
                owner = next(iter(borders))
                territory[owner] += len(region)
            else:
                neutral += len(region)

    black_score = float(counts[BLACK] + territory[BLACK])
    white_score = float(counts[WHITE] + territory[WHITE]) + komi
    if black_score > white_score:
        winner: int | None = BLACK
    elif white_score > black_score:
        winner = WHITE
    else:
        winner = None

    return {
        "stone_counts": counts,
        "territory": territory,
        "neutral": neutral,
        "scores": [black_score, white_score],
        "komi": komi,
        "winner": winner,
        "margin": abs(black_score - white_score),
    }


class Go9x9Env:
    """A small alternating-turn Gym-style environment for 9x9 Go."""

    metadata = {"render_modes": ["ansi"], "players": SUPPORTED_PLAYERS}

    def __init__(
        self,
        *,
        seed: int | None = None,
        max_plies: int | None = DEFAULT_MAX_PLIES,
        komi: float = DEFAULT_KOMI,
        board_rows: list[str] | None = None,
        actor: int = BLACK,
    ) -> None:
        if actor not in (BLACK, WHITE):
            raise ValueError("actor must be 0 for black or 1 for white")
        self.max_plies = max_plies
        self.komi = komi
        self.rng = random.Random(seed)
        self.initial_rows = list(board_rows) if board_rows is not None else None
        self.initial_actor = actor
        self.board = board_from_rows(board_rows) if board_rows is not None else new_board()
        self.actor = actor
        self.plies = 0
        self.pass_count = 0
        self.previous_board_key: str | None = None
        self.last_move: str | None = None
        self.last_captures: list[str] = []
        self.captures = [0, 0]
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
            raise ValueError("actor must be 0 for black or 1 for white")

        rows = board_rows if board_rows is not None else self.initial_rows
        self.board = board_from_rows(rows) if rows is not None else new_board()
        self.actor = self.initial_actor if actor is None else actor
        self.plies = 0
        self.pass_count = 0
        self.previous_board_key = None
        self.last_move = None
        self.last_captures = []
        self.captures = [0, 0]
        self.history = []
        return self.state(), {"actor": self.actor, "board": board_to_rows(self.board), "komi": self.komi}

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.is_done():
            outcome = self._outcome_info()
            return self.state(), 0.0, True, False, outcome

        legal = self.legal_actions()
        if action not in legal:
            raise IllegalActionError(f"{action!r} is not legal in the current position")

        actor = self.actor
        before_key = board_key(self.board)
        if action == PASS_ACTION:
            self.pass_count += 1
            self.last_captures = []
        else:
            played = play_on_board(self.board, actor, action, previous_board_key=self.previous_board_key)
            if played is None:
                raise IllegalActionError(f"{action!r} is not legal in the current position")
            self.board, captures = played
            self.last_captures = captures
            self.captures[actor] += len(captures)
            self.pass_count = 0

        self.previous_board_key = before_key
        self.last_move = action
        self.history.append(
            {
                "ply": self.plies + 1,
                "player": actor,
                "action": action,
                "captures": list(self.last_captures),
                "stone_counts": stone_counts(self.board),
            }
        )
        self.actor = opponent(actor)
        self.plies += 1

        terminated = self.is_done()
        truncated = bool(self.max_plies is not None and self.plies >= self.max_plies and not terminated)
        info = self._outcome_info() if terminated or truncated else {}
        reward = self._reward_for(actor, info)
        return self.state(), reward, terminated, truncated, info

    def legal_actions(self) -> list[str]:
        if self.is_done():
            return []
        return legal_actions_for(self.board, self.actor, previous_board_key=self.previous_board_key)

    def is_done(self) -> bool:
        return self.pass_count >= 2 or empty_count(self.board) == 0

    def state(self, player_id: int | None = None) -> dict[str, Any]:
        done = self.is_done()
        scoring = score_position(self.board, self.komi)
        status = self._status(done)
        winner = scoring["winner"] if done else None
        legal = [] if done else self.legal_actions()

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
            "stone_counts": {"black": scoring["stone_counts"][BLACK], "white": scoring["stone_counts"][WHITE]},
            "captures": {"black": self.captures[BLACK], "white": self.captures[WHITE]},
            "territory": {"black": scoring["territory"][BLACK], "white": scoring["territory"][WHITE]},
            "neutral_points": scoring["neutral"],
            "scores": scoring["scores"],
            "komi": self.komi,
            "empty_count": empty_count(self.board),
            "plies": self.plies,
            "pass_count": self.pass_count,
            "ko_active": self.previous_board_key is not None,
            "last_move": self.last_move,
            "last_captures": list(self.last_captures),
            "history": list(self.history),
            "winner": winner,
            "status": status,
            "result": _result_for_winner(winner) if done else "*",
        }

    def render(self) -> str:
        lines = ["  a b c d e f g h i"]
        for row in range(BOARD_SIZE - 1, -1, -1):
            cells = " ".join(self.board[row])
            lines.append(f"{row + 1} {cells} {row + 1}")
        lines.append("  a b c d e f g h i")
        scoring = score_position(self.board, self.komi)
        lines.append(
            f"black={scoring['scores'][BLACK]:.1f} white={scoring['scores'][WHITE]:.1f} "
            f"turn={COLORS[self.actor]} komi={self.komi:g}"
        )
        return "\n".join(lines)

    def _pieces(self) -> list[dict[str, str]]:
        pieces: list[dict[str, str]] = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                symbol = self.board[row][col]
                if symbol == EMPTY:
                    continue
                player = BLACK if symbol == SYMBOLS[BLACK] else WHITE
                pieces.append(
                    {
                        "square": coords_to_square(row, col),
                        "color": COLORS[player],
                        "symbol": symbol,
                    }
                )
        return sorted(pieces, key=lambda item: item["square"])

    def _status(self, done: bool) -> str | None:
        if not done:
            return None
        if empty_count(self.board) == 0:
            return "board_full"
        return "two_passes"

    def _outcome_info(self) -> dict[str, Any]:
        if self.max_plies is not None and self.plies >= self.max_plies and not self.is_done():
            scoring = score_position(self.board, self.komi)
            return {
                "status": "turn_limit",
                "winner": None,
                "result": "*",
                "termination": "turn_limit",
                "scores": scoring["scores"],
                "territory": scoring["territory"],
            }

        scoring = score_position(self.board, self.komi)
        status = self._status(True)
        return {
            "status": status,
            "winner": scoring["winner"],
            "result": _result_for_winner(scoring["winner"]),
            "termination": status,
            "scores": scoring["scores"],
            "territory": scoring["territory"],
            "margin": scoring["margin"],
        }

    @staticmethod
    def _reward_for(actor: int, info: dict[str, Any]) -> float:
        winner = info.get("winner")
        if winner is None:
            return 0.0
        return 1.0 if winner == actor else -1.0


class SystemBot:
    name = "system"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_action(self, state: dict[str, Any]) -> str:
        legal = state["legal_actions"]
        if not legal:
            raise ValueError("system bot received no legal actions")
        moves = [action for action in legal if action != PASS_ACTION]
        if not moves:
            return PASS_ACTION
        if state["empty_count"] <= 8 and _score_margin_for_actor(state) > 0:
            return PASS_ACTION

        board = board_from_rows(state["board"])
        player = state["actor"]
        scored: list[tuple[int, str]] = []
        for action in moves:
            played = play_on_board(board, player, action)
            if played is None:
                continue
            next_board, captures = played
            row, col = square_to_coords(action) or (0, 0)
            _, liberties = collect_group(next_board, row, col)
            score = 20 * len(captures) + 3 * len(liberties)
            score += 5 if 1 <= row <= 7 and 1 <= col <= 7 else -4
            score += 3 if any(next_board[nr][nc] == player_symbol(player) for nr, nc in neighbors(row, col)) else 0
            scored.append((score, action))
        if not scored:
            return PASS_ACTION
        best_score = max(score for score, _ in scored)
        best = [action for score, action in scored if score == best_score]
        return self.rng.choice(sorted(best))


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
        except BaseException as exc:  # noqa: BLE001 - preserve existing bot failure semantics.
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

    module_name = f"go9_user_bot_{uuid.uuid4().hex}"
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
    komi: float = DEFAULT_KOMI,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("go_9x9 only supports players=2")
    if seat not in (BLACK, WHITE):
        raise ValueError("seat must be 0 or 1")
    validate_decision_timeout(decision_timeout)

    game_seed = seed if seed is not None else random.randrange(1 << 30)
    rng = random.Random(game_seed)
    game_id = uuid.uuid4().hex[:12]
    log: list[str] = [f"G:{game_id}:N2:SEED{game_seed}:GAME:GO_9X9:KOMI:{komi:g}"]

    developer_bot = load_bot(bot_path)
    bots: list[Any] = [SystemBot(rng), SystemBot(rng)]
    bots[seat] = developer_bot
    bot_names = [getattr(bot, "name", f"bot_{index}") for index, bot in enumerate(bots)]

    env = Go9x9Env(seed=game_seed, max_plies=max_plies, komi=komi)
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
        except Exception as exc:  # noqa: BLE001 - bot exceptions are match results.
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
        if action == PASS_ACTION:
            log.append(f"T{env.plies}:P{actor}:PASS")
        else:
            captures = ",".join(env.last_captures)
            log.append(f"T{env.plies}:P{actor}:MOVE:{action}:CAP:{captures}:BOARD:{'/'.join(board_to_rows(env.board))}")
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
    elif status not in {"ok", "turn_limit"} and final_state["status"] is not None:
        status = final_state["status"]

    scores = final_state["scores"]
    log.append(
        "END:"
        f"{status}:WINNER:{winner}:RESULT:{result}:PLIES:{env.plies}:"
        f"SCORE:{scores[BLACK]:.1f}-{scores[WHITE]:.1f}:BOARD:{'/'.join(board_to_rows(env.board))}"
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
        "territory": final_state["territory"],
        "captures": final_state["captures"],
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
    komi: float = DEFAULT_KOMI,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("go_9x9 only supports players=2")
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
            komi=komi,
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


def get_match_log(game_id: str) -> list[str]:
    return list(_MATCH_LOGS.get(game_id, []))


def write_sample_bot(path: str | Path = "bot.py") -> None:
    sample = '''def choose_action(state):
    legal = state["legal_actions"]
    moves = [action for action in legal if action != "PASS"]
    if not moves:
        return "PASS"
    return sorted(moves)[0]
'''
    Path(path).write_text(sample, encoding="utf-8")


def _score_margin_for_actor(state: dict[str, Any]) -> float:
    scores = state["scores"]
    actor = state["actor"]
    return scores[actor] - scores[opponent(actor)]


def _result_for_winner(winner: int | None) -> str:
    if winner == BLACK:
        return "1-0"
    if winner == WHITE:
        return "0-1"
    return "1/2-1/2"


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="9x9 Go Gym-style env and bot battle referee")
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
    battle_parser.add_argument("--komi", type=float, default=DEFAULT_KOMI)
    battle_parser.add_argument("--decision-timeout", type=float, default=None)

    args = parser.parse_args(argv)

    try:
        if args.command == "sample-bot":
            write_sample_bot(args.output)
            print(f"wrote {args.output}")
            return 0

        if args.games == 1:
            _print_json(
                battle_once(
                    args.bot,
                    players=args.players,
                    seat=args.seat,
                    seed=args.seed,
                    keep_log=args.keep_logs,
                    max_plies=args.max_plies,
                    komi=args.komi,
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
                    komi=args.komi,
                    decision_timeout=args.decision_timeout,
                )
            )
        return 0
    except Exception:  # noqa: BLE001 - command line should show the failure.
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
