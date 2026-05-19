"""Two-player referee and bot battle API for Luqiangqi.

The public API mirrors the simple battle interface used by other games in this
workspace: battle_once, battle_many, battle_bots_once, battle_bots_many, and
get_match_log.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


BOARD_SIZE = 9
WALLS_PER_PLAYER = 10
SUPPORTED_PLAYERS = 2
DEFAULT_TURN_LIMIT = 400

MOVE_DELTAS = {
    "MOVE_UP": (-1, 0),
    "MOVE_DOWN": (1, 0),
    "MOVE_LEFT": (0, -1),
    "MOVE_RIGHT": (0, 1),
    "MOVE_UP_LEFT": (-1, -1),
    "MOVE_UP_RIGHT": (-1, 1),
    "MOVE_DOWN_LEFT": (1, -1),
    "MOVE_DOWN_RIGHT": (1, 1),
}

CARDINAL_MOVE_DELTAS = {
    "MOVE_UP": (-1, 0),
    "MOVE_DOWN": (1, 0),
    "MOVE_LEFT": (0, -1),
    "MOVE_RIGHT": (0, 1),
}

_MATCH_LOGS: dict[str, list[str]] = {}


class IllegalActionError(ValueError):
    """Raised when an action is not legal in the current state."""


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


@dataclass
class PlayerState:
    row: int
    col: int
    goal_row: int
    walls: int = WALLS_PER_PLAYER


@dataclass
class GameState:
    rng: random.Random
    current: int = 0
    turn: int = 0
    winner: int | None = None
    players: list[PlayerState] = field(
        default_factory=lambda: [
            PlayerState(row=BOARD_SIZE - 1, col=BOARD_SIZE // 2, goal_row=0),
            PlayerState(row=0, col=BOARD_SIZE // 2, goal_row=BOARD_SIZE - 1),
        ]
    )
    walls: set[tuple[str, int, int]] = field(default_factory=set)
    blocked_edges: set[tuple[str, int, int]] = field(default_factory=set)

    def bot_state(self, player_id: int) -> dict[str, Any]:
        return {
            "player_id": player_id,
            "num_players": SUPPORTED_PLAYERS,
            "phase": "turn",
            "actor": self.current,
            "legal_actions": self.legal_actions(self.current),
            "board_size": BOARD_SIZE,
            "positions": [[p.row, p.col] for p in self.players],
            "goal_rows": [p.goal_row for p in self.players],
            "walls_remaining": [p.walls for p in self.players],
            "walls": [
                {"dir": direction, "row": row, "col": col}
                for direction, row, col in sorted(self.walls)
            ],
            "turn": self.turn,
        }

    def legal_actions(self, player_id: int) -> list[str]:
        actions: list[str] = []
        player = self.players[player_id]

        for action in MOVE_DELTAS:
            if self._move_destination(player_id, action) is not None:
                actions.append(action)

        if player.walls > 0:
            for row in range(BOARD_SIZE - 1):
                for col in range(BOARD_SIZE - 1):
                    for direction in ("H", "V"):
                        action = f"WALL_{direction}_{row}_{col}"
                        if self.validate_wall(direction, row, col):
                            actions.append(action)

        return actions

    def apply_action(self, action: str) -> str:
        actor = self.current
        if action in MOVE_DELTAS:
            player = self.players[actor]
            old_row, old_col = player.row, player.col
            destination = self._move_destination(actor, action)
            if destination is None:
                raise ValueError(f"illegal move action {action!r}")
            player.row, player.col = destination
            if player.row == player.goal_row:
                self.winner = actor
            self.current = 1 - self.current
            self.turn += 1
            return f"T{self.turn}:P{actor}:MOVE:{old_row},{old_col}>{player.row},{player.col}"

        wall = parse_wall_action(action)
        if wall is None:
            raise ValueError(f"unknown action {action!r}")
        direction, row, col = wall
        self._add_wall(direction, row, col)
        self.players[actor].walls -= 1
        self.current = 1 - self.current
        self.turn += 1
        return f"T{self.turn}:P{actor}:WALL:{direction}:{row}:{col}"

    def validate_wall(self, direction: str, row: int, col: int) -> bool:
        if direction not in {"H", "V"}:
            return False
        if not (0 <= row < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1):
            return False
        if (direction, row, col) in self.walls:
            return False
        if (opposite_wall_direction(direction), row, col) in self.walls:
            return False
        if any(edge in self.blocked_edges for edge in wall_edges(direction, row, col)):
            return False

        self._add_wall(direction, row, col)
        has_paths = self._both_players_have_paths()
        self._remove_wall(direction, row, col)
        return has_paths

    def _both_players_have_paths(self) -> bool:
        return self._has_path_to_goal(0) and self._has_path_to_goal(1)

    def _has_path_to_goal(self, player_id: int) -> bool:
        player = self.players[player_id]
        queue: deque[tuple[int, int]] = deque([(player.row, player.col)])
        seen = {(player.row, player.col)}

        while queue:
            row, col = queue.popleft()
            if row == player.goal_row:
                return True
            for next_row, next_col in self._path_neighbors(row, col):
                key = (next_row, next_col)
                if key not in seen:
                    seen.add(key)
                    queue.append(key)
        return False

    def _path_neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for dr, dc in CARDINAL_MOVE_DELTAS.values():
            next_row = row + dr
            next_col = col + dc
            if self._in_bounds(next_row, next_col) and not self._has_wall_between(row, col, next_row, next_col):
                result.append((next_row, next_col))
        return result

    def _move_destination(self, player_id: int, action: str) -> tuple[int, int] | None:
        if action not in MOVE_DELTAS:
            return None
        player = self.players[player_id]
        opponent = self.players[1 - player_id]
        dr, dc = MOVE_DELTAS[action]

        if dr != 0 and dc != 0:
            return self._side_jump_destination(player, opponent, dr, dc)

        row = player.row + dr
        col = player.col + dc

        if not self._in_bounds(row, col):
            return None
        if self._has_wall_between(player.row, player.col, row, col):
            return None
        if row != opponent.row or col != opponent.col:
            return row, col

        jump_row = opponent.row + dr
        jump_col = opponent.col + dc
        if not self._in_bounds(jump_row, jump_col):
            return None
        if self._has_wall_between(opponent.row, opponent.col, jump_row, jump_col):
            return None
        return jump_row, jump_col

    def _side_jump_destination(
        self,
        player: PlayerState,
        opponent: PlayerState,
        dr: int,
        dc: int,
    ) -> tuple[int, int] | None:
        if abs(player.row - opponent.row) + abs(player.col - opponent.col) != 1:
            return None

        toward_row = opponent.row - player.row
        toward_col = opponent.col - player.col
        if dr != toward_row and dc != toward_col:
            return None
        if self._has_wall_between(player.row, player.col, opponent.row, opponent.col):
            return None

        row = player.row + dr
        col = player.col + dc
        if not self._in_bounds(row, col):
            return None
        if self._has_wall_between(opponent.row, opponent.col, row, col):
            return None
        return row, col

    def _has_wall_between(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        if from_row == to_row:
            return ("V", from_row, min(from_col, to_col)) in self.blocked_edges
        if from_col == to_col:
            return ("H", min(from_row, to_row), from_col) in self.blocked_edges
        return True

    def _add_wall(self, direction: str, row: int, col: int) -> None:
        self.walls.add((direction, row, col))
        self.blocked_edges.update(wall_edges(direction, row, col))

    def _remove_wall(self, direction: str, row: int, col: int) -> None:
        self.walls.remove((direction, row, col))
        for edge in wall_edges(direction, row, col):
            self.blocked_edges.remove(edge)

    @staticmethod
    def _in_bounds(row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


class LqqEnv:
    """A small alternating-turn Gym-style environment for Luqiangqi."""

    metadata = {"render_modes": ["ansi"], "players": SUPPORTED_PLAYERS}

    def __init__(
        self,
        *,
        seed: int | None = None,
        turn_limit: int | None = DEFAULT_TURN_LIMIT,
    ) -> None:
        self.turn_limit = turn_limit
        self.rng = random.Random(seed)
        self.game = GameState(rng=self.rng)

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        self.game = GameState(rng=self.rng)
        return self.state(), {"seed": seed}

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.is_done():
            return self.state(), 0.0, self.game.winner is not None, self._is_truncated(), self._info()

        legal = self.legal_actions()
        if action not in legal:
            raise IllegalActionError(f"{action!r} is not legal in the current state")

        actor = self.actor
        event = self.game.apply_action(action)
        terminated = self.game.winner is not None
        truncated = self._is_truncated() and not terminated
        info = self._info(event=event) if terminated or truncated else {"event": event}
        return self.state(), self._reward_for(actor), terminated, truncated, info

    @property
    def actor(self) -> int:
        return self.game.current

    def legal_actions(self) -> list[str]:
        if self.is_done():
            return []
        return self.game.legal_actions(self.game.current)

    def is_done(self) -> bool:
        return self.game.winner is not None or self._is_truncated()

    def state(self, player_id: int | None = None) -> dict[str, Any]:
        target = self.actor if player_id is None else player_id
        state = self.game.bot_state(target)
        truncated = self._is_truncated()
        if self.game.winner is not None or truncated:
            state["phase"] = "game_over"
            state["legal_actions"] = []
        state.update(
            {
                "winner": self.game.winner,
                "status": "win" if self.game.winner is not None else ("turn_limit" if truncated else None),
            }
        )
        return state

    def render(self) -> str:
        cells = [["." for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for index, player in enumerate(self.game.players):
            cells[player.row][player.col] = str(index)
        return "\n".join(" ".join(row) for row in cells)

    def _is_truncated(self) -> bool:
        return self.turn_limit is not None and self.game.turn >= self.turn_limit and self.game.winner is None

    def _info(self, *, event: str | None = None) -> dict[str, Any]:
        status = "win" if self.game.winner is not None else ("turn_limit" if self._is_truncated() else None)
        info: dict[str, Any] = {
            "status": status,
            "winner": self.game.winner,
            "turns": self.game.turn,
            "positions": [[p.row, p.col] for p in self.game.players],
            "walls_remaining": [p.walls for p in self.game.players],
        }
        if event is not None:
            info["event"] = event
        return info

    def _reward_for(self, actor: int) -> float:
        if self.game.winner is None:
            return 0.0
        return 1.0 if self.game.winner == actor else -1.0


class SystemBot:
    name = "system"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_action(self, state: dict[str, Any]) -> str:
        legal = state["legal_actions"]
        moves = [action for action in legal if action.startswith("MOVE_")]
        player_id = state["player_id"]
        current_row = state["positions"][player_id][0]
        goal_row = state["goal_rows"][player_id]

        preferred = "MOVE_UP" if goal_row < current_row else "MOVE_DOWN"
        if preferred in moves:
            return preferred
        if moves:
            return self.rng.choice(moves)
        return self.rng.choice(legal)


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


def wall_edges(direction: str, row: int, col: int) -> list[tuple[str, int, int]]:
    if direction == "H":
        return [("H", row, col), ("H", row, col + 1)]
    return [("V", row, col), ("V", row + 1, col)]


def opposite_wall_direction(direction: str) -> str:
    return "V" if direction == "H" else "H"


def parse_wall_action(action: str) -> tuple[str, int, int] | None:
    parts = action.split("_")
    if len(parts) != 4 or parts[0] != "WALL" or parts[1] not in {"H", "V"}:
        return None
    try:
        return parts[1], int(parts[2]), int(parts[3])
    except ValueError:
        return None


def load_bot(bot_path: str | Path) -> Any:
    path = Path(bot_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"bot file not found: {path}")

    module_name = f"lqq_user_bot_{uuid.uuid4().hex}"
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


def _run_match(
    bots: list[Any],
    *,
    seed: int | None = None,
    keep_log: bool = True,
    turn_limit: int = DEFAULT_TURN_LIMIT,
    decision_timeout: float | None = None,
    developer_seat: int | None = None,
    bot_indices_by_seat: list[int] | None = None,
    bot_paths: list[str] | None = None,
) -> dict[str, Any]:
    if len(bots) != SUPPORTED_PLAYERS:
        raise ValueError("lqq matches require exactly 2 bots")
    validate_decision_timeout(decision_timeout)

    game_seed = seed if seed is not None else random.randrange(1 << 30)
    rng = random.Random(game_seed)
    game_id = uuid.uuid4().hex[:12]
    log: list[str] = [f"G:{game_id}:N2:SEED{game_seed}"]

    bot_names = [getattr(bot, "name", f"bot_{index}") for index, bot in enumerate(bots)]

    state = GameState(rng=rng)
    status = "ok"
    error: str | None = None

    for _ in range(turn_limit):
        actor = state.current
        bot_state = state.bot_state(actor)
        legal_actions = bot_state["legal_actions"]

        if not legal_actions:
            status = "no_legal_actions"
            state.winner = 1 - actor
            error = f"player {actor} has no legal actions"
            log.append(f"T{state.turn}:ERR:P{actor}:NO_LEGAL_ACTIONS")
            break

        try:
            action = choose_action_with_timeout(bots[actor], bot_state, decision_timeout)
        except BotTimeoutError as exc:
            status = "timeout"
            state.winner = 1 - actor
            error = f"player {actor} timed out: {exc}"
            log.append(f"T{state.turn}:ERR:P{actor}:TIMEOUT")
            break
        except Exception as exc:  # noqa: BLE001 - bot exceptions are match results.
            status = "bot_exception"
            state.winner = 1 - actor
            error = f"{type(exc).__name__}: {exc}"
            log.append(f"T{state.turn}:ERR:P{actor}:EXCEPTION:{type(exc).__name__}")
            break

        if not isinstance(action, str) or action not in legal_actions:
            status = "invalid_action"
            state.winner = 1 - actor
            error = f"invalid action from player {actor}: {action!r}"
            log.append(f"T{state.turn}:ERR:P{actor}:INVALID:{action}")
            break

        log.append(state.apply_action(action))
        if state.winner is not None:
            break
    else:
        status = "turn_limit"

    log.append(
        "END:"
        f"{status}:WINNER:{state.winner}:"
        f"POS:{[[p.row, p.col] for p in state.players]}:"
        f"WALLS:{[p.walls for p in state.players]}"
    )

    if keep_log:
        _MATCH_LOGS[game_id] = log

    result: dict[str, Any] = {
        "game_id": game_id,
        "winner": state.winner,
        "status": status,
        "turns": state.turn,
        "positions": [[p.row, p.col] for p in state.players],
        "walls_remaining": [p.walls for p in state.players],
        "bot_names": bot_names,
        "error": error,
    }
    if developer_seat is not None:
        result["developer_seat"] = developer_seat
        result["developer_win"] = state.winner == developer_seat
    if bot_indices_by_seat is not None:
        result["winner_bot"] = None if state.winner is None else bot_indices_by_seat[state.winner]
        result["bot_seats"] = [
            bot_indices_by_seat.index(bot_index)
            for bot_index in range(len(bot_indices_by_seat))
        ]
    if bot_paths is not None:
        result["bot_paths"] = bot_paths
    return result


def battle_once(
    bot_path: str | Path,
    players: int = SUPPORTED_PLAYERS,
    seat: int = 0,
    seed: int | None = None,
    keep_log: bool = True,
    turn_limit: int = DEFAULT_TURN_LIMIT,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("lqq_multi only supports players=2")
    if seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")

    game_seed = seed if seed is not None else random.randrange(1 << 30)
    rng = random.Random(game_seed)
    developer_bot = load_bot(bot_path)
    bots: list[Any] = [SystemBot(rng), SystemBot(rng)]
    bots[seat] = developer_bot
    return _run_match(
        bots,
        seed=game_seed,
        keep_log=keep_log,
        turn_limit=turn_limit,
        decision_timeout=decision_timeout,
        developer_seat=seat,
    )


def battle_many(
    bot_path: str | Path,
    games: int = 100,
    players: int = SUPPORTED_PLAYERS,
    seat: int = 0,
    seed: int | None = None,
    alternate_seats: bool = True,
    keep_logs: bool = False,
    turn_limit: int = DEFAULT_TURN_LIMIT,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("lqq_multi only supports players=2")
    if games < 1:
        raise ValueError("games must be >= 1")
    validate_decision_timeout(decision_timeout)

    wins_by_seat = [0, 0]
    statuses: Counter[str] = Counter()
    developer_wins = 0
    developer_losses = 0
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
            turn_limit=turn_limit,
            decision_timeout=decision_timeout,
        )
        statuses[result["status"]] += 1
        if result["winner"] in (0, 1):
            wins_by_seat[result["winner"]] += 1
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
    bot0_seat: int = 0,
    seed: int | None = None,
    keep_log: bool = True,
    turn_limit: int = DEFAULT_TURN_LIMIT,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    """Run one game between two bot files.

    ``bot0_path`` and ``bot1_path`` are identified as bot indexes 0 and 1 in
    the returned ``winner_bot`` and ``wins_by_bot`` fields, independent of seat.
    """

    if bot0_seat not in (0, 1):
        raise ValueError("bot0_seat must be 0 or 1")

    bot_paths = [str(Path(bot0_path)), str(Path(bot1_path))]
    bot0 = load_bot(bot0_path)
    bot1 = load_bot(bot1_path)
    bots: list[Any] = [bot0, bot1] if bot0_seat == 0 else [bot1, bot0]
    bot_indices_by_seat = [0, 1] if bot0_seat == 0 else [1, 0]
    return _run_match(
        bots,
        seed=seed,
        keep_log=keep_log,
        turn_limit=turn_limit,
        decision_timeout=decision_timeout,
        bot_indices_by_seat=bot_indices_by_seat,
        bot_paths=bot_paths,
    )


def battle_bots_many(
    bot0_path: str | Path,
    bot1_path: str | Path,
    *,
    games: int = 100,
    bot0_seat: int = 0,
    seed: int | None = None,
    alternate_seats: bool = True,
    keep_logs: bool = False,
    turn_limit: int = DEFAULT_TURN_LIMIT,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    """Run many games between two arbitrary bot files."""

    if games < 1:
        raise ValueError("games must be >= 1")
    if bot0_seat not in (0, 1):
        raise ValueError("bot0_seat must be 0 or 1")
    validate_decision_timeout(decision_timeout)

    wins_by_bot = [0, 0]
    wins_by_seat = [0, 0]
    statuses: Counter[str] = Counter()
    game_ids: list[str] = []

    for index in range(games):
        game_bot0_seat = (bot0_seat + index) % 2 if alternate_seats else bot0_seat
        game_seed = None if seed is None else seed + index
        result = battle_bots_once(
            bot0_path,
            bot1_path,
            bot0_seat=game_bot0_seat,
            seed=game_seed,
            keep_log=keep_logs,
            turn_limit=turn_limit,
            decision_timeout=decision_timeout,
        )
        statuses[result["status"]] += 1
        if result["winner"] in (0, 1):
            wins_by_seat[result["winner"]] += 1
        if result["winner_bot"] in (0, 1):
            wins_by_bot[result["winner_bot"]] += 1
        if keep_logs:
            game_ids.append(result["game_id"])

    summary: dict[str, Any] = {
        "games": games,
        "players": SUPPORTED_PLAYERS,
        "bot_paths": [str(Path(bot0_path)), str(Path(bot1_path))],
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
    player_id = state["player_id"]
    row = state["positions"][player_id][0]
    goal = state["goal_rows"][player_id]

    forward = "MOVE_UP" if goal < row else "MOVE_DOWN"
    if forward in legal:
        return forward

    moves = [action for action in legal if action.startswith("MOVE_")]
    if moves:
        return moves[0]

    return legal[0]
'''
    Path(path).write_text(sample, encoding="utf-8")


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Luqiangqi two-player bot battle referee")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample-bot", help="write a sample bot.py")
    sample_parser.add_argument("--output", default="bot.py")

    battle_parser = subparsers.add_parser("battle", help="run one or many games")
    battle_parser.add_argument("--bot", required=True)
    battle_parser.add_argument("--players", type=int, default=SUPPORTED_PLAYERS)
    battle_parser.add_argument("--games", type=int, default=1)
    battle_parser.add_argument("--seat", type=int, default=0)
    battle_parser.add_argument("--seed", type=int, default=None)
    battle_parser.add_argument("--keep-logs", action="store_true")
    battle_parser.add_argument("--fixed-seat", action="store_true")
    battle_parser.add_argument("--turn-limit", type=int, default=DEFAULT_TURN_LIMIT)
    battle_parser.add_argument("--decision-timeout", type=float, default=None)

    duel_parser = subparsers.add_parser("duel", help="run games between two bot files")
    duel_parser.add_argument("--bot0", required=True)
    duel_parser.add_argument("--bot1", required=True)
    duel_parser.add_argument("--games", type=int, default=1)
    duel_parser.add_argument("--seed", type=int, default=None)
    duel_parser.add_argument("--keep-logs", action="store_true")
    duel_parser.add_argument("--fixed-seats", action="store_true")
    duel_parser.add_argument("--bot0-seat", type=int, default=0)
    duel_parser.add_argument("--turn-limit", type=int, default=DEFAULT_TURN_LIMIT)
    duel_parser.add_argument("--decision-timeout", type=float, default=None)

    args = parser.parse_args(argv)

    try:
        if args.command == "sample-bot":
            write_sample_bot(args.output)
            print(f"wrote {args.output}")
            return 0

        if args.command == "battle" and args.games == 1:
            _print_json(
                battle_once(
                    args.bot,
                    players=args.players,
                    seat=args.seat,
                    seed=args.seed,
                    keep_log=args.keep_logs,
                    turn_limit=args.turn_limit,
                    decision_timeout=args.decision_timeout,
                )
            )
        elif args.command == "battle":
            _print_json(
                battle_many(
                    args.bot,
                    games=args.games,
                    players=args.players,
                    seat=args.seat,
                    seed=args.seed,
                    alternate_seats=not args.fixed_seat,
                    keep_logs=args.keep_logs,
                    turn_limit=args.turn_limit,
                    decision_timeout=args.decision_timeout,
                )
            )
        elif args.command == "duel" and args.games == 1:
            _print_json(
                battle_bots_once(
                    args.bot0,
                    args.bot1,
                    bot0_seat=args.bot0_seat,
                    seed=args.seed,
                    keep_log=args.keep_logs,
                    turn_limit=args.turn_limit,
                    decision_timeout=args.decision_timeout,
                )
            )
        elif args.command == "duel":
            _print_json(
                battle_bots_many(
                    args.bot0,
                    args.bot1,
                    games=args.games,
                    bot0_seat=args.bot0_seat,
                    seed=args.seed,
                    alternate_seats=not args.fixed_seats,
                    keep_logs=args.keep_logs,
                    turn_limit=args.turn_limit,
                    decision_timeout=args.decision_timeout,
                )
            )
        return 0
    except Exception:  # noqa: BLE001 - command line should show the failure.
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
