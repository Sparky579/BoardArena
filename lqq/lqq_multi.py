"""Two-player referee and bot battle API for Luqiangqi.

The public API mirrors the simple battle interface used by other games in this
workspace: battle_once, battle_many, and get_match_log.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
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
}

_MATCH_LOGS: dict[str, list[str]] = {}


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
        opponent = self.players[1 - player_id]

        for action, (dr, dc) in MOVE_DELTAS.items():
            row = player.row + dr
            col = player.col + dc
            if not self._in_bounds(row, col):
                continue
            if row == opponent.row and col == opponent.col:
                continue
            if not self._has_wall_between(player.row, player.col, row, col):
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
            dr, dc = MOVE_DELTAS[action]
            player.row += dr
            player.col += dc
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
        for dr, dc in MOVE_DELTAS.values():
            next_row = row + dr
            next_col = col + dc
            if self._in_bounds(next_row, next_col) and not self._has_wall_between(row, col, next_row, next_col):
                result.append((next_row, next_col))
        return result

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


def battle_once(
    bot_path: str | Path,
    players: int = SUPPORTED_PLAYERS,
    seat: int = 0,
    seed: int | None = None,
    keep_log: bool = True,
    turn_limit: int = DEFAULT_TURN_LIMIT,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("lqq_multi only supports players=2")
    if seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")

    game_seed = seed if seed is not None else random.randrange(1 << 30)
    rng = random.Random(game_seed)
    game_id = uuid.uuid4().hex[:12]
    log: list[str] = [f"G:{game_id}:N2:SEED{game_seed}"]

    developer_bot = load_bot(bot_path)
    bots: list[Any] = [SystemBot(rng), SystemBot(rng)]
    bots[seat] = developer_bot
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
            action = bots[actor].choose_action(bot_state)
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

    developer_win = state.winner == seat
    return {
        "game_id": game_id,
        "winner": state.winner,
        "status": status,
        "turns": state.turn,
        "positions": [[p.row, p.col] for p in state.players],
        "walls_remaining": [p.walls for p in state.players],
        "bot_names": bot_names,
        "developer_seat": seat,
        "developer_win": developer_win,
        "error": error,
    }


def battle_many(
    bot_path: str | Path,
    games: int = 100,
    players: int = SUPPORTED_PLAYERS,
    seat: int = 0,
    seed: int | None = None,
    alternate_seats: bool = True,
    keep_logs: bool = False,
    turn_limit: int = DEFAULT_TURN_LIMIT,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("lqq_multi only supports players=2")
    if games < 1:
        raise ValueError("games must be >= 1")

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
                    turn_limit=args.turn_limit,
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
                    turn_limit=args.turn_limit,
                )
            )
        return 0
    except Exception:  # noqa: BLE001 - command line should show the failure.
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
