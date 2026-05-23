#!/usr/bin/env python3
"""Simplified 6 nimmt! referee and bot battle API.

Rules in this module are intentionally small:
- 2-6 players.
- The deck is exactly 1..10N for N players.
- Every player gets 10 cards, all cards are played in one round.
- The winner is the player or players with the fewest bull heads.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


MIN_PLAYERS = 2
MAX_PLAYERS = 6
CARDS_PER_PLAYER = 10
ROW_COUNT = 4
ROW_LIMIT = 5

_MATCH_LOGS: dict[str, list[str]] = {}


class IllegalActionError(ValueError):
    """Raised when one or more simultaneous actions are illegal."""


class BotTimeoutError(TimeoutError):
    """Raised when a bot does not return an action before the deadline."""


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: list[float]
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def as_tuple(self) -> tuple[dict[str, Any], list[float], bool, bool, dict[str, Any]]:
        return self.observation, self.reward, self.terminated, self.truncated, self.info


def bull_count(card: int) -> int:
    """Return original-style bull-head value for a card."""
    if card == 55:
        return 7
    if card % 11 == 0:
        return 5
    if card % 10 == 0:
        return 3
    if card % 5 == 0:
        return 2
    return 1


def row_bulls(row: list[int]) -> int:
    return sum(bull_count(card) for card in row)


def play_action(card: int, take_row: int | None = None) -> str:
    if take_row is None:
        return f"PLAY_{card}"
    return f"PLAY_{card}_TAKE_{take_row}"


def parse_play_action(action: str) -> tuple[int, int | None] | None:
    parts = action.split("_")
    if len(parts) == 2 and parts[0] == "PLAY":
        try:
            return int(parts[1]), None
        except ValueError:
            return None
    if len(parts) == 4 and parts[0] == "PLAY" and parts[2] == "TAKE":
        try:
            return int(parts[1]), int(parts[3])
        except ValueError:
            return None
    return None


@dataclass
class GameState:
    hands: list[list[int]]
    rows: list[list[int]] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)
    turn: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def new(cls, players: int, rng: random.Random) -> "GameState":
        if not MIN_PLAYERS <= players <= MAX_PLAYERS:
            raise ValueError(f"players must be between {MIN_PLAYERS} and {MAX_PLAYERS}")
        deck = list(range(1, players * CARDS_PER_PLAYER + 1))
        rng.shuffle(deck)
        hands = [
            sorted(deck[index * CARDS_PER_PLAYER : (index + 1) * CARDS_PER_PLAYER])
            for index in range(players)
        ]
        return cls(hands=hands, scores=[0 for _ in range(players)])

    @property
    def num_players(self) -> int:
        return len(self.hands)

    def legal_actions(self, player_id: int) -> list[str]:
        hand = self.hands[player_id]
        if not hand:
            return []

        if len(self.rows) < ROW_COUNT:
            return [play_action(card) for card in hand]

        min_top = min(row[-1] for row in self.rows)
        actions: list[str] = []
        for card in hand:
            if card < min_top:
                actions.extend(play_action(card, row_index) for row_index in range(len(self.rows)))
            else:
                actions.append(play_action(card))
        return actions

    def bot_state(self, player_id: int) -> dict[str, Any]:
        return {
            "player_id": player_id,
            "num_players": self.num_players,
            "phase": "play",
            "actor": player_id,
            "legal_actions": self.legal_actions(player_id),
            "hand": list(self.hands[player_id]),
            "hand_sizes": [len(hand) for hand in self.hands],
            "rows": [list(row) for row in self.rows],
            "row_bulls": [row_bulls(row) for row in self.rows],
            "scores": list(self.scores),
            "turn": self.turn,
            "cards_per_player": CARDS_PER_PLAYER,
            "deck_max": self.num_players * CARDS_PER_PLAYER,
            "bull_values": {card: bull_count(card) for card in self.hands[player_id]},
            "history": list(self.history),
        }

    def apply_revealed_cards(self, revealed: list[tuple[int, int, int | None]]) -> list[str]:
        events: list[str] = []
        for player, card, _ in revealed:
            self.hands[player].remove(card)
            events.append(f"T{self.turn}:P{player}:PLAY:{card}")

        for player, card, take_row in sorted(revealed, key=lambda item: item[1]):
            if len(self.rows) < ROW_COUNT:
                self.rows.append([card])
                row_index = len(self.rows) - 1
                events.append(f"T{self.turn}:P{player}:OPEN:R{row_index}:{card}")
                continue

            candidates = [
                (row[-1], row_index)
                for row_index, row in enumerate(self.rows)
                if row[-1] < card
            ]
            if not candidates:
                if take_row is None or not 0 <= take_row < len(self.rows):
                    raise ValueError(f"card {card} needs TAKE row, got {take_row}")
                taken = row_bulls(self.rows[take_row])
                old_row = list(self.rows[take_row])
                self.scores[player] += taken
                self.rows[take_row] = [card]
                events.append(
                    f"T{self.turn}:P{player}:TAKE:R{take_row}:B{taken}:"
                    f"{','.join(map(str, old_row))}>{card}"
                )
                continue

            _, row_index = max(candidates)
            if len(self.rows[row_index]) >= ROW_LIMIT:
                taken = row_bulls(self.rows[row_index])
                old_row = list(self.rows[row_index])
                self.scores[player] += taken
                self.rows[row_index] = [card]
                events.append(
                    f"T{self.turn}:P{player}:SIXTH:R{row_index}:B{taken}:"
                    f"{','.join(map(str, old_row))}>{card}"
                )
            else:
                self.rows[row_index].append(card)
                events.append(f"T{self.turn}:P{player}:PLACE:R{row_index}:{card}")

        self.history.append(
            {
                "turn": self.turn,
                "revealed": [
                    {"player": player, "card": card, "take_row": take_row}
                    for player, card, take_row in sorted(revealed, key=lambda item: item[1])
                ],
                "scores": list(self.scores),
                "rows": [list(row) for row in self.rows],
            }
        )
        self.turn += 1
        return events

    def finished(self) -> bool:
        return all(not hand for hand in self.hands)

    def winners(self) -> list[int]:
        best = min(self.scores)
        return [player for player, score in enumerate(self.scores) if score == best]


class NimmtEnv:
    """A simultaneous-turn Gym-style environment for simplified 6 nimmt!."""

    metadata = {"render_modes": ["ansi"], "players": f"{MIN_PLAYERS}-{MAX_PLAYERS}"}

    def __init__(self, *, players: int = 2, seed: int | None = None) -> None:
        if not MIN_PLAYERS <= players <= MAX_PLAYERS:
            raise ValueError(f"players must be between {MIN_PLAYERS} and {MAX_PLAYERS}")
        self.players = players
        self.rng = random.Random(seed)
        self.game = GameState.new(players, self.rng)

    def reset(
        self,
        *,
        seed: int | None = None,
        players: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if players is not None:
            if not MIN_PLAYERS <= players <= MAX_PLAYERS:
                raise ValueError(f"players must be between {MIN_PLAYERS} and {MAX_PLAYERS}")
            self.players = players
        if seed is not None:
            self.rng.seed(seed)
        self.game = GameState.new(self.players, self.rng)
        return self.state(), {"seed": seed, "players": self.players}

    def step(
        self,
        actions: list[str] | dict[int, str],
    ) -> tuple[dict[str, Any], list[float], bool, bool, dict[str, Any]]:
        if self.game.finished():
            return self.state(), self._rewards(), True, False, self._info()

        action_list = self._normalize_actions(actions)
        revealed: list[tuple[int, int, int | None]] = []
        for player, action in enumerate(action_list):
            legal = self.game.legal_actions(player)
            if action not in legal:
                raise IllegalActionError(f"player {player} action {action!r} is not legal")
            parsed = parse_play_action(action)
            if parsed is None:
                raise IllegalActionError(f"player {player} action {action!r} cannot be parsed")
            card, take_row = parsed
            revealed.append((player, card, take_row))

        events = self.game.apply_revealed_cards(revealed)
        terminated = self.game.finished()
        info = self._info(events=events) if terminated else {"events": events}
        return self.state(), self._rewards() if terminated else [0.0 for _ in range(self.players)], terminated, False, info

    def legal_actions(self, player_id: int) -> list[str]:
        if self.game.finished():
            return []
        return self.game.legal_actions(player_id)

    def state(self, player_id: int | None = None) -> dict[str, Any]:
        if player_id is not None:
            state = self.game.bot_state(player_id)
            if self.game.finished():
                state["phase"] = "game_over"
                state["legal_actions"] = []
                state["winners"] = self.game.winners()
            return state

        return {
            "num_players": self.players,
            "phase": "game_over" if self.game.finished() else "play",
            "turn": self.game.turn,
            "hands": [list(hand) for hand in self.game.hands],
            "hand_sizes": [len(hand) for hand in self.game.hands],
            "rows": [list(row) for row in self.game.rows],
            "row_bulls": [row_bulls(row) for row in self.game.rows],
            "scores": list(self.game.scores),
            "winners": self.game.winners() if self.game.finished() else [],
            "history": list(self.game.history),
        }

    def render(self) -> str:
        rows = "\n".join(
            f"R{index}: {' '.join(map(str, row))} ({row_bulls(row)})"
            for index, row in enumerate(self.game.rows)
        )
        scores = " ".join(f"P{index}={score}" for index, score in enumerate(self.game.scores))
        return f"Turn {self.game.turn}\n{rows}\nScores: {scores}"

    def _normalize_actions(self, actions: list[str] | dict[int, str]) -> list[str]:
        if isinstance(actions, dict):
            missing = [player for player in range(self.players) if player not in actions]
            if missing:
                raise IllegalActionError(f"missing actions for players: {missing}")
            return [actions[player] for player in range(self.players)]
        if len(actions) != self.players:
            raise IllegalActionError(f"expected {self.players} actions, got {len(actions)}")
        return list(actions)

    def _rewards(self) -> list[float]:
        if not self.game.finished():
            return [0.0 for _ in range(self.players)]
        winners = set(self.game.winners())
        return [1.0 if player in winners else -1.0 for player in range(self.players)]

    def _info(self, *, events: list[str] | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "status": "ok" if self.game.finished() else None,
            "winners": self.game.winners() if self.game.finished() else [],
            "scores": list(self.game.scores),
            "rows": [list(row) for row in self.game.rows],
            "turns": self.game.turn,
        }
        if events is not None:
            info["events"] = events
        return info


class BotLike(Protocol):
    name: str

    def choose_action(self, state: dict[str, Any]) -> str:
        ...


class SystemBot:
    def __init__(self, name: str = "system", seed: int = 0) -> None:
        self.name = name
        self.rng = random.Random(seed)

    def choose_action(self, state: dict[str, Any]) -> str:
        legal = list(state["legal_actions"])
        if not legal:
            raise ValueError("system bot received no legal actions")
        return self.rng.choice(legal)


class CallableBot:
    def __init__(self, choose_action: Callable[[dict[str, Any]], str], name: str) -> None:
        self.choose_action = choose_action
        self.name = name


def choose_action_with_timeout(
    bot: BotLike,
    state: dict[str, Any],
    decision_timeout: float | None,
) -> str:
    state = dict(state)
    if state.get("decision_timeout") is None:
        state["decision_timeout"] = decision_timeout
    if state.get("time_limit") is None:
        state["time_limit"] = state.get("decision_timeout")
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


def load_bot(bot_path: str | Path) -> BotLike:
    path = Path(bot_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"bot file not found: {path}")

    module_name = f"nimmt_user_bot_{uuid.uuid4().hex}"
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


def run_match(
    bots: list[BotLike],
    seed: int | None = None,
    keep_log: bool = True,
    developer_seat: int | None = None,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if not MIN_PLAYERS <= len(bots) <= MAX_PLAYERS:
        raise ValueError(f"bots length must be between {MIN_PLAYERS} and {MAX_PLAYERS}")
    validate_decision_timeout(decision_timeout)

    game_seed = seed if seed is not None else random.randrange(1 << 30)
    rng = random.Random(game_seed)
    game_id = uuid.uuid4().hex[:12]
    state = GameState.new(len(bots), rng)
    bot_names = [getattr(bot, "name", f"bot_{index}") for index, bot in enumerate(bots)]
    log = [f"G:{game_id}:N{len(bots)}:SEED{game_seed}"]
    log.extend(
        f"H:P{player}:{','.join(map(str, hand))}"
        for player, hand in enumerate(state.hands)
    )

    status = "ok"
    error: str | None = None

    while not state.finished():
        revealed: list[tuple[int, int, int | None]] = []
        for player, bot in enumerate(bots):
            public = state.bot_state(player)
            legal = public["legal_actions"]
            if not legal:
                status = "no_legal_actions"
                error = f"player {player} has no legal actions"
                log.append(f"T{state.turn}:ERR:P{player}:NO_LEGAL_ACTIONS")
                return _result(game_id, state, bot_names, developer_seat, status, error, keep_log, log)

            try:
                action = choose_action_with_timeout(bot, public, decision_timeout)
            except BotTimeoutError as exc:
                status = "timeout"
                error = f"player {player} timed out: {exc}"
                log.append(f"T{state.turn}:ERR:P{player}:TIMEOUT")
                return _forfeit_result(
                    game_id, state, bot_names, player, developer_seat, status, error, keep_log, log
                )
            except Exception as exc:  # noqa: BLE001 - bot failures are match results.
                status = "bot_exception"
                error = f"{type(exc).__name__}: {exc}"
                log.append(f"T{state.turn}:ERR:P{player}:EXCEPTION:{type(exc).__name__}")
                return _forfeit_result(
                    game_id, state, bot_names, player, developer_seat, status, error, keep_log, log
                )

            if not isinstance(action, str) or action not in legal:
                status = "invalid_action"
                error = f"invalid action from player {player}: {action!r}"
                log.append(f"T{state.turn}:ERR:P{player}:INVALID:{action}")
                return _forfeit_result(
                    game_id, state, bot_names, player, developer_seat, status, error, keep_log, log
                )

            parsed = parse_play_action(action)
            if parsed is None:
                status = "invalid_action"
                error = f"cannot parse action from player {player}: {action!r}"
                log.append(f"T{state.turn}:ERR:P{player}:INVALID:{action}")
                return _forfeit_result(
                    game_id, state, bot_names, player, developer_seat, status, error, keep_log, log
                )
            card, take_row = parsed
            revealed.append((player, card, take_row))

        log.extend(state.apply_revealed_cards(revealed))

    return _result(game_id, state, bot_names, developer_seat, status, error, keep_log, log)


def battle_once(
    bot_path: str | Path,
    players: int = 2,
    seat: int = 0,
    seed: int | None = None,
    keep_log: bool = True,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if not MIN_PLAYERS <= players <= MAX_PLAYERS:
        raise ValueError(f"players must be between {MIN_PLAYERS} and {MAX_PLAYERS}")
    if not 0 <= seat < players:
        raise ValueError("seat must be within player range")
    validate_decision_timeout(decision_timeout)

    developer_bot = load_bot(bot_path)
    game_seed = seed if seed is not None else random.randrange(1 << 30)
    bots: list[BotLike] = [
        SystemBot(name=f"system_{player}", seed=game_seed + player)
        for player in range(players)
    ]
    bots[seat] = developer_bot
    return run_match(
        bots,
        seed=game_seed,
        keep_log=keep_log,
        developer_seat=seat,
        decision_timeout=decision_timeout,
    )


def battle_many(
    bot_path: str | Path,
    games: int = 100,
    players: int = 2,
    seed: int | None = None,
    alternate_seats: bool = True,
    keep_logs: bool = False,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if games < 1:
        raise ValueError("games must be >= 1")
    if not MIN_PLAYERS <= players <= MAX_PLAYERS:
        raise ValueError(f"players must be between {MIN_PLAYERS} and {MAX_PLAYERS}")
    validate_decision_timeout(decision_timeout)

    wins_by_seat = [0 for _ in range(players)]
    developer_wins = 0
    statuses: Counter[str] = Counter()
    ties = 0
    game_ids: list[str] = []

    for index in range(games):
        game_seat = index % players if alternate_seats else 0
        game_seed = None if seed is None else seed + index
        developer_bot = load_bot(bot_path)
        bots: list[BotLike] = [
            SystemBot(name=f"system_{player}", seed=(game_seed or 0) + player)
            for player in range(players)
        ]
        bots[game_seat] = developer_bot
        result = run_match(
            bots,
            seed=game_seed,
            keep_log=keep_logs,
            developer_seat=game_seat,
            decision_timeout=decision_timeout,
        )
        statuses[result["status"]] += 1
        for winner in result["winners"]:
            wins_by_seat[winner] += 1
        if len(result["winners"]) > 1:
            ties += 1
        if result["developer_win"]:
            developer_wins += 1
        if keep_logs:
            game_ids.append(result["game_id"])

    summary: dict[str, Any] = {
        "games": games,
        "players": players,
        "wins_by_seat": wins_by_seat,
        "ties": ties,
        "developer_wins": developer_wins,
        "developer_losses": games - developer_wins,
        "developer_win_rate": developer_wins / games,
        "statuses": dict(statuses),
    }
    if keep_logs:
        summary["game_ids"] = game_ids
    return summary


def get_match_log(game_id: str) -> list[str]:
    if game_id not in _MATCH_LOGS:
        raise KeyError(f"unknown game_id: {game_id}")
    return list(_MATCH_LOGS[game_id])


def write_sample_bot(path: str | Path = "bot_random.py") -> None:
    Path(path).write_text(SAMPLE_BOT.lstrip(), encoding="utf-8")


def _result(
    game_id: str,
    state: GameState,
    bot_names: list[str],
    developer_seat: int | None,
    status: str,
    error: str | None,
    keep_log: bool,
    log: list[str],
) -> dict[str, Any]:
    winners = state.winners()
    log.append(
        f"END:{status}:WINNERS:{','.join(map(str, winners))}:"
        f"SCORES:{','.join(map(str, state.scores))}:ROWS:{state.rows}"
    )
    if keep_log:
        _MATCH_LOGS[game_id] = log
    return {
        "game_id": game_id,
        "winner": winners[0] if winners else None,
        "winners": winners,
        "status": status,
        "turns": state.turn,
        "scores": list(state.scores),
        "rows": [list(row) for row in state.rows],
        "bot_names": bot_names,
        "developer_seat": developer_seat,
        "developer_win": (
            None if developer_seat is None else developer_seat in winners
        ),
        "error": error,
    }


def _forfeit_result(
    game_id: str,
    state: GameState,
    bot_names: list[str],
    forfeiter: int,
    developer_seat: int | None,
    status: str,
    error: str | None,
    keep_log: bool,
    log: list[str],
) -> dict[str, Any]:
    candidates = [player for player in range(state.num_players) if player != forfeiter]
    best_score = min(state.scores[player] for player in candidates)
    winners = [player for player in candidates if state.scores[player] == best_score]
    log.append(
        f"END:{status}:FORFEIT:{forfeiter}:WINNERS:{','.join(map(str, winners))}:"
        f"SCORES:{','.join(map(str, state.scores))}"
    )
    if keep_log:
        _MATCH_LOGS[game_id] = log
    return {
        "game_id": game_id,
        "winner": winners[0] if winners else None,
        "winners": winners,
        "status": status,
        "turns": state.turn,
        "scores": list(state.scores),
        "rows": [list(row) for row in state.rows],
        "bot_names": bot_names,
        "developer_seat": developer_seat,
        "developer_win": (
            None if developer_seat is None else developer_seat in winners
        ),
        "error": error,
    }


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simplified 6 nimmt! bot battle referee")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample-bot", help="write a random sample bot")
    sample_parser.add_argument("--output", default="bot_random.py")

    battle_parser = subparsers.add_parser("battle", help="run one or many games")
    battle_parser.add_argument("--bot", required=True)
    battle_parser.add_argument("--players", type=int, default=2)
    battle_parser.add_argument("--games", type=int, default=1)
    battle_parser.add_argument("--seat", type=int, default=0)
    battle_parser.add_argument("--seed", type=int, default=None)
    battle_parser.add_argument("--keep-logs", action="store_true")
    battle_parser.add_argument("--fixed-seat", action="store_true")
    battle_parser.add_argument("--decision-timeout", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
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
                        decision_timeout=args.decision_timeout,
                    )
                )
            else:
                _print_json(
                    battle_many(
                        args.bot,
                        games=args.games,
                        players=args.players,
                        seed=args.seed,
                        alternate_seats=not args.fixed_seat,
                        keep_logs=args.keep_logs,
                        decision_timeout=args.decision_timeout,
                    )
                )
            return 0
    except Exception:  # noqa: BLE001 - command line should show the failure.
        traceback.print_exc()
        return 1

    parser.error("unknown command")
    return 2


SAMPLE_BOT = r'''
import random


class Bot:
    name = "random_nimmt"

    def choose_action(self, state):
        return random.choice(state["legal_actions"])
'''


if __name__ == "__main__":
    raise SystemExit(main())
