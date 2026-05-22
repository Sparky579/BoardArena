#!/usr/bin/env python3
"""Gym-style two-player UNO environment and bot battle API."""

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


SUPPORTED_PLAYERS = 2
HAND_SIZE = 7
DEFAULT_MAX_PLIES = 500

COLORS = ("red", "yellow", "green", "blue")
COLOR_SHORT = {"red": "R", "yellow": "Y", "green": "G", "blue": "B"}
ACTION_KINDS = {"skip", "reverse", "draw_two"}
WILD_KINDS = {"wild", "wild_draw_four"}
DRAW_ACTION = "draw"
PASS_ACTION = "pass"

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


def opponent(player: int) -> int:
    return 1 - player


def build_deck() -> list[dict[str, Any]]:
    deck: list[dict[str, Any]] = []
    card_id = 0

    def add(color: str | None, kind: str, value: int | None = None, copies: int = 1) -> None:
        nonlocal card_id
        for _ in range(copies):
            deck.append({"id": card_id, "color": color, "kind": kind, "value": value})
            card_id += 1

    for color in COLORS:
        add(color, "number", 0, copies=1)
        for value in range(1, 10):
            add(color, "number", value, copies=2)
        for kind in ("skip", "reverse", "draw_two"):
            add(color, kind, copies=2)
    add(None, "wild", copies=4)
    add(None, "wild_draw_four", copies=4)
    return deck


def card_label(card: dict[str, Any]) -> str:
    kind = card["kind"]
    if kind == "wild":
        return "W"
    if kind == "wild_draw_four":
        return "W+4"
    prefix = COLOR_SHORT[str(card["color"])]
    if kind == "number":
        return f"{prefix}{card['value']}"
    if kind == "skip":
        return f"{prefix}S"
    if kind == "reverse":
        return f"{prefix}R"
    if kind == "draw_two":
        return f"{prefix}+2"
    return f"{prefix}?"


def card_view(card: dict[str, Any], *, current_color: str | None = None) -> dict[str, Any]:
    return {
        "id": card["id"],
        "color": card["color"],
        "kind": card["kind"],
        "value": card["value"],
        "label": card_label(card),
        "current_color": current_color if card["kind"] in WILD_KINDS else card["color"],
    }


def _card_sort_key(card: dict[str, Any]) -> tuple[int, int, int]:
    color_order = {color: index for index, color in enumerate(COLORS)}
    kind_order = {
        "number": 0,
        "skip": 10,
        "reverse": 11,
        "draw_two": 12,
        "wild": 20,
        "wild_draw_four": 21,
    }
    return (
        color_order.get(card["color"], 9),
        int(card["value"] if card["value"] is not None else kind_order[card["kind"]]),
        int(card["id"]),
    )


def is_playable(card: dict[str, Any], top_card: dict[str, Any], current_color: str) -> bool:
    if card["kind"] in WILD_KINDS:
        return True
    if card["color"] == current_color:
        return True
    if card["kind"] == "number" and top_card["kind"] == "number":
        return card["value"] == top_card["value"]
    return card["kind"] == top_card["kind"] and card["kind"] in ACTION_KINDS


class UnoEnv:
    """A small alternating-turn Gym-style environment for two-player UNO."""

    metadata = {"render_modes": ["ansi"], "players": SUPPORTED_PLAYERS}

    def __init__(
        self,
        *,
        seed: int | None = None,
        max_plies: int | None = DEFAULT_MAX_PLIES,
        initial_hands: list[list[dict[str, Any]]] | None = None,
        draw_pile: list[dict[str, Any]] | None = None,
        discard_pile: list[dict[str, Any]] | None = None,
        actor: int = 0,
        current_color: str | None = None,
    ) -> None:
        if actor not in (0, 1):
            raise ValueError("actor must be 0 or 1")
        self.max_plies = max_plies
        self.rng = random.Random(seed)
        self.initial_config = {
            "hands": initial_hands,
            "draw_pile": draw_pile,
            "discard_pile": discard_pile,
            "actor": actor,
            "current_color": current_color,
        }
        self.hands: list[list[dict[str, Any]]] = [[], []]
        self.draw_pile: list[dict[str, Any]] = []
        self.discard_pile: list[dict[str, Any]] = []
        self.actor = actor
        self.current_color: str = "red"
        self.plies = 0
        self.last_action: str | None = None
        self.last_draw_count = 0
        self.consecutive_passes = 0
        self.winner: int | None = None
        self.terminal_status: str | None = None
        self.history: list[dict[str, Any]] = []
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        actor: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        if actor is not None and actor not in (0, 1):
            raise ValueError("actor must be 0 or 1")

        config = self.initial_config
        if config["hands"] is not None:
            self.hands = [[dict(card) for card in hand] for hand in config["hands"]]
            self.draw_pile = [dict(card) for card in (config["draw_pile"] or [])]
            self.discard_pile = [dict(card) for card in (config["discard_pile"] or [])]
            if not self.discard_pile:
                raise ValueError("discard_pile is required with custom hands")
            self.actor = config["actor"] if actor is None else actor
            self.current_color = config["current_color"] or self.discard_pile[-1]["color"]
            if self.current_color not in COLORS:
                raise ValueError("current_color must be one of red/yellow/green/blue")
        else:
            self._deal_new_round(actor=0 if actor is None else actor)

        self.plies = 0
        self.last_action = None
        self.last_draw_count = 0
        self.consecutive_passes = 0
        self.winner = None
        self.terminal_status = None
        self.history = []
        return self.state(), {
            "actor": self.actor,
            "top_card": card_view(self.discard_pile[-1], current_color=self.current_color),
            "current_color": self.current_color,
        }

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.is_done():
            return self.state(), 0.0, True, False, self._outcome_info()

        legal = self.legal_actions()
        if action not in legal:
            raise IllegalActionError(f"{action!r} is not legal in the current position")

        actor = self.actor
        draw_count = 0
        public_action = action

        if action == DRAW_ACTION:
            draw_count = self._draw_cards(actor, 1)
            self.last_draw_count = draw_count
            self.consecutive_passes = 0 if draw_count else self.consecutive_passes + 1
            self.actor = opponent(actor)
        elif action == PASS_ACTION:
            self.last_draw_count = 0
            self.consecutive_passes += 1
            self.actor = opponent(actor)
        else:
            card, chosen_color = self._parse_play_action(action)
            self.hands[actor].remove(card)
            self.discard_pile.append(card)
            self.current_color = chosen_color or str(card["color"])
            self.last_draw_count = 0
            self.consecutive_passes = 0
            public_action = self._public_play_action(card, chosen_color)
            self._apply_card_effect(card, actor)

            if not self.hands[actor]:
                self.winner = actor
                self.terminal_status = "empty_hand"

        self.plies += 1
        self.last_action = public_action
        self.history.append(
            {
                "ply": self.plies,
                "player": actor,
                "action": public_action,
                "draw_count": draw_count,
                "hand_counts": [len(hand) for hand in self.hands],
            }
        )

        if self.terminal_status is None and self.consecutive_passes >= SUPPORTED_PLAYERS:
            self.winner = self._leader_by_hand_size()
            self.terminal_status = "blocked"

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
        actions: list[str] = []
        top_card = self.discard_pile[-1]
        for card in sorted(self.hands[self.actor], key=_card_sort_key):
            if not is_playable(card, top_card, self.current_color):
                continue
            if card["kind"] in WILD_KINDS:
                actions.extend(f"play:{card['id']}:{color}" for color in COLORS)
            else:
                actions.append(f"play:{card['id']}")
        if actions:
            return actions
        if self._can_draw():
            return [DRAW_ACTION]
        return [PASS_ACTION]

    def is_done(self) -> bool:
        return self.terminal_status is not None

    def state(self, player_id: int | None = None) -> dict[str, Any]:
        viewer = self.actor if player_id is None else player_id
        if viewer not in (0, 1):
            raise ValueError("player_id must be 0 or 1")

        done = self.is_done()
        legal = [] if done else (self.legal_actions() if viewer == self.actor else [])
        hand = [card_view(card) for card in sorted(self.hands[viewer], key=_card_sort_key)]
        legal_set = set(legal)
        for item in hand:
            item["legal_actions"] = [action for action in legal if action.startswith(f"play:{item['id']}")]

        return {
            "player_id": viewer,
            "num_players": SUPPORTED_PLAYERS,
            "phase": "game_over" if done else "turn",
            "actor": self.actor,
            "turn": f"player_{self.actor}",
            "legal_actions": legal,
            "hand": hand,
            "hand_count": len(self.hands[viewer]),
            "hand_counts": [len(hand_cards) for hand_cards in self.hands],
            "opponent_hand_count": len(self.hands[opponent(viewer)]),
            "top_card": card_view(self.discard_pile[-1], current_color=self.current_color),
            "current_color": self.current_color,
            "draw_pile_count": len(self.draw_pile),
            "discard_pile_count": len(self.discard_pile),
            "can_draw": DRAW_ACTION in legal_set,
            "can_pass": PASS_ACTION in legal_set,
            "plies": self.plies,
            "last_action": self.last_action,
            "last_draw_count": self.last_draw_count,
            "history": list(self.history),
            "winner": self.winner,
            "status": self.terminal_status,
            "result": _result_for_winner(self.winner) if done else "*",
        }

    def render(self) -> str:
        lines = [
            f"top={card_label(self.discard_pile[-1])} color={self.current_color} actor=P{self.actor}",
            f"P0({len(self.hands[0])}): {' '.join(card_label(card) for card in sorted(self.hands[0], key=_card_sort_key))}",
            f"P1({len(self.hands[1])}): {' '.join(card_label(card) for card in sorted(self.hands[1], key=_card_sort_key))}",
            f"draw={len(self.draw_pile)} discard={len(self.discard_pile)} status={self.terminal_status}",
        ]
        return "\n".join(lines)

    def _deal_new_round(self, *, actor: int) -> None:
        deck = build_deck()
        self.rng.shuffle(deck)
        self.hands = [[], []]
        for _ in range(HAND_SIZE):
            self.hands[0].append(deck.pop())
            self.hands[1].append(deck.pop())
        initial_index = next(
            index
            for index, card in enumerate(deck)
            if card["kind"] == "number" and card["color"] in COLORS
        )
        initial_card = deck.pop(initial_index)
        self.draw_pile = deck
        self.discard_pile = [initial_card]
        self.current_color = str(initial_card["color"])
        self.actor = actor

    def _parse_play_action(self, action: str) -> tuple[dict[str, Any], str | None]:
        parts = action.split(":")
        if len(parts) not in (2, 3) or parts[0] != "play":
            raise IllegalActionError(f"invalid play action format: {action!r}")
        try:
            card_id = int(parts[1])
        except ValueError as exc:
            raise IllegalActionError(f"invalid card id in action: {action!r}") from exc
        card = next((item for item in self.hands[self.actor] if item["id"] == card_id), None)
        if card is None:
            raise IllegalActionError(f"card {card_id} is not in actor hand")
        chosen_color = parts[2] if len(parts) == 3 else None
        if card["kind"] in WILD_KINDS:
            if chosen_color not in COLORS:
                raise IllegalActionError("wild cards require a chosen color")
        elif chosen_color is not None:
            raise IllegalActionError("non-wild cards cannot choose color")
        return card, chosen_color

    def _public_play_action(self, card: dict[str, Any], chosen_color: str | None) -> str:
        if chosen_color is None:
            return f"play:{card_label(card)}"
        return f"play:{card_label(card)}:{chosen_color}"

    def _apply_card_effect(self, card: dict[str, Any], actor: int) -> None:
        target = opponent(actor)
        kind = card["kind"]
        if kind == "draw_two":
            self._draw_cards(target, 2)
            self.actor = actor
        elif kind == "wild_draw_four":
            self._draw_cards(target, 4)
            self.actor = actor
        elif kind in {"skip", "reverse"}:
            self.actor = actor
        else:
            self.actor = target

    def _draw_cards(self, player: int, count: int) -> int:
        drawn = 0
        for _ in range(count):
            if not self.draw_pile and not self._reshuffle_discard_into_draw():
                break
            self.hands[player].append(self.draw_pile.pop())
            drawn += 1
        return drawn

    def _can_draw(self) -> bool:
        return bool(self.draw_pile) or len(self.discard_pile) > 1

    def _reshuffle_discard_into_draw(self) -> bool:
        if len(self.discard_pile) <= 1:
            return False
        top = self.discard_pile.pop()
        self.draw_pile = self.discard_pile
        self.rng.shuffle(self.draw_pile)
        self.discard_pile = [top]
        return bool(self.draw_pile)

    def _leader_by_hand_size(self) -> int | None:
        counts = [len(self.hands[0]), len(self.hands[1])]
        if counts[0] == counts[1]:
            return None
        return 0 if counts[0] < counts[1] else 1

    def _outcome_info(self) -> dict[str, Any]:
        if (
            self.max_plies is not None
            and self.plies >= self.max_plies
            and not self.is_done()
        ):
            return {
                "status": "turn_limit",
                "winner": self._leader_by_hand_size(),
                "result": _result_for_winner(self._leader_by_hand_size()),
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


class SystemBot:
    """A simple built-in opponent for battle mode."""

    name = "system"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_action(self, state: dict[str, Any]) -> str:
        legal = state["legal_actions"]
        if not legal:
            raise ValueError("system bot received no legal actions")
        if DRAW_ACTION in legal:
            return DRAW_ACTION
        if PASS_ACTION in legal:
            return PASS_ACTION

        scored = [(_action_score(state, action), action) for action in legal]
        best = max(score for score, _ in scored)
        choices = [action for score, action in scored if score == best]
        return self.rng.choice(sorted(choices))


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

    module_name = f"uno_user_bot_{uuid.uuid4().hex}"
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
    seat: int = 0,
    seed: int | None = None,
    keep_log: bool = True,
    max_plies: int = DEFAULT_MAX_PLIES,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("uno only supports players=2")
    if seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    validate_decision_timeout(decision_timeout)

    game_seed = seed if seed is not None else random.randrange(1 << 30)
    rng = random.Random(game_seed)
    game_id = uuid.uuid4().hex[:12]
    log: list[str] = [f"G:{game_id}:N2:SEED{game_seed}:GAME:UNO"]

    developer_bot = load_bot(bot_path)
    bots: list[Any] = [SystemBot(rng), SystemBot(rng)]
    bots[seat] = developer_bot
    bot_names = [getattr(bot, "name", f"bot_{index}") for index, bot in enumerate(bots)]

    env = UnoEnv(seed=game_seed, max_plies=max_plies)
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
        top = env.discard_pile[-1]
        log.append(
            f"T{env.plies}:P{actor}:ACTION:{env.last_action}:COLOR:{env.current_color}:"
            f"TOP:{card_label(top)}:HANDS:{len(env.hands[0])}-{len(env.hands[1])}:DRAW:{len(env.draw_pile)}"
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
        winner = env._leader_by_hand_size()
        result = _result_for_winner(winner)

    hand_counts = [len(env.hands[0]), len(env.hands[1])]
    log.append(
        "END:"
        f"{status}:WINNER:{winner}:RESULT:{result}:PLIES:{env.plies}:"
        f"HANDS:{hand_counts[0]}-{hand_counts[1]}:TOP:{card_label(env.discard_pile[-1])}:COLOR:{env.current_color}"
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
        "hand_counts": hand_counts,
        "top_card": card_view(env.discard_pile[-1], current_color=env.current_color),
        "current_color": env.current_color,
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
    seat: int = 0,
    seed: int | None = None,
    alternate_seats: bool = True,
    keep_logs: bool = False,
    max_plies: int = DEFAULT_MAX_PLIES,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("uno only supports players=2")
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
        if result["winner"] in (0, 1):
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
    if "draw" in legal:
        return "draw"
    return legal[0]
'''
    Path(path).write_text(sample, encoding="utf-8")


def _result_for_winner(winner: int | None) -> str:
    if winner == 0:
        return "1-0"
    if winner == 1:
        return "0-1"
    return "1/2-1/2"


def _action_score(state: dict[str, Any], action: str) -> int:
    if action == DRAW_ACTION:
        return -100
    if action == PASS_ACTION:
        return -200
    card_id = int(action.split(":")[1])
    card = next(item for item in state["hand"] if item["id"] == card_id)
    score = 10
    if card["kind"] == "wild_draw_four":
        score += 70
    elif card["kind"] == "draw_two":
        score += 55
    elif card["kind"] in {"skip", "reverse"}:
        score += 35
    elif card["kind"] == "wild":
        score += 20
    if state["opponent_hand_count"] <= 2 and card["kind"] in {"draw_two", "wild_draw_four", "skip", "reverse"}:
        score += 60
    score -= _color_count(state["hand"], card.get("color")) if card.get("color") else 0
    if card["kind"] in WILD_KINDS:
        chosen = action.split(":")[2]
        score += 4 * _color_count(state["hand"], chosen)
    return score


def _color_count(hand: list[dict[str, Any]], color: str | None) -> int:
    if color is None:
        return 0
    return sum(1 for card in hand if card["color"] == color)


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Two-player UNO Gym-style env and bot battle referee")
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
    battle_parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
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
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
