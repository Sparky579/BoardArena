#!/usr/bin/env python3
"""Multiplayer simplified Skull judge and bot battle API.

This module is intentionally separate from ``skull_cfr.py``. The CFR trainer
and bundled policies are two-player artifacts, while this judge supports 2-6
players and a stable developer bot interface.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import random
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

FLOWER = "F"
SKULL = "S"
PLAY_FLOWER = "PLAY_F"
PLAY_SKULL = "PLAY_S"
PASS = "PASS"

Hand = Tuple[int, int]
Hands = Tuple[Hand, ...]
Pile = Tuple[str, ...]
Piles = Tuple[Pile, ...]
Scores = Tuple[int, ...]
BoolTuple = Tuple[bool, ...]
IntTuple = Tuple[int, ...]

MAX_PLAYERS = 6
WIN_SCORE = 2

MATCH_LOGS: Dict[str, List[str]] = {}


def hand_total(hand: Hand) -> int:
    return hand[0] + hand[1]


def pile_counts(pile: Pile) -> Hand:
    return (pile.count(FLOWER), pile.count(SKULL))


def add_hands(a: Hand, b: Hand) -> Hand:
    return (a[0] + b[0], a[1] + b[1])


def sub_card(hand: Hand, card: str) -> Hand:
    flowers, skulls = hand
    if card == FLOWER:
        if flowers <= 0:
            raise ValueError("cannot remove missing flower")
        return (flowers - 1, skulls)
    if card == SKULL:
        if skulls <= 0:
            raise ValueError("cannot remove missing skull")
        return (flowers, skulls - 1)
    raise ValueError(f"unknown card: {card}")


def bid_action(n: int) -> str:
    return f"BID_{n}"


def is_bid(action: str) -> bool:
    return action.startswith("BID_")


def bid_value(action: str) -> int:
    return int(action.split("_", 1)[1])


def flip_action(player: int) -> str:
    return f"FLIP_{player}"


def is_flip(action: str) -> bool:
    return action.startswith("FLIP_")


def flip_target(action: str) -> int:
    return int(action.split("_", 1)[1])


def weighted_choice(rng: random.Random, items: Sequence[Tuple[str, float]]) -> str:
    total = sum(weight for _, weight in items)
    if total <= 0.0:
        return items[rng.randrange(len(items))][0]
    target = rng.random() * total
    upto = 0.0
    for item, weight in items:
        upto += weight
        if upto >= target:
            return item
    return items[-1][0]


@dataclass(frozen=True)
class MultiState:
    hands: Hands
    piles: Piles
    scores: Scores
    actor: int = 0
    phase: str = "play"  # play, bid, challenge, remove
    current_bid: int = 0
    high_bidder: int = -1
    passed: BoolTuple = ()
    pending_loser: int = -1
    challenge_remaining: int = 0
    flipped_counts: IntTuple = ()
    turn: int = 0

    @classmethod
    def new(cls, num_players: int) -> "MultiState":
        if not 2 <= num_players <= MAX_PLAYERS:
            raise ValueError(f"num_players must be between 2 and {MAX_PLAYERS}")
        return cls(
            hands=tuple((3, 1) for _ in range(num_players)),
            piles=tuple(() for _ in range(num_players)),
            scores=tuple(0 for _ in range(num_players)),
            passed=tuple(False for _ in range(num_players)),
            flipped_counts=tuple(0 for _ in range(num_players)),
        )

    @property
    def num_players(self) -> int:
        return len(self.hands)

    def total_cards(self, player: int) -> int:
        return hand_total(self.hands[player]) + len(self.piles[player])

    def total_piled(self) -> int:
        return sum(len(pile) for pile in self.piles)

    def active_players(self) -> List[int]:
        return [p for p in range(self.num_players) if self.total_cards(p) > 0]

    def winner(self) -> Optional[int]:
        for player, score in enumerate(self.scores):
            if score >= WIN_SCORE:
                return player
        active = self.active_players()
        if len(active) == 1:
            return active[0]
        return None

    def can_start_bid(self) -> bool:
        active = self.active_players()
        return bool(active) and all(len(self.piles[player]) > 0 for player in active)

    def legal_actions(self) -> List[str]:
        if self.winner() is not None or self.phase == "remove":
            return []

        if self.phase == "play":
            actions: List[str] = []
            flowers, skulls = self.hands[self.actor]
            if flowers > 0:
                actions.append(PLAY_FLOWER)
            if skulls > 0:
                actions.append(PLAY_SKULL)
            if self.can_start_bid():
                actions.extend(bid_action(n) for n in range(1, self.total_piled() + 1))
            return actions

        if self.phase == "bid":
            actions = [PASS]
            actions.extend(
                bid_action(n) for n in range(self.current_bid + 1, self.total_piled() + 1)
            )
            return actions

        if self.phase == "challenge":
            actions = []
            for player in self.active_players():
                if player == self.high_bidder:
                    continue
                if self.flipped_counts[player] < len(self.piles[player]):
                    actions.append(flip_action(player))
            return actions

        raise ValueError(f"unknown phase: {self.phase}")

    def apply_action(self, action: str) -> Tuple["MultiState", List[str]]:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}; legal={self.legal_actions()}")

        if self.phase == "play":
            if action == PLAY_FLOWER:
                return self._play_card(FLOWER)
            if action == PLAY_SKULL:
                return self._play_card(SKULL)
            if is_bid(action):
                return self._start_bidding(bid_value(action))

        if self.phase == "bid":
            if action == PASS:
                return self._pass_bid()
            if is_bid(action):
                return self._raise_bid(bid_value(action))

        if self.phase == "challenge" and is_flip(action):
            return self._flip_opponent_card(flip_target(action))

        raise ValueError(f"illegal action {action} in phase {self.phase}")

    def chance_outcomes(self) -> List[Tuple[str, float]]:
        if self.phase != "remove":
            return []
        loser = self.pending_loser
        total = add_hands(self.hands[loser], pile_counts(self.piles[loser]))
        count = hand_total(total)
        if count <= 0:
            return []
        outcomes: List[Tuple[str, float]] = []
        if total[0] > 0:
            outcomes.append((FLOWER, total[0] / count))
        if total[1] > 0:
            outcomes.append((SKULL, total[1] / count))
        return outcomes

    def apply_chance(self, removed_card: str) -> Tuple["MultiState", List[str]]:
        if self.phase != "remove":
            raise ValueError("not at a chance node")

        loser = self.pending_loser
        totals = [
            add_hands(self.hands[player], pile_counts(self.piles[player]))
            for player in range(self.num_players)
        ]
        totals[loser] = sub_card(totals[loser], removed_card)
        next_actor = self._next_play_actor(loser, totals)
        state = MultiState(
            hands=tuple(totals),
            piles=tuple(() for _ in range(self.num_players)),
            scores=self.scores,
            actor=next_actor,
            phase="play",
            passed=tuple(False for _ in range(self.num_players)),
            flipped_counts=tuple(0 for _ in range(self.num_players)),
            turn=self.turn + 1,
        )
        return state, [f"T{self.turn}:X:P{loser}:{removed_card}"]

    def public_state(self, player_id: int) -> Dict[str, Any]:
        return {
            "player_id": player_id,
            "num_players": self.num_players,
            "phase": self.phase,
            "actor": self.actor,
            "legal_actions": self.legal_actions() if self.actor == player_id else [],
            "scores": list(self.scores),
            "hand": {"flowers": self.hands[player_id][0], "skulls": self.hands[player_id][1]},
            "own_pile": list(self.piles[player_id]),
            "pile_sizes": [len(pile) for pile in self.piles],
            "total_cards": [self.total_cards(player) for player in range(self.num_players)],
            "current_bid": self.current_bid,
            "high_bidder": self.high_bidder,
            "passed": list(self.passed),
            "challenge_remaining": self.challenge_remaining,
            "flipped_counts": list(self.flipped_counts),
            "turn": self.turn,
        }

    def _play_card(self, card: str) -> Tuple["MultiState", List[str]]:
        hands = list(self.hands)
        piles = [tuple(pile) for pile in self.piles]
        hands[self.actor] = sub_card(hands[self.actor], card)
        piles[self.actor] = piles[self.actor] + (card,)
        actor = self._next_play_actor(self.actor, hands)
        state = MultiState(
            hands=tuple(hands),
            piles=tuple(piles),
            scores=self.scores,
            actor=actor,
            phase="play",
            passed=self.passed,
            flipped_counts=self.flipped_counts,
            turn=self.turn + 1,
        )
        return state, [f"T{self.turn}:P{self.actor}:PLAY_{card}"]

    def _start_bidding(self, amount: int) -> Tuple["MultiState", List[str]]:
        passed = [self.total_cards(player) <= 0 for player in range(self.num_players)]
        passed[self.actor] = False
        state = MultiState(
            hands=self.hands,
            piles=self.piles,
            scores=self.scores,
            actor=self._next_bid_actor(self.actor, tuple(passed), self.actor),
            phase="bid",
            current_bid=amount,
            high_bidder=self.actor,
            passed=tuple(passed),
            flipped_counts=self.flipped_counts,
            turn=self.turn + 1,
        )
        return state, [f"T{self.turn}:P{self.actor}:B{amount}"]

    def _pass_bid(self) -> Tuple["MultiState", List[str]]:
        passed = list(self.passed)
        passed[self.actor] = True
        events = [f"T{self.turn}:P{self.actor}:PASS"]
        if self._bid_is_over(tuple(passed)):
            state, challenge_events = self._start_challenge(self.high_bidder, self.current_bid)
            return state, events + challenge_events
        state = MultiState(
            hands=self.hands,
            piles=self.piles,
            scores=self.scores,
            actor=self._next_bid_actor(self.actor, tuple(passed), self.high_bidder),
            phase="bid",
            current_bid=self.current_bid,
            high_bidder=self.high_bidder,
            passed=tuple(passed),
            flipped_counts=self.flipped_counts,
            turn=self.turn + 1,
        )
        return state, events

    def _raise_bid(self, amount: int) -> Tuple["MultiState", List[str]]:
        passed = list(self.passed)
        passed[self.actor] = False
        events = [f"T{self.turn}:P{self.actor}:B{amount}"]
        if self._bid_is_over(tuple(passed), high_bidder=self.actor):
            state, challenge_events = self._start_challenge(self.actor, amount)
            return state, events + challenge_events
        state = MultiState(
            hands=self.hands,
            piles=self.piles,
            scores=self.scores,
            actor=self._next_bid_actor(self.actor, tuple(passed), self.actor),
            phase="bid",
            current_bid=amount,
            high_bidder=self.actor,
            passed=tuple(passed),
            flipped_counts=self.flipped_counts,
            turn=self.turn + 1,
        )
        return state, events

    def _start_challenge(self, bidder: int, amount: int) -> Tuple["MultiState", List[str]]:
        remaining = amount
        flipped = [0 for _ in range(self.num_players)]
        events: List[str] = []
        own_to_flip = min(len(self.piles[bidder]), remaining)
        for idx in range(own_to_flip):
            card = self.piles[bidder][len(self.piles[bidder]) - 1 - idx]
            events.append(f"T{self.turn}:R:P{bidder}:{card}")
            flipped[bidder] += 1
            remaining -= 1
            if card == SKULL:
                state = self._remove_state(bidder)
                return state, events + [f"T{self.turn}:BAD:P{bidder}"]

        if remaining <= 0:
            state = self._score_state(bidder)
            return state, events + [f"T{self.turn}:OK:P{bidder}"]

        state = MultiState(
            hands=self.hands,
            piles=self.piles,
            scores=self.scores,
            actor=bidder,
            phase="challenge",
            current_bid=amount,
            high_bidder=bidder,
            passed=self.passed,
            challenge_remaining=remaining,
            flipped_counts=tuple(flipped),
            turn=self.turn + 1,
        )
        return state, events

    def _flip_opponent_card(self, target: int) -> Tuple["MultiState", List[str]]:
        flipped = list(self.flipped_counts)
        card_index = len(self.piles[target]) - 1 - flipped[target]
        card = self.piles[target][card_index]
        flipped[target] += 1
        remaining = self.challenge_remaining - 1
        events = [f"T{self.turn}:R:P{target}:{card}"]
        if card == SKULL:
            state = self._remove_state(self.high_bidder)
            return state, events + [f"T{self.turn}:BAD:P{self.high_bidder}"]
        if remaining <= 0:
            state = self._score_state(self.high_bidder)
            return state, events + [f"T{self.turn}:OK:P{self.high_bidder}"]
        state = MultiState(
            hands=self.hands,
            piles=self.piles,
            scores=self.scores,
            actor=self.high_bidder,
            phase="challenge",
            current_bid=self.current_bid,
            high_bidder=self.high_bidder,
            passed=self.passed,
            challenge_remaining=remaining,
            flipped_counts=tuple(flipped),
            turn=self.turn + 1,
        )
        return state, events

    def _remove_state(self, loser: int) -> "MultiState":
        return MultiState(
            hands=self.hands,
            piles=self.piles,
            scores=self.scores,
            actor=-1,
            phase="remove",
            pending_loser=loser,
            passed=tuple(False for _ in range(self.num_players)),
            flipped_counts=tuple(0 for _ in range(self.num_players)),
            turn=self.turn + 1,
        )

    def _score_state(self, bidder: int) -> "MultiState":
        scores = list(self.scores)
        scores[bidder] += 1
        totals = tuple(
            add_hands(self.hands[player], pile_counts(self.piles[player]))
            for player in range(self.num_players)
        )
        actor = self._next_play_actor(bidder, totals)
        return MultiState(
            hands=totals,
            piles=tuple(() for _ in range(self.num_players)),
            scores=tuple(scores),
            actor=actor,
            phase="play",
            passed=tuple(False for _ in range(self.num_players)),
            flipped_counts=tuple(0 for _ in range(self.num_players)),
            turn=self.turn + 1,
        )

    def _next_play_actor(self, start: int, hands: Sequence[Hand]) -> int:
        can_bid = all(
            len(self.piles[player]) > 0
            for player in range(self.num_players)
            if hand_total(hands[player]) + len(self.piles[player]) > 0
        )
        for offset in range(1, self.num_players + 1):
            player = (start + offset) % self.num_players
            has_cards = hand_total(hands[player]) + len(self.piles[player]) > 0
            if has_cards and (hand_total(hands[player]) > 0 or can_bid):
                return player
        return start

    def _next_bid_actor(self, start: int, passed: BoolTuple, high_bidder: int) -> int:
        for offset in range(1, self.num_players + 1):
            player = (start + offset) % self.num_players
            if player == high_bidder:
                continue
            if self.total_cards(player) > 0 and not passed[player]:
                return player
        return high_bidder

    def _bid_is_over(self, passed: BoolTuple, high_bidder: Optional[int] = None) -> bool:
        high = self.high_bidder if high_bidder is None else high_bidder
        contenders = [
            player
            for player in range(self.num_players)
            if self.total_cards(player) > 0 and not passed[player]
        ]
        return contenders == [high]


class BotLike(Protocol):
    name: str

    def choose_action(self, state: Dict[str, Any]) -> str:
        ...


class SystemBot:
    def __init__(self, name: str = "system", seed: int = 0) -> None:
        self.name = name
        self.rng = random.Random(seed)

    def choose_action(self, state: Dict[str, Any]) -> str:
        legal = list(state["legal_actions"])
        if not legal:
            raise ValueError("system bot received no legal actions")

        if state["phase"] == "challenge":
            target_sizes = [
                (idx, size - state["flipped_counts"][idx])
                for idx, size in enumerate(state["pile_sizes"])
                if flip_action(idx) in legal
            ]
            target = min(target_sizes, key=lambda item: item[1])[0]
            return flip_action(target)

        if state["phase"] == "bid":
            raises = [action for action in legal if is_bid(action)]
            if raises and self.rng.random() < 0.32:
                return raises[0]
            return PASS

        bids = [action for action in legal if is_bid(action)]
        plays = [action for action in legal if action in (PLAY_FLOWER, PLAY_SKULL)]
        hand = state["hand"]
        own_safe = hand["skulls"] == 0 or self.rng.random() < 0.72
        if bids and (not plays or self.rng.random() < 0.16):
            return bids[0]
        if PLAY_FLOWER in plays and own_safe:
            return PLAY_FLOWER
        if PLAY_SKULL in plays:
            return PLAY_SKULL
        return legal[0]


class FunctionBot:
    def __init__(self, name: str, func: Callable[[Dict[str, Any]], str]) -> None:
        self.name = name
        self.func = func

    def choose_action(self, state: Dict[str, Any]) -> str:
        return str(self.func(state))


class BotTimeoutError(TimeoutError):
    """Raised when a bot does not return an action before the deadline."""


def choose_action_with_timeout(
    bot: BotLike,
    state: Dict[str, Any],
    decision_timeout: Optional[float],
) -> str:
    if decision_timeout is None:
        return bot.choose_action(state)

    result_queue: queue.Queue[Tuple[bool, Any]] = queue.Queue(maxsize=1)

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


def validate_decision_timeout(decision_timeout: Optional[float]) -> None:
    if decision_timeout is not None and decision_timeout <= 0:
        raise ValueError("decision_timeout must be positive seconds or None")


@dataclass(frozen=True)
class MatchResult:
    game_id: str
    winner: Optional[int]
    status: str
    turns: int
    scores: Scores
    bot_names: Tuple[str, ...]
    developer_seat: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "winner": self.winner,
            "status": self.status,
            "turns": self.turns,
            "scores": list(self.scores),
            "bot_names": list(self.bot_names),
            "developer_seat": self.developer_seat,
            "developer_win": (
                None if self.developer_seat is None or self.winner is None else self.winner == self.developer_seat
            ),
            "error": self.error,
        }


def load_bot(path: Path | str, name: Optional[str] = None) -> BotLike:
    bot_path = Path(path)
    module_name = f"skull_user_bot_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, bot_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load bot from {bot_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "Bot"):
        instance = module.Bot()
        if not hasattr(instance, "choose_action"):
            raise ValueError("Bot class must define choose_action(state)")
        if not hasattr(instance, "name"):
            instance.name = name or bot_path.stem
        return instance

    if hasattr(module, "choose_action"):
        return FunctionBot(name or bot_path.stem, module.choose_action)

    raise ValueError("bot.py must expose choose_action(state) or Bot.choose_action(state)")


def run_match(
    bots: Sequence[BotLike],
    seed: int = 1,
    game_id: Optional[str] = None,
    keep_log: bool = True,
    max_turns: int = 512,
    developer_seat: Optional[int] = None,
    decision_timeout: Optional[float] = None,
) -> MatchResult:
    if not 2 <= len(bots) <= MAX_PLAYERS:
        raise ValueError(f"bots length must be between 2 and {MAX_PLAYERS}")
    validate_decision_timeout(decision_timeout)

    rng = random.Random(seed)
    state = MultiState.new(len(bots))
    gid = game_id or uuid.uuid4().hex[:12]
    log = [f"G:{gid}:N{len(bots)}:SEED{seed}"]

    while state.winner() is None and state.turn < max_turns:
        if state.phase == "remove":
            card = weighted_choice(rng, state.chance_outcomes())
            state, events = state.apply_chance(card)
            log.extend(events)
            continue

        actor = state.actor
        public = state.public_state(actor)
        try:
            action = choose_action_with_timeout(bots[actor], public, decision_timeout)
        except BotTimeoutError as exc:
            result = _forfeit_result(gid, bots, state, actor, "timeout", str(exc), developer_seat)
            log.append(f"T{state.turn}:ERR:P{actor}:TIMEOUT")
            if keep_log:
                MATCH_LOGS[gid] = log
            return result
        except Exception as exc:  # pragma: no cover - exercised by external bots
            result = _forfeit_result(gid, bots, state, actor, "bot_exception", str(exc), developer_seat)
            log.append(f"T{state.turn}:ERR:P{actor}:EXCEPTION:{type(exc).__name__}")
            if keep_log:
                MATCH_LOGS[gid] = log
            return result

        legal = state.legal_actions()
        if action not in legal:
            result = _forfeit_result(
                gid,
                bots,
                state,
                actor,
                "invalid_action",
                f"{action!r} not in {legal!r}",
                developer_seat,
            )
            log.append(f"T{state.turn}:ERR:P{actor}:INVALID:{action}")
            if keep_log:
                MATCH_LOGS[gid] = log
            return result

        state, events = state.apply_action(action)
        log.extend(events)

    status = "ok" if state.winner() is not None else "turn_limit"
    log.append(f"END:{status}:WINNER:{state.winner()}:SCORES:{','.join(map(str, state.scores))}")
    if keep_log:
        MATCH_LOGS[gid] = log
    return MatchResult(
        game_id=gid,
        winner=state.winner(),
        status=status,
        turns=state.turn,
        scores=state.scores,
        bot_names=tuple(bot.name for bot in bots),
        developer_seat=developer_seat,
    )


def battle_once(
    bot_path: Path | str,
    players: int = 2,
    seat: int = 0,
    seed: int = 1,
    keep_log: bool = True,
    decision_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    validate_decision_timeout(decision_timeout)
    bot = load_bot(bot_path)
    bots: List[BotLike] = [SystemBot(name=f"system_{idx}", seed=seed + idx) for idx in range(players)]
    bots[seat] = bot
    result = run_match(
        bots,
        seed=seed,
        keep_log=keep_log,
        developer_seat=seat,
        decision_timeout=decision_timeout,
    )
    return result.to_dict()


def battle_many(
    bot_path: Path | str,
    games: int = 100,
    players: int = 2,
    seed: int = 1,
    alternate_seats: bool = True,
    keep_logs: bool = False,
    decision_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    if games < 1:
        raise ValueError("games must be positive")
    validate_decision_timeout(decision_timeout)
    bot = load_bot(bot_path)
    wins = [0 for _ in range(players)]
    developer_wins = 0
    statuses: Dict[str, int] = {}
    game_ids: List[str] = []

    for idx in range(games):
        seat = idx % players if alternate_seats else 0
        bots: List[BotLike] = [
            SystemBot(name=f"system_{player}", seed=seed + idx * 31 + player)
            for player in range(players)
        ]
        bots[seat] = bot
        result = run_match(
            bots,
            seed=seed + idx,
            keep_log=keep_logs,
            developer_seat=seat,
            decision_timeout=decision_timeout,
        )
        statuses[result.status] = statuses.get(result.status, 0) + 1
        game_ids.append(result.game_id)
        if result.winner is not None:
            wins[result.winner] += 1
            if result.winner == seat:
                developer_wins += 1

    return {
        "games": games,
        "players": players,
        "wins_by_seat": wins,
        "developer_wins": developer_wins,
        "developer_losses": games - developer_wins,
        "developer_win_rate": developer_wins / games,
        "statuses": statuses,
        "game_ids": game_ids if keep_logs else [],
    }


def get_match_log(game_id: str) -> List[str]:
    if game_id not in MATCH_LOGS:
        raise KeyError(f"unknown game_id: {game_id}")
    return list(MATCH_LOGS[game_id])


def _forfeit_result(
    game_id: str,
    bots: Sequence[BotLike],
    state: MultiState,
    actor: int,
    status: str,
    error: str,
    developer_seat: Optional[int],
) -> MatchResult:
    candidates = [player for player in range(len(bots)) if player != actor and state.total_cards(player) > 0]
    winner = candidates[0] if candidates else None
    return MatchResult(
        game_id=game_id,
        winner=winner,
        status=status,
        turns=state.turn,
        scores=state.scores,
        bot_names=tuple(bot.name for bot in bots),
        developer_seat=developer_seat,
        error=error,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    battle = sub.add_parser("battle", help="run one or many developer-bot battles")
    battle.add_argument("--bot", type=Path, required=True)
    battle.add_argument("--players", type=int, default=2)
    battle.add_argument("--games", type=int, default=1)
    battle.add_argument("--seat", type=int, default=0)
    battle.add_argument("--seed", type=int, default=1)
    battle.add_argument("--keep-logs", action="store_true")
    battle.add_argument("--fixed-seat", action="store_true")
    battle.add_argument("--decision-timeout", type=float, default=None)

    sample = sub.add_parser("sample-bot", help="print a minimal bot.py example")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "sample-bot":
        print(SAMPLE_BOT.strip())
        return 0

    if args.cmd == "battle":
        if not 2 <= args.players <= MAX_PLAYERS:
            parser.error(f"--players must be between 2 and {MAX_PLAYERS}")
        if not 0 <= args.seat < args.players:
            parser.error("--seat must be within player range")
        if args.games == 1:
            result = battle_once(
                args.bot,
                players=args.players,
                seat=args.seat,
                seed=args.seed,
                keep_log=args.keep_logs,
                decision_timeout=args.decision_timeout,
            )
        else:
            result = battle_many(
                args.bot,
                games=args.games,
                players=args.players,
                seed=args.seed,
                alternate_seats=not args.fixed_seat,
                keep_logs=args.keep_logs,
                decision_timeout=args.decision_timeout,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    parser.error("unknown command")
    return 2


SAMPLE_BOT = r'''
def choose_action(state):
    legal = state["legal_actions"]
    if state["phase"] == "challenge":
        return legal[0]
    if "PLAY_F" in legal:
        return "PLAY_F"
    bids = [action for action in legal if action.startswith("BID_")]
    if bids:
        return bids[0]
    return "PASS" if "PASS" in legal else legal[0]
'''


if __name__ == "__main__":
    raise SystemExit(main())
