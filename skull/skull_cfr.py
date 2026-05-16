#!/usr/bin/env python3
"""Two-player simplified Skull with recursive information-set CFR.

Rules implemented here:
- Each player starts with 3 flowers and 1 skull.
- A round alternates face-down plays until someone bids.
- Bidding then alternates raises or pass. With two players, a pass resolves
  the current highest bid immediately.
- The bidder flips up to the bid amount from their own stack first, then the
  opponent's stack. Any skull is a failed challenge.
- A successful challenge scores 1 point. Two points wins the match.
- A failed challenge removes one random card from the loser. If a player has
  no cards left, the other player wins.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FLOWER = "F"
SKULL = "S"
PLAY_FLOWER = "PLAY_F"
PLAY_SKULL = "PLAY_S"
PASS = "PASS"

Hand = Tuple[int, int]  # (flowers, skulls)
Hands = Tuple[Hand, Hand]
Pile = Tuple[str, ...]
Piles = Tuple[Pile, Pile]
Scores = Tuple[int, int]


def other(player: int) -> int:
    return 1 - player


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
    if skulls <= 0:
        raise ValueError("cannot remove missing skull")
    return (flowers, skulls - 1)


def bid_action(n: int) -> str:
    return f"BID_{n}"


def is_bid(action: str) -> bool:
    return action.startswith("BID_")


def bid_value(action: str) -> int:
    return int(action.split("_", 1)[1])


@dataclass(frozen=True)
class State:
    hands: Hands = ((3, 1), (3, 1))
    piles: Piles = ((), ())
    scores: Scores = (0, 0)
    actor: int = 0
    phase: str = "play"  # play, bid, remove
    current_bid: int = 0
    high_bidder: int = -1
    pending_loser: int = -1
    history: Tuple[str, ...] = ()

    def total_cards(self, player: int) -> int:
        return hand_total(self.hands[player]) + len(self.piles[player])

    def total_piled(self) -> int:
        return len(self.piles[0]) + len(self.piles[1])

    def winner(self) -> Optional[int]:
        if self.scores[0] >= 2:
            return 0
        if self.scores[1] >= 2:
            return 1
        if self.total_cards(0) <= 0:
            return 1
        if self.total_cards(1) <= 0:
            return 0
        return None

    def legal_actions(self) -> List[str]:
        if self.winner() is not None or self.phase == "remove":
            return []

        total_piled = self.total_piled()
        actions: List[str] = []

        if self.phase == "play":
            flowers, skulls = self.hands[self.actor]
            if flowers > 0:
                actions.append(PLAY_FLOWER)
            if skulls > 0:
                actions.append(PLAY_SKULL)
            if all(len(pile) > 0 for pile in self.piles):
                actions.extend(bid_action(n) for n in range(1, total_piled + 1))
            return actions

        if self.phase == "bid":
            actions.append(PASS)
            actions.extend(
                bid_action(n) for n in range(self.current_bid + 1, total_piled + 1)
            )
            return actions

        raise ValueError(f"unknown phase: {self.phase}")

    def apply_action(self, action: str) -> "State":
        if self.phase == "play":
            if action == PLAY_FLOWER:
                return self._play_card(FLOWER)
            if action == PLAY_SKULL:
                return self._play_card(SKULL)
            if is_bid(action):
                amount = bid_value(action)
                if not 1 <= amount <= self.total_piled():
                    raise ValueError(f"illegal bid {amount}")
                return State(
                    hands=self.hands,
                    piles=self.piles,
                    scores=self.scores,
                    actor=other(self.actor),
                    phase="bid",
                    current_bid=amount,
                    high_bidder=self.actor,
                    history=self.history + (f"P{self.actor}:B{amount}",),
                )

        if self.phase == "bid":
            if action == PASS:
                return self._resolve_bid()
            if is_bid(action):
                amount = bid_value(action)
                if not self.current_bid < amount <= self.total_piled():
                    raise ValueError(f"illegal raise {amount}")
                return State(
                    hands=self.hands,
                    piles=self.piles,
                    scores=self.scores,
                    actor=other(self.actor),
                    phase="bid",
                    current_bid=amount,
                    high_bidder=self.actor,
                    history=self.history + (f"P{self.actor}:B{amount}",),
                )

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

    def apply_chance(self, removed_card: str) -> "State":
        if self.phase != "remove":
            raise ValueError("not at a chance node")

        loser = self.pending_loser
        totals = [
            add_hands(self.hands[0], pile_counts(self.piles[0])),
            add_hands(self.hands[1], pile_counts(self.piles[1])),
        ]
        totals[loser] = sub_card(totals[loser], removed_card)
        return State(
            hands=(totals[0], totals[1]),
            piles=((), ()),
            scores=self.scores,
            actor=loser,
            phase="play",
            history=self.history + (f"P{loser}:LOSE",),
        )

    def info_key(self, player: int, recall: str = "compact") -> str:
        own = player
        opp = other(player)
        high = "none" if self.high_bidder < 0 else ("me" if self.high_bidder == own else "opp")
        parts = [
            f"p={player}",
            f"score={self.scores[own]}-{self.scores[opp]}",
            f"hand={self.hands[own][0]},{self.hands[own][1]}",
            f"pile={''.join(self.piles[own]) or '-'}",
            f"opp_cards={self.total_cards(opp)}",
            f"opp_pile={len(self.piles[opp])}",
            f"phase={self.phase}",
            f"bid={self.current_bid}",
            f"high={high}",
            f"turn={'me' if self.actor == own else 'opp'}",
        ]
        if recall == "perfect":
            parts.append("hist=" + ",".join(self.history))
        elif recall != "compact":
            raise ValueError("recall must be compact or perfect")
        return "|".join(parts)

    def public_view(self) -> str:
        high = "-" if self.high_bidder < 0 else f"P{self.high_bidder}"
        return (
            f"score P0 {self.scores[0]} - P1 {self.scores[1]} | "
            f"cards P0 {self.total_cards(0)} - P1 {self.total_cards(1)} | "
            f"piles P0 {len(self.piles[0])} - P1 {len(self.piles[1])} | "
            f"{self.phase} actor P{self.actor} bid {self.current_bid} high {high}"
        )

    def _play_card(self, card: str) -> "State":
        hand = sub_card(self.hands[self.actor], card)
        hands = list(self.hands)
        piles = [self.piles[0], self.piles[1]]
        hands[self.actor] = hand
        piles[self.actor] = piles[self.actor] + (card,)
        return State(
            hands=(hands[0], hands[1]),
            piles=(piles[0], piles[1]),
            scores=self.scores,
            actor=other(self.actor),
            phase="play",
            current_bid=0,
            high_bidder=-1,
            history=self.history + (f"P{self.actor}:PLAY",),
        )

    def _resolve_bid(self) -> "State":
        bidder = self.high_bidder
        if bidder < 0:
            raise ValueError("cannot resolve without a bidder")

        success = self._challenge_succeeds(bidder, self.current_bid)
        if success:
            scores = list(self.scores)
            scores[bidder] += 1
            totals = (
                add_hands(self.hands[0], pile_counts(self.piles[0])),
                add_hands(self.hands[1], pile_counts(self.piles[1])),
            )
            return State(
                hands=totals,
                piles=((), ()),
                scores=(scores[0], scores[1]),
                actor=bidder,
                phase="play",
                history=self.history + (f"P{self.actor}:PASS", f"P{bidder}:SCORE"),
            )

        return State(
            hands=self.hands,
            piles=self.piles,
            scores=self.scores,
            actor=-1,
            phase="remove",
            current_bid=0,
            high_bidder=-1,
            pending_loser=bidder,
            history=self.history + (f"P{self.actor}:PASS", f"P{bidder}:BUST"),
        )

    def _challenge_succeeds(self, bidder: int, amount: int) -> bool:
        remaining = amount

        own_flips = list(reversed(self.piles[bidder]))[:remaining]
        if SKULL in own_flips:
            return False
        remaining -= len(own_flips)
        if remaining <= 0:
            return True

        opp_flips = list(reversed(self.piles[other(bidder)]))[:remaining]
        return SKULL not in opp_flips


class InfoSet:
    def __init__(self) -> None:
        self.regret_sum: Dict[str, float] = {}
        self.strategy_sum: Dict[str, float] = {}
        self.visits = 0

    def strategy(self, legal_actions: Sequence[str], epsilon: float = 0.0) -> Dict[str, float]:
        for action in legal_actions:
            self.regret_sum.setdefault(action, 0.0)
            self.strategy_sum.setdefault(action, 0.0)

        positive = [max(self.regret_sum[action], 0.0) for action in legal_actions]
        normalizer = sum(positive)
        if normalizer > 1e-12:
            strategy = {
                action: positive[i] / normalizer for i, action in enumerate(legal_actions)
            }
        else:
            p = 1.0 / len(legal_actions)
            strategy = {action: p for action in legal_actions}

        if epsilon > 0.0 and len(legal_actions) > 1:
            uniform = 1.0 / len(legal_actions)
            strategy = {
                action: (1.0 - epsilon) * strategy[action] + epsilon * uniform
                for action in legal_actions
            }
        return strategy

    def add_average(self, strategy: Dict[str, float], weight: float) -> None:
        self.visits += 1
        if weight <= 0.0:
            return
        for action, prob in strategy.items():
            self.strategy_sum[action] = self.strategy_sum.get(action, 0.0) + weight * prob

    def average_strategy(self) -> Dict[str, float]:
        total = sum(max(v, 0.0) for v in self.strategy_sum.values())
        if total > 1e-12:
            return {action: max(v, 0.0) / total for action, v in self.strategy_sum.items()}
        if not self.strategy_sum:
            return {}
        p = 1.0 / len(self.strategy_sum)
        return {action: p for action in self.strategy_sum}


class CFRTrainer:
    def __init__(
        self,
        recall: str = "compact",
        seed: int = 1,
        explore: float = 0.02,
        rollouts_per_action: int = 16,
        updates_per_trajectory: int = 8,
        bust_penalty: float = 0.08,
        card_loss_penalty: float = 0.02,
        point_reward: float = 0.10,
    ) -> None:
        self.recall = recall
        self.rng = random.Random(seed)
        self.explore = explore
        self.rollouts_per_action = rollouts_per_action
        self.updates_per_trajectory = updates_per_trajectory
        self.bust_penalty = bust_penalty
        self.card_loss_penalty = card_loss_penalty
        self.point_reward = point_reward
        self.nodes: Dict[str, InfoSet] = {}
        self.iterations = 0

    def train(self, iterations: int, log_every: int = 0) -> None:
        start = time.perf_counter()
        for i in range(1, iterations + 1):
            self.sampled_cfr(
                State(), updating_player=0, updates_left=self.updates_per_trajectory
            )
            self.sampled_cfr(
                State(), updating_player=1, updates_left=self.updates_per_trajectory
            )
            self.iterations += 1
            if log_every and i % log_every == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"iter={i} infosets={len(self.nodes)} "
                    f"elapsed={elapsed:.2f}s it/s={i / max(elapsed, 1e-9):.1f}"
                )

    def sampled_cfr(self, state: State, updating_player: int, updates_left: int) -> float:
        """Recursive sampled CFR update.

        Full tree traversal is too expensive here because a player can branch
        at many decision points across several rounds. This samples one match
        trajectory, but at each visited information set for the updating player
        it estimates every legal action by rollout and applies regret matching.
        """
        winner = state.winner()
        if winner is not None:
            return self.terminal_utility(winner, updating_player)

        if state.phase == "remove":
            outcomes = state.chance_outcomes()
            card = weighted_choice(self.rng, outcomes)
            child = state.apply_chance(card)
            return self.sampled_cfr(
                child, updating_player, updates_left
            ) + self.transition_reward(state, child, updating_player)

        legal = state.legal_actions()
        if not legal:
            raise RuntimeError(f"non-terminal state has no legal actions: {state}")

        actor = state.actor

        if actor == updating_player and updates_left > 0:
            key = state.info_key(actor, self.recall)
            node = self.nodes.setdefault(key, InfoSet())
            strategy = node.strategy(legal, epsilon=self.explore)
            node.add_average(strategy, 1.0)
            action_utils: Dict[str, float] = {}
            for action in legal:
                total = 0.0
                for _ in range(self.rollouts_per_action):
                    child = state.apply_action(action)
                    total += self.transition_reward(
                        state, child, updating_player
                    ) + self.rollout(child, updating_player)
                action_utils[action] = total / self.rollouts_per_action
            node_util = sum(strategy[action] * action_utils[action] for action in legal)
            for action in legal:
                node.regret_sum[action] += action_utils[action] - node_util
            sampled = weighted_choice(self.rng, list(strategy.items()))
            child = state.apply_action(sampled)
            return self.sampled_cfr(
                child, updating_player, updates_left - 1
            ) + self.transition_reward(state, child, updating_player)

        sampled = self.sample_existing_policy_action(state, legal)
        child = state.apply_action(sampled)
        return self.sampled_cfr(
            child, updating_player, updates_left
        ) + self.transition_reward(state, child, updating_player)

    def rollout(self, state: State, perspective: int) -> float:
        steps = 0
        total_reward = 0.0
        while True:
            winner = state.winner()
            if winner is not None:
                return total_reward + self.terminal_utility(winner, perspective)
            steps += 1
            if steps > 256:
                raise RuntimeError(f"rollout exceeded step limit: {state}")

            if state.phase == "remove":
                card = weighted_choice(self.rng, state.chance_outcomes())
                child = state.apply_chance(card)
                total_reward += self.transition_reward(state, child, perspective)
                state = child
                continue

            legal = state.legal_actions()
            action = self.sample_existing_policy_action(state, legal)
            child = state.apply_action(action)
            total_reward += self.transition_reward(state, child, perspective)
            state = child

    def terminal_utility(self, winner: int, perspective: int) -> float:
        return 1.0 if winner == perspective else -1.0

    def transition_reward(self, before: State, after: State, perspective: int) -> float:
        opponent = other(perspective)
        reward = 0.0

        score_delta = (
            after.scores[perspective]
            - before.scores[perspective]
            - after.scores[opponent]
            + before.scores[opponent]
        )
        reward += self.point_reward * score_delta

        if before.phase != "remove" and after.phase == "remove":
            reward += (
                -self.bust_penalty
                if after.pending_loser == perspective
                else self.bust_penalty
            )

        card_delta = (
            after.total_cards(perspective)
            - before.total_cards(perspective)
            - after.total_cards(opponent)
            + before.total_cards(opponent)
        )
        reward += self.card_loss_penalty * card_delta
        return reward

    def sample_existing_policy_action(self, state: State, legal: Sequence[str]) -> str:
        key = state.info_key(state.actor, self.recall)
        node = self.nodes.get(key)
        if node is None:
            return fallback_action(state, self.rng)
        strategy = node.strategy(legal, epsilon=self.explore)
        return weighted_choice(self.rng, list(strategy.items()))

    def average_policy(self) -> Dict[str, Dict[str, float]]:
        return {
            key: node.average_strategy()
            for key, node in self.nodes.items()
            if node.average_strategy()
        }

    def save_policy(self, path: Path) -> None:
        payload = {
            "game": "simplified-two-player-skull",
            "algorithm": "sampled-recursive-infoset-cfr",
            "iterations": self.iterations,
            "recall": self.recall,
            "explore": self.explore,
            "rollouts_per_action": self.rollouts_per_action,
            "updates_per_trajectory": self.updates_per_trajectory,
            "bust_penalty": self.bust_penalty,
            "card_loss_penalty": self.card_loss_penalty,
            "point_reward": self.point_reward,
            "infosets": len(self.nodes),
            "policy": self.average_policy(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def load_policy(path: Path) -> Tuple[Dict[str, Dict[str, float]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["policy"], payload.get("recall", "compact")


def policy_action(
    state: State,
    policy: Dict[str, Dict[str, float]],
    recall: str,
    rng: random.Random,
    greedy: bool = False,
) -> str:
    legal = state.legal_actions()
    key = state.info_key(state.actor, recall)
    probs = policy.get(key)
    if probs:
        legal_probs = [(action, probs.get(action, 0.0)) for action in legal]
        total = sum(prob for _, prob in legal_probs)
        if total > 1e-12:
            if greedy:
                return max(legal_probs, key=lambda item: item[1])[0]
            return weighted_choice(rng, [(action, prob / total) for action, prob in legal_probs])
    return fallback_action(state, rng)


def fallback_action(state: State, rng: random.Random) -> str:
    legal = state.legal_actions()
    if state.phase == "bid":
        raises = [action for action in legal if is_bid(action)]
        if raises and rng.random() < 0.35:
            return raises[0]
        return PASS

    bids = [action for action in legal if is_bid(action)]
    plays = [action for action in legal if action in (PLAY_FLOWER, PLAY_SKULL)]
    if bids and (not plays or rng.random() < 0.18):
        return bids[0]
    if PLAY_FLOWER in plays and (PLAY_SKULL not in plays or rng.random() < 0.75):
        return PLAY_FLOWER
    if PLAY_SKULL in plays:
        return PLAY_SKULL
    return legal[0]


def play_game(
    policy: Dict[str, Dict[str, float]],
    recall: str,
    rng: random.Random,
    greedy: bool = False,
    trace: bool = False,
) -> int:
    state = State()
    while state.winner() is None:
        if state.phase == "remove":
            card = weighted_choice(rng, state.chance_outcomes())
            if trace:
                print(f"chance removes one hidden card from P{state.pending_loser}")
            state = state.apply_chance(card)
            continue

        action = policy_action(state, policy, recall, rng, greedy=greedy)
        if trace:
            print(state.public_view())
            print(f"P{state.actor} -> {action}")
        state = state.apply_action(action)
    if trace:
        print(state.public_view())
        print(f"winner P{state.winner()}")
    return int(state.winner())


def evaluate_policy(
    policy: Dict[str, Dict[str, float]],
    recall: str,
    games: int,
    seed: int,
    greedy: bool = False,
) -> Tuple[int, int]:
    rng = random.Random(seed)
    wins = [0, 0]
    for _ in range(games):
        winner = play_game(policy, recall, rng, greedy=greedy)
        wins[winner] += 1
    return wins[0], wins[1]


def interactive_game(
    policy: Dict[str, Dict[str, float]], recall: str, human: int, seed: int
) -> None:
    rng = random.Random(seed)
    state = State()
    print(f"You are P{human}. Cards are shown only for your own hand and pile.")
    while state.winner() is None:
        if state.phase == "remove":
            card = weighted_choice(rng, state.chance_outcomes())
            loser = state.pending_loser
            state = state.apply_chance(card)
            if loser == human:
                print(f"You lost one {card}.")
            else:
                print(f"P{loser} lost one hidden card.")
            continue

        print()
        print(state.public_view())
        print(f"your hand={state.hands[human]} your pile={''.join(state.piles[human]) or '-'}")
        legal = state.legal_actions()
        if state.actor == human:
            for idx, action in enumerate(legal, 1):
                print(f"{idx}. {action}")
            choice = read_choice(len(legal))
            action = legal[choice - 1]
        else:
            action = policy_action(state, policy, recall, rng, greedy=False)
            print(f"CPU P{state.actor} -> {action}")
        state = state.apply_action(action)
    print()
    print(state.public_view())
    print("You win." if state.winner() == human else "CPU wins.")


def read_choice(max_choice: int) -> int:
    while True:
        raw = input("> ").strip()
        if raw.isdigit():
            value = int(raw)
            if 1 <= value <= max_choice:
                return value
        print(f"Enter a number from 1 to {max_choice}.")


def print_root_policy(policy: Dict[str, Dict[str, float]], recall: str) -> None:
    root = State()
    key = root.info_key(0, recall)
    probs = policy.get(key, {})
    if not probs:
        print("root policy is unavailable")
        return
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    print("root P0 policy:", ", ".join(f"{a}={p:.3f}" for a, p in ranked))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="train an average strategy with MCCFR")
    train.add_argument("--iterations", type=int, default=10_000)
    train.add_argument("--out", type=Path, default=Path("skull_policy.json"))
    train.add_argument("--recall", choices=("compact", "perfect"), default="compact")
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--explore", type=float, default=0.02)
    train.add_argument("--rollouts-per-action", type=int, default=16)
    train.add_argument("--updates-per-trajectory", type=int, default=8)
    train.add_argument("--log-every", type=int, default=0)
    train.add_argument("--eval-games", type=int, default=2_000)

    battle = sub.add_parser("battle", help="run CPU self-play from a saved policy")
    battle.add_argument("--policy", type=Path, default=Path("skull_policy.json"))
    battle.add_argument("--games", type=int, default=1_000)
    battle.add_argument("--seed", type=int, default=2)
    battle.add_argument("--greedy", action="store_true")
    battle.add_argument("--trace", action="store_true")

    play = sub.add_parser("play", help="play against the trained CPU")
    play.add_argument("--policy", type=Path, default=Path("skull_policy.json"))
    play.add_argument("--human", type=int, choices=(0, 1), default=0)
    play.add_argument("--seed", type=int, default=3)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "train":
        trainer = CFRTrainer(
            recall=args.recall,
            seed=args.seed,
            explore=args.explore,
            rollouts_per_action=max(1, args.rollouts_per_action),
            updates_per_trajectory=max(1, args.updates_per_trajectory),
        )
        start = time.perf_counter()
        trainer.train(args.iterations, log_every=args.log_every)
        elapsed = time.perf_counter() - start
        trainer.save_policy(args.out)
        policy = trainer.average_policy()
        p0, p1 = evaluate_policy(
            policy, args.recall, games=args.eval_games, seed=args.seed + 10_000
        )
        print(
            f"trained {args.iterations} iterations in {elapsed:.2f}s | "
            f"infosets={len(trainer.nodes)} | saved={args.out}"
        )
        print(f"self-play eval over {args.eval_games} games: P0 {p0} - P1 {p1}")
        print_root_policy(policy, args.recall)
        return 0

    if args.cmd == "battle":
        policy, recall = load_policy(args.policy)
        if args.trace:
            winner = play_game(
                policy, recall, random.Random(args.seed), greedy=args.greedy, trace=True
            )
            print(f"winner P{winner}")
            return 0
        p0, p1 = evaluate_policy(
            policy, recall, games=args.games, seed=args.seed, greedy=args.greedy
        )
        print(f"CPU self-play over {args.games} games: P0 {p0} - P1 {p1}")
        return 0

    if args.cmd == "play":
        policy, recall = load_policy(args.policy)
        interactive_game(policy, recall, human=args.human, seed=args.seed)
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
