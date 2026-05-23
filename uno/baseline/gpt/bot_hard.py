"""MCTS baseline UNO bot for BoardArena/uno.

The bot uses determinized Monte Carlo search: each simulation samples plausible
hidden opponent cards and draw-pile order from the public state, applies a root
candidate action, then completes the game with a fast rollout policy.
"""

from __future__ import annotations

import math
import random
import time


name = "gpt_mcts_hard"

COLORS = ("red", "yellow", "green", "blue")
ACTION_KINDS = {"skip", "reverse", "draw_two"}
WILD_KINDS = {"wild", "wild_draw_four"}
DRAW_ACTION = "draw"
PASS_ACTION = "pass"

TIME_BUDGET = 0.28
SAFETY_MARGIN_SECONDS = 0.15
MAX_ITERATIONS = 420
ROLLOUT_DEPTH = 82
EXPLORATION = 0.72


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        raise ValueError("no legal actions")
    if len(legal) == 1:
        return legal[0]
    if DRAW_ACTION in legal:
        return DRAW_ACTION
    if PASS_ACTION in legal:
        return PASS_ACTION

    winning = _immediate_winning_actions(state)
    if winning:
        return _best_tactical_action(state, winning)

    forcing = _forcing_root_actions(state, legal)
    if forcing:
        legal = forcing
        if len(legal) == 1:
            return legal[0]

    rng = random.Random(_state_seed(state))
    priors = {action: _root_prior(state, action) for action in legal}
    visits = {action: 0 for action in legal}
    values = {action: 0.0 for action in legal}
    total_visits = 0
    deadline = time.perf_counter() + _time_budget(state)
    max_iterations = _iteration_budget(state, legal)

    for _ in range(max_iterations):
        if time.perf_counter() >= deadline:
            break
        action = _select_ucb_action(legal, visits, values, priors, total_visits)
        sim = _sample_simulation(state, rng)
        if action not in sim.legal_actions():
            value = -1.0
        else:
            sim.step(action)
            value = _rollout(sim, int(state["player_id"]), rng)
        visits[action] += 1
        values[action] += value
        total_visits += 1

    if total_visits == 0:
        return _best_tactical_action(state, legal)

    ranked = []
    for action in legal:
        average = values[action] / visits[action] if visits[action] else -2.0
        combined = 0.08 * average + 1.15 * priors[action]
        ranked.append((combined, average, visits[action], priors[action], _stable_tie_key(state, action), action))
    ranked.sort(reverse=True)
    return ranked[0][5]


def _iteration_budget(state, legal):
    if state["opponent_hand_count"] <= 2 or state["hand_count"] <= 3:
        return MAX_ITERATIONS
    if len(legal) >= 10:
        return 190
    return 225


def _select_ucb_action(legal, visits, values, priors, total_visits):
    unvisited = [action for action in legal if visits[action] == 0]
    if unvisited:
        return max(unvisited, key=lambda action: (priors[action], action))

    log_total = math.log(total_visits + 1)
    best_action = legal[0]
    best_score = -10**9
    for action in legal:
        average = values[action] / visits[action]
        explore = EXPLORATION * math.sqrt(log_total / visits[action])
        prior = 0.25 * priors[action]
        score = average + explore + prior
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _rollout(sim, perspective, rng):
    for _ in range(ROLLOUT_DEPTH):
        if sim.is_done():
            break
        legal = sim.legal_actions()
        if not legal:
            break
        action = _rollout_action(sim, legal, rng)
        sim.step(action)
    return sim.value(perspective)


def _rollout_action(sim, legal, rng):
    if len(legal) == 1:
        return legal[0]
    if DRAW_ACTION in legal:
        return DRAW_ACTION
    if PASS_ACTION in legal:
        return PASS_ACTION

    scored = []
    for action in legal:
        card = sim.card_for_action(action)
        score = _sim_action_score(sim, card, action)
        scored.append((score, action))
    scored.sort(reverse=True)
    if rng.random() < 0.86:
        return scored[0][1]
    return rng.choice(scored[: min(3, len(scored))])[1]


def _sample_simulation(state, rng):
    me = int(state["player_id"])
    top_card = _card_from_view(state["top_card"])
    known_ids = {int(card["id"]) for card in state["hand"]}
    known_ids.add(int(top_card["id"]))

    unknown = [_copy_card(card) for card in FULL_DECK if int(card["id"]) not in known_ids]
    rng.shuffle(unknown)

    opponent_count = int(state["opponent_hand_count"])
    draw_count = int(state["draw_pile_count"])
    discard_extra_count = max(0, int(state["discard_pile_count"]) - 1)

    opponent_hand = unknown[:opponent_count]
    draw_pile = unknown[opponent_count:opponent_count + draw_count]
    discard_extra = unknown[
        opponent_count + draw_count:opponent_count + draw_count + discard_extra_count
    ]

    hands = [[], []]
    hands[me] = [_card_from_view(card) for card in state["hand"]]
    hands[1 - me] = opponent_hand
    consecutive_passes = _public_pass_tail(state.get("history", []))
    return SimState(
        hands=hands,
        draw_pile=draw_pile,
        discard_pile=discard_extra + [top_card],
        actor=int(state["actor"]),
        current_color=str(state["current_color"]),
        consecutive_passes=consecutive_passes,
        rng=rng,
    )


class SimState:
    def __init__(
        self,
        *,
        hands,
        draw_pile,
        discard_pile,
        actor,
        current_color,
        consecutive_passes,
        rng,
    ):
        self.hands = hands
        self.draw_pile = draw_pile
        self.discard_pile = discard_pile
        self.actor = actor
        self.current_color = current_color
        self.consecutive_passes = consecutive_passes
        self.rng = rng
        self.winner = None
        self.status = None

    def legal_actions(self):
        if self.is_done():
            return []
        top = self.discard_pile[-1]
        actions = []
        for card in sorted(self.hands[self.actor], key=_card_sort_key):
            if not _is_playable(card, top, self.current_color):
                continue
            if card["kind"] in WILD_KINDS:
                actions.extend(f"play:{card['id']}:{color}" for color in COLORS)
            else:
                actions.append(f"play:{card['id']}")
        if actions:
            return actions
        if self.draw_pile or len(self.discard_pile) > 1:
            return [DRAW_ACTION]
        return [PASS_ACTION]

    def step(self, action):
        actor = self.actor
        if action == DRAW_ACTION:
            drawn = self._draw_cards(actor, 1)
            self.consecutive_passes = 0 if drawn else self.consecutive_passes + 1
            self.actor = 1 - actor
        elif action == PASS_ACTION:
            self.consecutive_passes += 1
            self.actor = 1 - actor
        else:
            card, chosen_color = self._parse_play(action)
            self.hands[actor].remove(card)
            self.discard_pile.append(card)
            self.current_color = chosen_color or str(card["color"])
            self.consecutive_passes = 0
            self._apply_effect(card, actor)
            if not self.hands[actor]:
                self.winner = actor
                self.status = "empty_hand"

        if self.status is None and self.consecutive_passes >= 2:
            self.winner = self._leader_by_hand_size()
            self.status = "blocked"

    def card_for_action(self, action):
        card_id = int(action.split(":")[1])
        return next(card for card in self.hands[self.actor] if int(card["id"]) == card_id)

    def is_done(self):
        return self.status is not None

    def value(self, perspective):
        if self.is_done():
            if self.winner is None:
                return 0.0
            return 1.0 if self.winner == perspective else -1.0

        other = 1 - perspective
        own_count = len(self.hands[perspective])
        other_count = len(self.hands[other])
        count_score = 0.23 * (other_count - own_count)
        turn_score = 0.08 if self.actor == perspective else -0.08
        playable_score = 0.025 * (
            self._playable_count(perspective) - self._playable_count(other)
        )
        return _clamp(count_score + turn_score + playable_score, -0.95, 0.95)

    def _parse_play(self, action):
        parts = action.split(":")
        card_id = int(parts[1])
        card = next(card for card in self.hands[self.actor] if int(card["id"]) == card_id)
        chosen_color = parts[2] if len(parts) == 3 else None
        return card, chosen_color

    def _apply_effect(self, card, actor):
        target = 1 - actor
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

    def _draw_cards(self, player, count):
        drawn = 0
        for _ in range(count):
            if not self.draw_pile and not self._reshuffle_discard_into_draw():
                break
            self.hands[player].append(self.draw_pile.pop())
            drawn += 1
        return drawn

    def _reshuffle_discard_into_draw(self):
        if len(self.discard_pile) <= 1:
            return False
        top = self.discard_pile.pop()
        self.draw_pile = self.discard_pile
        self.rng.shuffle(self.draw_pile)
        self.discard_pile = [top]
        return bool(self.draw_pile)

    def _leader_by_hand_size(self):
        counts = [len(self.hands[0]), len(self.hands[1])]
        if counts[0] == counts[1]:
            return None
        return 0 if counts[0] < counts[1] else 1

    def _playable_count(self, player):
        top = self.discard_pile[-1]
        return sum(1 for card in self.hands[player] if _is_playable(card, top, self.current_color))


def _sim_action_score(sim, card, action):
    actor = sim.actor
    target = 1 - actor
    own_count = len(sim.hands[actor])
    target_count = len(sim.hands[target])
    kind = card["kind"]
    score = 30

    if own_count == 1:
        score += 10000
    if kind == "wild_draw_four":
        score += 175
    elif kind == "draw_two":
        score += 135
    elif kind in {"skip", "reverse"}:
        score += 105
    elif kind == "wild":
        score += 55
    elif kind == "number":
        score += 18 + int(card.get("value") or 0)

    if target_count <= 2 and kind in {"wild_draw_four", "draw_two", "skip", "reverse"}:
        score += 170
    elif target_count >= 8 and kind in {"wild_draw_four", "draw_two"}:
        score -= 35

    if kind in WILD_KINDS:
        chosen = action.split(":")[2]
        score += 22 * _color_count_after_play(sim.hands[actor], chosen, card)
    else:
        score += 9 * _color_count_after_play(sim.hands[actor], card.get("color"), card)
        if card.get("color") == sim.current_color:
            score += 8

    if kind == "wild" and own_count > 3 and target_count > 2:
        score -= 28
    return score


def _root_prior(state, action):
    if action in {DRAW_ACTION, PASS_ACTION}:
        return -1.0
    card = _card_for_root_action(state, action)
    raw = 0.0
    kind = card["kind"]
    opponent_count = int(state["opponent_hand_count"])
    own_count = int(state["hand_count"])

    if own_count == 1:
        raw += 200
    if kind == "wild_draw_four":
        raw += 190
    elif kind == "draw_two":
        raw += 152
    elif kind in {"skip", "reverse"}:
        raw += 126
    elif kind == "wild":
        raw += 78
    else:
        raw += 24 + int(card.get("value") or 0)

    if opponent_count <= 2 and kind in {"wild_draw_four", "draw_two", "skip", "reverse"}:
        raw += 145
    if kind in WILD_KINDS:
        raw += 18 * _root_color_count_after(state, action.split(":")[2], card)
        if own_count > 4 and _has_nonwild_play(state):
            raw -= 14
    else:
        if card.get("color") == state["current_color"]:
            raw += 10
    return _clamp(raw / 220.0, -1.0, 1.0)


def _best_tactical_action(state, legal):
    return max(legal, key=lambda action: (_root_prior(state, action), _stable_tie_key(state, action), action))


def _immediate_winning_actions(state):
    if int(state["hand_count"]) != 1:
        return []
    return [action for action in state["legal_actions"] if action.startswith("play:")]


def _forcing_root_actions(state, legal):
    by_priority = {}
    for action in legal:
        if not action.startswith("play:"):
            continue
        card = _card_for_root_action(state, action)
        priority = {
            "wild_draw_four": 5,
            "draw_two": 4,
            "skip": 3,
            "reverse": 3,
            "wild": 2,
        }.get(card["kind"], 0)
        if priority:
            by_priority.setdefault(priority, []).append(action)
    if not by_priority:
        return []
    return by_priority[max(by_priority)]


def _stable_tie_key(state, action):
    if action in {DRAW_ACTION, PASS_ACTION}:
        return (-100, action)
    card = _card_for_root_action(state, action)
    color = action.split(":")[2] if card["kind"] in WILD_KINDS else card.get("color", "")
    return (
        _root_color_count_after(state, color, card),
        _power_weight(card),
        int(card.get("value") or -1),
        -int(card["id"]),
    )


def _card_for_root_action(state, action):
    card_id = int(action.split(":")[1])
    return next(card for card in state["hand"] if int(card["id"]) == card_id)


def _has_nonwild_play(state):
    for action in state["legal_actions"]:
        if not action.startswith("play:"):
            continue
        card = _card_for_root_action(state, action)
        if card["kind"] not in WILD_KINDS:
            return True
    return False


def _root_color_count_after(state, color, played_card):
    if color is None:
        return 0
    return sum(
        1
        for card in state["hand"]
        if int(card["id"]) != int(played_card["id"]) and card["color"] == color
    )


def _color_count_after_play(hand, color, played_card):
    if color is None:
        return 0
    return sum(
        1
        for card in hand
        if int(card["id"]) != int(played_card["id"]) and card["color"] == color
    )


def _public_pass_tail(history):
    count = 0
    for item in reversed(history):
        if item.get("action") == PASS_ACTION:
            count += 1
        else:
            break
    return count


def _state_seed(state):
    parts = [
        str(state.get("player_id")),
        str(state.get("actor")),
        str(state.get("current_color")),
        str(state.get("draw_pile_count")),
        str(state.get("discard_pile_count")),
        str(state.get("plies")),
        str(state.get("opponent_hand_count")),
        str(state.get("top_card", {}).get("id")),
        ",".join(str(card["id"]) for card in state.get("hand", [])),
    ]
    value = 1469598103934665603
    for char in "|".join(parts):
        value ^= ord(char)
        value *= 1099511628211
        value &= (1 << 64) - 1
    return value


def _is_playable(card, top_card, current_color):
    if card["kind"] in WILD_KINDS:
        return True
    if card["color"] == current_color:
        return True
    if card["kind"] == "number" and top_card["kind"] == "number":
        return card["value"] == top_card["value"]
    return card["kind"] == top_card["kind"] and card["kind"] in ACTION_KINDS


def _power_weight(card):
    kind = card["kind"]
    if kind == "wild_draw_four":
        return 6
    if kind == "draw_two":
        return 5
    if kind in {"skip", "reverse"}:
        return 4
    if kind == "wild":
        return 3
    return 1


def _card_sort_key(card):
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


def _card_from_view(card):
    return {
        "id": int(card["id"]),
        "color": card["color"],
        "kind": card["kind"],
        "value": card["value"],
    }


def _copy_card(card):
    return {
        "id": int(card["id"]),
        "color": card["color"],
        "kind": card["kind"],
        "value": card["value"],
    }


def _clamp(value, low, high):
    return max(low, min(high, value))


def _time_budget(state):
    timeout = state.get("decision_timeout") or state.get("time_limit")
    if timeout:
        return max(0.05, float(timeout) - SAFETY_MARGIN_SECONDS)
    return TIME_BUDGET


def _build_deck():
    deck = []
    card_id = 0

    def add(color, kind, value=None, copies=1):
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


FULL_DECK = _build_deck()
