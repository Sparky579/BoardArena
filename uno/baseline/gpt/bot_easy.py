"""Easy baseline UNO bot for BoardArena/uno."""

from __future__ import annotations


name = "gpt_easy"

POWER_ORDER = {
    "wild_draw_four": 5,
    "draw_two": 4,
    "skip": 3,
    "reverse": 3,
    "wild": 2,
    "number": 1,
}


def choose_action(state):
    legal = state["legal_actions"]
    if not legal:
        raise ValueError("no legal actions")
    if "draw" in legal:
        return "draw"
    if "pass" in legal:
        return "pass"

    scored = []
    for action in legal:
        card = _card_for_action(state, action)
        score = POWER_ORDER.get(card["kind"], 0)
        if card["kind"] in {"wild", "wild_draw_four"}:
            score += _color_count(state["hand"], action.split(":")[2])
        scored.append((score, action))

    best = max(score for score, _ in scored)
    return sorted(action for score, action in scored if score == best)[0]


def _card_for_action(state, action):
    card_id = int(action.split(":")[1])
    return next(card for card in state["hand"] if card["id"] == card_id)


def _color_count(hand, color):
    return sum(1 for card in hand if card["color"] == color)

