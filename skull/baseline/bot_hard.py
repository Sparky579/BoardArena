import json
import os
import random

from pathlib import Path


_HERE = Path(__file__).resolve().parent
_POLICY_FILE = os.environ.get("BOARDARENA_SKULL_POLICY", str(_HERE.parent.parent / "skull_policy.json"))

_POLICY: dict[str, dict[str, float]] | None = None


def _load_policy():
    global _POLICY
    if _POLICY is not None:
        return _POLICY
    path = Path(_POLICY_FILE)
    if not path.exists():
        _POLICY = {}
        return _POLICY
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    _POLICY = data.get("policy", {})
    return _POLICY


def _format_pile(pile):
    return "".join(pile)


def _build_policy_key(state):
    player_id = state["player_id"]
    opp_id = 1 - player_id
    phase = state["phase"]

    scores = state["scores"]
    own_score = scores[player_id]
    opp_score = scores[opp_id]

    hand = state["hand"]
    flowers = hand["flowers"]
    skulls = hand["skulls"]

    own_pile = state["own_pile"]
    pile_str = _format_pile(own_pile)

    opp_cards = state["total_cards"][opp_id]
    opp_pile = state["pile_sizes"][opp_id]

    turn = "me" if state["actor"] == player_id else "opp"

    key_parts = [
        f"p={player_id}",
        f"score={own_score}-{opp_score}",
        f"hand={flowers},{skulls}",
        f"pile={pile_str}",
        f"opp_cards={opp_cards}",
        f"opp_pile={opp_pile}",
        f"phase={phase}",
    ]

    if phase == "bid":
        key_parts.append(f"bid={state['current_bid']}")
        high = state["high_bidder"]
        if high == player_id:
            key_parts.append("high=me")
        elif high == opp_id:
            key_parts.append("high=opp")
        else:
            key_parts.append("high=none")
        key_parts.append(f"turn={turn}")

    return "|".join(key_parts)


def _choose_policy(state):
    policy = _load_policy()
    key = _build_policy_key(state)

    action_probs = policy.get(key)
    if action_probs is None:
        return None

    legal = set(state["legal_actions"])
    valid = {a: p for a, p in action_probs.items() if a in legal}
    if not valid:
        return None

    total = sum(valid.values())
    if total <= 0:
        return None

    target = random.random() * total
    cumulative = 0.0
    for action, prob in valid.items():
        cumulative += prob
        if cumulative >= target:
            return action

    return max(valid, key=valid.get)


def _choose_heuristic(state):
    legal = state["legal_actions"]
    if state["phase"] == "challenge":
        return legal[0]
    if "PLAY_F" in legal:
        return "PLAY_F"
    bids = [a for a in legal if a.startswith("BID_")]
    if bids:
        return bids[0]
    return "PASS" if "PASS" in legal else legal[0]


class Bot:
    name = "hard"

    def choose_action(self, state):
        if state["num_players"] == 2:
            action = _choose_policy(state)
            if action is not None:
                return action
        return _choose_heuristic(state)
