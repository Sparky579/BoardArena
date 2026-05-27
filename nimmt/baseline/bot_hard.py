import random
import time


TIME_BUDGET = 0.28
SAFETY_MARGIN_SECONDS = 0.15
DETERMINIZATIONS = 60
ROLLOUT_DEPTH = 10

_BULL_CACHE: dict[int, int] = {}


def _bull_count(card):
    if card in _BULL_CACHE:
        return _BULL_CACHE[card]
    if card == 55:
        v = 7
    elif card % 11 == 0:
        v = 5
    elif card % 10 == 0:
        v = 3
    elif card % 5 == 0:
        v = 2
    else:
        v = 1
    _BULL_CACHE[card] = v
    return v


def _row_bulls(row):
    return sum(_bull_count(c) for c in row)


def _parse_action(action):
    parts = action.split("_")
    card = int(parts[1])
    take_row = int(parts[3]) if len(parts) == 4 else None
    return card, take_row


def _choose_greedy(state):
    rows = state["rows"]
    row_bulls = state["row_bulls"]
    legal = state["legal_actions"]

    def _score(action):
        card, take_row = _parse_action(action)
        if len(rows) < 4:
            return (0, card)
        if take_row is not None:
            return (row_bulls[take_row], card)
        candidates = [(row[-1], i) for i, row in enumerate(rows) if row[-1] < card]
        if not candidates:
            return (99, card)
        _, ri = max(candidates)
        imm = row_bulls[ri] if len(rows[ri]) >= 5 else 0
        return (imm, len(rows[ri]), card)

    return min(legal, key=_score)


def _time_budget(state):
    timeout = state.get("decision_timeout") or state.get("time_limit")
    if timeout:
        return max(0.05, float(timeout) - SAFETY_MARGIN_SECONDS)
    return TIME_BUDGET


class Bot:
    name = "hard"

    def choose_action(self, state):
        legal = state["legal_actions"]
        if not legal:
            raise ValueError("no legal actions")
        if len(legal) == 1:
            return legal[0]

        deadline = time.perf_counter() + _time_budget(state)

        my_hand = list(state["hand"])
        hand_sizes = list(state["hand_sizes"])
        my_id = state["player_id"]
        num_players = state["num_players"]

        seen = set(my_hand)
        for row in state["rows"]:
            for c in row:
                seen.add(c)
        deck_max = state["deck_max"]
        pool = [c for c in range(1, deck_max + 1) if c not in seen]

        my_card_set = set(my_hand)
        scores_accum = {action: 0.0 for action in legal}
        det_count = 0
        rng = random.Random(time.perf_counter_ns() & 0xFFFFFFFF)

        while det_count < DETERMINIZATIONS:
            if time.perf_counter() >= deadline:
                break

            opp_hands = _sample_opponent_hands(rng, pool.copy(), hand_sizes, my_id, my_card_set)
            value = _evaluate_determinization(
                legal, my_id, num_players, state["rows"], state["scores"],
                state["turn"], opp_hands, rng
            )
            for action in legal:
                scores_accum[action] += value.get(action, 0.0)
            det_count += 1

        if det_count == 0:
            return _choose_greedy(state)

        best = None
        best_score = float("-inf")
        for action in legal:
            avg = scores_accum[action] / det_count
            if avg > best_score:
                best_score = avg
                best = action

        return best


def _sample_opponent_hands(rng, pool, hand_sizes, my_id, my_card_set):
    n = len(hand_sizes)
    rng.shuffle(pool)
    opp_hands = []
    idx = 0
    for p in range(n):
        if p == my_id:
            opp_hands.append(None)
        else:
            k = hand_sizes[p]
            hand = pool[idx:idx + k]
            idx += k
            opp_hands.append(set(hand))
    return opp_hands


def _evaluate_determinization(legal_actions, my_id, num_players, rows, scores, turn, opp_hands, rng):
    action_values = {}

    sim_rows = [[c for c in row] for row in rows]
    sim_scores = list(scores)
    card_set = set()

    for action in legal_actions:
        card, _ = _parse_action(action)
        card_set.add(card)

    all_hands = []
    for p in range(num_players):
        if p == my_id:
            all_hands.append(card_set.copy())
        else:
            all_hands.append(opp_hands[p].copy())

    for depth in range(ROLLOUT_DEPTH):
        if not any(all_hands):
            break

        revealed = []
        for p in range(num_players):
            hand = all_hands[p]
            if not hand:
                continue
            card = rng.choice(sorted(hand))
            hand.discard(card)

            if len(sim_rows) < 4:
                revealed.append((p, card, None))
                continue

            min_top = min(r[-1] for r in sim_rows)
            if card < min_top:
                best_ri = min(range(len(sim_rows)), key=lambda i: _row_bulls(sim_rows[i]))
                revealed.append((p, card, best_ri))
            else:
                candidates = [(r[-1], i) for i, r in enumerate(sim_rows) if r[-1] < card]
                if not candidates:
                    best_ri = min(range(len(sim_rows)), key=lambda i: _row_bulls(sim_rows[i]))
                    revealed.append((p, card, best_ri))
                else:
                    revealed.append((p, card, None))

        _apply_revealed(sim_rows, sim_scores, sorted(revealed, key=lambda x: x[1]))

    best_score = min(sim_scores)
    winners = [i for i, s in enumerate(sim_scores) if s == best_score]

    for action in legal_actions:
        if my_id in winners:
            action_values[action] = 1.0
        elif len(winners) == num_players:
            action_values[action] = 0.0
        else:
            action_values[action] = -1.0 / (num_players - 1)

    return action_values


def _apply_revealed(rows, scores, revealed):
    for player, card, take_row in revealed:
        if len(rows) < 4:
            rows.append([card])
            continue

        candidates = [(r[-1], i) for i, r in enumerate(rows) if r[-1] < card]
        if not candidates:
            if take_row is not None and 0 <= take_row < len(rows):
                scores[player] += _row_bulls(rows[take_row])
                rows[take_row] = [card]
            continue

        _, ri = max(candidates)
        if len(rows[ri]) >= 5:
            scores[player] += _row_bulls(rows[ri])
            rows[ri] = [card]
        else:
            rows[ri].append(card)
