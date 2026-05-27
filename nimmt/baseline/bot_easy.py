def _parse_action(action):
    parts = action.split("_")
    card = int(parts[1])
    take_row = int(parts[3]) if len(parts) == 4 else None
    return card, take_row


class Bot:
    name = "easy"

    def choose_action(self, state):
        rows = state["rows"]
        row_bulls = state["row_bulls"]

        def score(action):
            card, take_row = _parse_action(action)
            if len(rows) < 4:
                return (0, card)

            if take_row is not None:
                return (row_bulls[take_row], card)

            candidates = [
                (row[-1], row_index)
                for row_index, row in enumerate(rows)
                if row[-1] < card
            ]
            if not candidates:
                return (99, card)

            _, row_index = max(candidates)
            immediate_bulls = row_bulls[row_index] if len(rows[row_index]) >= 5 else 0
            crowded_row_penalty = len(rows[row_index])
            return (immediate_bulls, crowded_row_penalty, card)

        return min(state["legal_actions"], key=score)
