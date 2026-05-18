class Bot:
    name = "greedy_lqq"

    def choose_action(self, state):
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
