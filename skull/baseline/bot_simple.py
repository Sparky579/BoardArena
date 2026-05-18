class Bot:
    name = "simple"

    def choose_action(self, state):
        legal = state["legal_actions"]
        if state["phase"] == "challenge":
            return legal[0]
        if "PLAY_F" in legal:
            return "PLAY_F"
        bids = [action for action in legal if action.startswith("BID_")]
        if bids:
            return bids[0]
        return "PASS" if "PASS" in legal else legal[0]
