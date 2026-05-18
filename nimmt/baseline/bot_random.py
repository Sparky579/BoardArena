import random


class Bot:
    name = "random_nimmt"

    def choose_action(self, state):
        return random.choice(state["legal_actions"])
