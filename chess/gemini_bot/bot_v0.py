import random

def choose_action(state):
    return random.choice(state["legal_actions"])
