"""Stable entry point for the current general anti-fork bot."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_CORE_PATH = Path(__file__).resolve().parent / "gpt-hard-anti-fork" / "bot.py"
_SPEC = importlib.util.spec_from_file_location("_lqq_antifork_core", _CORE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot import anti-fork core: {_CORE_PATH}")
_CORE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CORE
_SPEC.loader.exec_module(_CORE)


def choose_action(state):
    return Bot().choose_action(state)


class Bot:
    name = "bot_v3"

    def __init__(self):
        self.core = _CORE.Bot()

    def choose_action(self, state):
        legal = list(state.get("legal_actions", ()))
        if not legal:
            return ""
        action = self.core.choose_action(state)
        if action in legal:
            return action
        return legal[0]
