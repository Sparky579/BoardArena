"""Shared Stockfish-UCI-engine driver for the ``stockfish_<elo>.py`` bots.

The thin per-Elo wrappers in this directory just call ``make_bot(elo)``
with their target Elo. Everything shared — finding the binary, spawning
the engine once per process, plumbing the per-move time limit — lives
here so calibrating a new Elo level is a one-line file.

Calibration notes (from Stockfish docs):

  * ``UCI_LimitStrength`` + ``UCI_Elo`` is calibrated against ~1s per
    move. Longer time controls overshoot the requested Elo by some
    margin, shorter undershoot. For Elo *comparison* against another
    bot, the cleanest setup is to give both bots the same time/move.

  * ``UCI_Elo`` accepts 1320–3190 (continuous integer). Skill Level is
    a coarser 0–20 alternative.

Environment overrides:

  ``STOCKFISH_PATH``  — path to the ``stockfish`` binary
                        (default: ``stockfish`` on PATH, then
                        ``~/.local/bin/stockfish``).
  ``STOCKFISH_TIME``  — seconds per move (default: 1.0).
  ``STOCKFISH_THREADS`` — UCI ``Threads`` option (default: 1).
  ``STOCKFISH_HASH``  — UCI ``Hash`` MB (default: 16).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import chess
import chess.engine


@contextlib.contextmanager
def _daemon_thread_default():
    """Force ``threading.Thread`` to spawn with daemon=True for the
    duration of this context. python-chess's SimpleEngine spawns a
    *non-daemon* background asyncio thread to drive UCI; if the caller
    forgets to call ``engine.quit()`` before exit, the interpreter
    waits forever on that thread. Using this around ``popen_uci`` makes
    the asyncio thread daemon so plain ``sys.exit()`` is fine —
    Stockfish notices the pipe close and shuts down cleanly.
    """
    orig = threading.Thread.__init__

    def patched(self, *args, **kwargs):
        if kwargs.get("daemon") is None:
            kwargs["daemon"] = True
        orig(self, *args, **kwargs)

    threading.Thread.__init__ = patched
    try:
        yield
    finally:
        threading.Thread.__init__ = orig


def _find_stockfish() -> str:
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        return env
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    user_local = Path.home() / ".local" / "bin" / "stockfish"
    if user_local.is_file():
        return str(user_local)
    raise FileNotFoundError(
        "Stockfish binary not found. Set STOCKFISH_PATH or place the "
        "binary on PATH (e.g. ~/.local/bin/stockfish)."
    )


# One engine process per python process, cached lazily so importing the
# module doesn't spawn anything until the first move.
_ENGINE: chess.engine.SimpleEngine | None = None


def _engine() -> chess.engine.SimpleEngine:
    """Spawn (or return cached) Stockfish process. The SimpleEngine
    background thread is born daemonised so plain ``sys.exit()`` works
    without explicit ``engine.quit()``.
    """
    global _ENGINE
    if _ENGINE is None:
        path = _find_stockfish()
        with _daemon_thread_default():
            _ENGINE = chess.engine.SimpleEngine.popen_uci(path)
        _ENGINE.configure({
            "Threads": int(os.environ.get("STOCKFISH_THREADS", "1")),
            "Hash": int(os.environ.get("STOCKFISH_HASH", "16")),
        })
    return _ENGINE


def quit_engine() -> None:
    """Optional explicit shutdown for hosts that want a clean teardown."""
    global _ENGINE
    if _ENGINE is not None:
        try:
            _ENGINE.quit()
        except Exception:  # noqa: BLE001 - engine already gone is fine.
            pass
        _ENGINE = None


def make_bot(elo: int) -> type:
    """Build a ``Bot`` class pinned to a specific UCI_Elo. The per-Elo
    wrapper files just do ``Bot = make_bot(1500)``.
    """
    move_time = float(os.environ.get("STOCKFISH_TIME", "1.0"))
    limit = chess.engine.Limit(time=move_time)

    class _StockfishBot:
        name = f"stockfish_{elo}"

        def __init__(self) -> None:
            engine = _engine()
            # Reconfiguring per-instance is fine — Stockfish honors the
            # last setting. Sharing one process across multiple loaded
            # Elo bots in the *same* duel would clobber each other; we
            # don't do that, but if someone runs e.g. round-robin in the
            # same python process, see _per_call below.
            engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": elo,
            })
            self._elo = elo

        def choose_action(self, state: dict[str, Any]) -> str:
            engine = _engine()
            # Re-pin the Elo on every move so a previously loaded bot
            # at a different Elo in the same process can't influence us.
            engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": self._elo,
            })
            board = chess.Board(state["fen"])
            result = engine.play(board, limit)
            move = result.move
            uci = move.uci() if move is not None else ""
            legal = state.get("legal_actions") or []
            if uci in legal:
                return uci
            # Engine returned an unexpected promotion shorthand or null
            # move — fall back to any legal action so we don't forfeit.
            return legal[0] if legal else ""

    _StockfishBot.__name__ = f"StockfishBot{elo}"
    return _StockfishBot
