"""Shared Edax driver for the ``bot_edax_<elo>.py`` Othello bots.

Edax (https://github.com/abulmo/edax-reversi) is the strongest open
source Othello engine. Unlike Stockfish for chess it has no clean
``UCI_Elo`` knob — strength is controlled by search level (depth). The
per-Elo wrapper files just pick an approximate ``level`` and pass it to
``make_bot(elo, level)`` here.

Approximate Edax level → Elo at 1s/move (community reports, rough):

    level  1   ≈ 1300
    level  3   ≈ 1500
    level  5   ≈ 1700
    level  7   ≈ 1900
    level  9   ≈ 2100
    level 12   ≈ 2300
    level 16   ≈ 2500
    level 20   ≈ 2700+

Environment overrides:

  ``EDAX_PATH``      — edax binary (default: ``edax`` on PATH, then
                       ``~/.local/bin/edax``).
  ``EDAX_DATA_DIR``  — directory containing ``data/eval.dat`` (default:
                       ``~/.local/share/edax``).
  ``EDAX_TIME``      — seconds per move (default 1.0).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def _find_edax() -> str:
    env = os.environ.get("EDAX_PATH")
    if env:
        return env
    on_path = shutil.which("edax")
    if on_path:
        return on_path
    user_local = Path.home() / ".local" / "bin" / "edax"
    if user_local.is_file():
        return str(user_local)
    raise FileNotFoundError(
        "Edax binary not found. Set EDAX_PATH or place the binary on "
        "PATH (e.g. ~/.local/bin/edax)."
    )


def _data_dir() -> Path:
    env = os.environ.get("EDAX_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / ".local" / "share" / "edax"


# One engine process per python process.
_ENGINE: "EdaxEngine | None" = None
_ENGINE_LOCK = threading.Lock()


class EdaxEngine:
    """Thin wrapper around an interactive Edax subprocess.

    Edax has no clean machine-readable protocol; in default mode it
    prints "Edax plays X" (or "Edax passes") on a line of its own after
    ``go``, then a fresh board picture. We just send commands and read
    stdout until that announcement line shows up.
    """

    _PLAY_RE = re.compile(r"Edax plays\s+([A-Ha-h][1-8])", re.IGNORECASE)
    _PASS_RE = re.compile(r"Edax\s+passes", re.IGNORECASE)

    def __init__(self) -> None:
        binary = _find_edax()
        cwd = _data_dir()
        if not (cwd / "data" / "eval.dat").is_file():
            raise FileNotFoundError(
                f"Edax data/eval.dat not found under {cwd}. Place the "
                "eval weights there, or set EDAX_DATA_DIR."
            )
        # ``bufsize=1`` + line-buffered text mode so we can read replies
        # promptly. Edax inherits cwd so it can find data/eval.dat.
        self.proc = subprocess.Popen(
            [binary],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        # Initialize: quieten output, disable book so wrappers are
        # deterministic, default to mode=0 (no auto play).
        self._send("verbose 0")
        self._send("book-usage off")
        self._send("mode 0")
        self._send("auto-start off")

    def _send(self, cmd: str) -> None:
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, pattern: re.Pattern[str], passes_ok: bool, timeout: float):
        """Read stdout lines until ``pattern`` matches (return Match) or
        ``self._PASS_RE`` matches (return ``"pass"`` if passes_ok else
        None). Times out after ``timeout`` seconds total."""
        deadline = time.perf_counter() + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            line = self.proc.stdout.readline()
            if not line:
                return None
            m = pattern.search(line)
            if m:
                return m
            if passes_ok and self._PASS_RE.search(line):
                return "pass"

    def setboard_and_move(
        self,
        board_rows: list[list[str]],
        actor: int,
        level: int,
        move_time: float,
    ) -> str | None:
        """Set board directly via ``setboard``, then ``go``.

        Avoids the fragile ``init`` + ``play <moves>`` replay path
        (which breaks if BoardArena history contains a PASS — Edax has
        no ``play pass`` command and silently desyncs).

        Edax setboard format (see board.c::board_set): 64 chars for
        A1..H8 row-major, then a side-to-move char. We use the
        absolute-color convention 'X' = BLACK, 'O' = WHITE, '-' empty;
        trailing 'X' if BLACK to move else 'O'.
        """
        chars = []
        for row in board_rows:  # row[0] = rank 1
            for cell in row:
                if cell == "B":
                    chars.append("X")
                elif cell == "W":
                    chars.append("O")
                else:
                    chars.append("-")
        side = "X" if actor == 0 else "O"
        board_str = "".join(chars) + " " + side
        with self._lock:
            self._send(f"setboard {board_str}")
            self._send(f"level {level}")
            self._send(f"move-time 0:00:{move_time:06.3f}")
            self._send("go")
            result = self._read_until(
                self._PLAY_RE,
                passes_ok=True,
                timeout=move_time + 5.0,
            )
            if result is None:
                return None
            if result == "pass":
                return "PASS"
            return result.group(1).lower()


def _engine() -> EdaxEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None or _ENGINE.proc.poll() is not None:
            _ENGINE = EdaxEngine()
        return _ENGINE


def make_bot(elo: int, level: int) -> type:
    """Build a ``Bot`` class pinned to an Edax level + Elo label."""
    move_time = float(os.environ.get("EDAX_TIME", "1.0"))

    class _EdaxBot:
        name = f"edax_{elo}_lvl{level}"

        def __init__(self) -> None:
            self._level = level
            self._move_time = move_time
            self._engine = _engine()

        def choose_action(self, state: dict[str, Any]) -> str:
            board = state.get("board")
            actor = state.get("actor", 0)
            legal = state.get("legal_actions", []) or []
            # PASS is the only legal action when no flips are available.
            if legal == ["PASS"]:
                return "PASS"
            if not board:
                return legal[0] if legal else ""
            move = self._engine.setboard_and_move(
                board, actor, self._level, self._move_time,
            )
            if move is None:
                # Edax stuck/dead — fall back to any legal action so we
                # don't forfeit.
                return legal[0] if legal else ""
            if move == "PASS":
                return "PASS" if "PASS" in legal else (legal[0] if legal else "")
            if move in legal:
                return move
            # case mismatch
            move_l = move.lower()
            move_u = move.upper()
            for m in (move_l, move_u):
                if m in legal:
                    return m
            return legal[0] if legal else ""

    _EdaxBot.__name__ = f"EdaxBot{elo}"
    return _EdaxBot
