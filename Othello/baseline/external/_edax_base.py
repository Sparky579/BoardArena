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
import queue
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

    Edax runs in CC (computer-vs-computer) mode by default and
    auto-plays both sides after each ``go``.  We use a background
    reader thread that drains stdout into a Queue so that every
    ``_read_until`` call has a real per-read timeout — ``readline()``
    alone blocks indefinitely.

    Protocol per move:
      1. Send ``setboard <pos> <side>``, ``level <n>``,
         ``move-time 0:00:<t>``, ``go``.
      2. Read the *first* ``Edax plays X`` line — that is the move we
         asked for.
      3. Drain the *second* ``Edax plays X`` that the CC auto-play
         produces, so the queue is clean for the next call.
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
        # Background reader thread — continuously drains stdout into queue.
        self._q: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        # Give Edax a moment to finish its startup banner, then discard it.
        time.sleep(0.5)
        self._drain_queue()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Daemon thread: push every stdout line into ``self._q``."""
        try:
            for line in self.proc.stdout:
                self._q.put(line)
        except Exception:
            pass

    def _drain_queue(self) -> None:
        """Discard everything currently in the queue (non-blocking)."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _send(self, cmd: str) -> None:
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(
        self,
        pattern: re.Pattern[str],
        passes_ok: bool,
        timeout: float,
    ):
        """Read queued lines until ``pattern`` matches or timeout expires.

        Returns:
          re.Match  — the matching line
          "pass"    — if passes_ok and Edax passes
          None      — timeout or process died
        """
        deadline = time.perf_counter() + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            try:
                # Block at most 0.5 s per iteration so we can recheck the
                # deadline without getting stuck in queue.get() forever.
                line = self._q.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            m = pattern.search(line)
            if m:
                return m
            if passes_ok and self._PASS_RE.search(line):
                return "pass"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setboard_and_move(
        self,
        board_rows: list[list[str]],
        actor: int,
        level: int,
        move_time: float,
    ) -> str | None:
        """Set board via ``setboard``, ask Edax for a move, return it.

        Edax setboard format (board.c::board_set): 64 chars A1..H8
        row-major (rank 1 first), then a side-to-move char.
        Convention: 'X' = BLACK, 'O' = WHITE, '-' = empty.

        ``board_rows`` is ``state["board"]`` from the Othello env, whose
        index 0 is rank 8 (display top).  We reverse it so rank 1 comes
        first, matching Edax's expected ordering.
        """
        chars = []
        for row in reversed(board_rows):  # rank 1 first for Edax A1..H8
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
            # Discard any residual output left from a previous auto-play.
            self._drain_queue()

            self._send(f"setboard {board_str}")
            self._send(f"level {level}")
            self._send(f"move-time 0:00:{move_time:06.3f}")
            self._send("go")

            # First "Edax plays X" — the move we requested.
            result = self._read_until(
                self._PLAY_RE,
                passes_ok=True,
                timeout=move_time + 10.0,
            )
            # Drain any CC auto-play response (short timeout — Edax may not
            # produce one depending on mode; _drain_queue() at call start
            # handles residual from prior calls).
            self._read_until(
                self._PLAY_RE,
                passes_ok=True,
                timeout=1.0,
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
            if legal == ["PASS"]:
                return "PASS"
            if not board:
                return legal[0] if legal else ""
            move = self._engine.setboard_and_move(
                board, actor, self._level, self._move_time,
            )
            if move is None:
                return legal[0] if legal else ""
            if move == "PASS":
                return "PASS" if "PASS" in legal else (legal[0] if legal else "")
            if move in legal:
                return move
            move_l = move.lower()
            move_u = move.upper()
            for m in (move_l, move_u):
                if m in legal:
                    return m
            return legal[0] if legal else ""

    _EdaxBot.__name__ = f"EdaxBot{elo}"
    return _EdaxBot
