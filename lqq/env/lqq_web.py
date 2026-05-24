#!/usr/bin/env python3
"""Local browser interface for BoardArena Luqiangqi."""

from __future__ import annotations

import argparse
import errno
import json
import random
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from lqq_env import DEFAULT_TURN_LIMIT, BotTimeoutError, LqqEnv, SystemBot, choose_action_with_timeout, load_bot


HERE = Path(__file__).resolve().parent
DEFAULT_BOT_ID = "/baseline/bot_greedy"
BASELINE_DIR = HERE.parent / "baseline"
DEFAULT_DECISION_TIMEOUT = 1.0
ALLOWED_DECISION_TIMEOUTS = {1.0, 3.0, 8.0}
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


class GameSession:
    def __init__(
        self,
        *,
        mode: str,
        human_seat: int,
        bot_id: str,
        bot_path: Path,
        decision_timeout: float,
        seed: int | None,
        turn_limit: int,
    ) -> None:
        if mode not in {"human-human", "human-bot"}:
            raise ValueError("mode must be human-human or human-bot")
        if human_seat not in (0, 1):
            raise ValueError("human_seat must be 0 or 1")

        self.mode = mode
        self.bot_id = bot_id
        self.decision_timeout = decision_timeout
        self.human_seats = {0, 1} if mode == "human-human" else {human_seat}
        self.rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
        self.env = LqqEnv(seed=seed, turn_limit=turn_limit)
        self.log: list[dict[str, Any]] = []
        self.forfeit: dict[str, Any] | None = None
        self.bots: dict[int, Any] = {}
        self.lock = threading.Lock()

        if mode == "human-bot":
            bot_seat = 1 - human_seat
            self.bots[bot_seat] = load_bot(bot_path) if bot_path.exists() else SystemBot(self.rng)

    def set_bot(self, bot_id: str, bot_path: Path) -> None:
        with self.lock:
            if self.mode != "human-bot":
                raise ValueError("bot can only be changed in human-bot mode")
            bot_seat = next(seat for seat in (0, 1) if seat not in self.human_seats)
            self.bot_id = bot_id
            self.bots[bot_seat] = load_bot(bot_path) if bot_path.exists() else SystemBot(self.rng)
            self.log.append(
                {
                    "turn": self.env.game.turn,
                    "seat": bot_seat,
                    "action": "",
                    "text": f"Player {bot_seat + 1} bot changed to {bot_id}",
                }
            )

    def apply_human_action(self, action: str) -> None:
        with self.lock:
            if self.forfeit is not None:
                return
            state = self.env.state()
            actor = state["actor"]
            if state["phase"] == "game_over" or actor not in self.human_seats:
                raise ValueError("not a human turn")
            if action not in state["legal_actions"]:
                raise ValueError("illegal action")
            self._push_action(action, source="human")

    def advance_bots(self, decision_timeout: float | None = None) -> None:
        with self.lock:
            if decision_timeout is not None:
                self.decision_timeout = decision_timeout
            guard = 0
            while self.forfeit is None:
                state = self.env.state(decision_timeout=self.decision_timeout)
                if state["phase"] == "game_over" or state["actor"] in self.human_seats:
                    return
                guard += 1
                if guard > 64:
                    raise RuntimeError("bot advance exceeded guard limit")

                actor = state["actor"]
                bot = self.bots[actor]
                try:
                    action = choose_action_with_timeout(bot, state, self.decision_timeout)
                except BotTimeoutError as exc:
                    self.forfeit = {
                        "winner": 1 - actor,
                        "status": "timeout",
                        "error": str(exc),
                    }
                    return
                except Exception as exc:  # noqa: BLE001 - bot failures are game results.
                    self.forfeit = {
                        "winner": 1 - actor,
                        "status": "bot_exception",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    return
                if action not in state["legal_actions"]:
                    self.forfeit = {
                        "winner": 1 - actor,
                        "status": "invalid_action",
                        "error": f"invalid action from bot seat {actor}: {action!r}",
                    }
                    return
                self._push_action(action, source="bot")

    def view(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            state = self.env.state()
            human_turn = (
                self.forfeit is None
                and state["phase"] != "game_over"
                and state["actor"] in self.human_seats
            )
            bot_turn = (
                self.forfeit is None
                and self.mode == "human-bot"
                and state["phase"] != "game_over"
                and state["actor"] not in self.human_seats
            )
            if not human_turn:
                state["legal_actions"] = []

            if self.forfeit is not None:
                state["phase"] = "game_over"
                state["winner"] = self.forfeit["winner"]
                state["status"] = self.forfeit["status"]
                state["error"] = self.forfeit["error"]

            state.update(
                {
                    "session": session_id,
                    "mode": self.mode,
                    "bot_id": self.bot_id,
                    "decision_timeout": self.decision_timeout,
                    "human_seats": sorted(self.human_seats),
                    "human_turn": human_turn,
                    "bot_turn": bot_turn,
                    "status_text": self.status_text(state, bot_turn),
                    # Was self.log[-100:] which silently dropped older
                    # entries in long games, making the UI move-log look
                    # like a sliding window. Lqq tops out at 400 turns,
                    # so sending the full log is cheap.
                    "log": self.log,
                }
            )
            return state

    def status_text(self, state: dict[str, Any], bot_turn: bool) -> str:
        if state["phase"] == "game_over":
            winner = state.get("winner")
            status = state.get("status")
            error = state.get("error")
            if winner is None:
                return "Turn limit reached"
            res = f"Player {winner + 1} wins"
            if status in {"timeout", "bot_exception", "invalid_action"}:
                res += f" ({status}"
                if error:
                    res += f": {error}"
                res += ")"
            return res
        label = f"Player {state['actor'] + 1}"
        if bot_turn:
            return f"{label} bot is thinking"
        return f"{label} to move"

    def _push_action(self, action: str, *, source: str) -> None:
        actor = self.env.actor
        _, _, terminated, truncated, info = self.env.step(action)
        text = format_action(actor, action, source)
        self.log.append({"turn": self.env.game.turn, "seat": actor, "action": action, "text": text})
        if terminated or truncated:
            winner = info.get("winner")
            end = "Draw" if winner is None else f"Player {winner + 1} wins"
            self.log.append({"turn": self.env.game.turn, "seat": -1, "action": "", "text": end})


class LqqServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        bot_paths: dict[str, Path],
        default_bot_id: str,
        seed: int | None,
        turn_limit: int,
    ) -> None:
        super().__init__(server_address, Handler)
        self.bot_paths = bot_paths
        self.default_bot_id = default_bot_id
        self.seed = seed
        self.turn_limit = turn_limit
        self.sessions: dict[str, GameSession] = {}
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: LqqServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/play.html"):
            self.send_static("play.html")
            return
        if parsed.path in ("/play.js", "/play.css", "/index.html", "/game.js", "/styles.css"):
            self.send_static(parsed.path.lstrip("/"))
            return
        if parsed.path == "/api/state":
            session_id = parse_qs(parsed.query).get("session", [""])[0]
            session = self.server.sessions.get(session_id)
            if not session:
                self.send_error_json("session not found", HTTPStatus.NOT_FOUND)
                return
            self.send_json(session.view(session_id))
            return
        if parsed.path == "/api/bots":
            self.send_json({"bots": list(self.server.bot_paths), "default": self.server.default_bot_id})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            data = self.read_json()
            if self.path == "/api/new":
                mode = str(data.get("mode", "human-human"))
                human_seat = int(data.get("human_seat", 0))
                bot_id = str(data.get("bot", self.server.default_bot_id))
                if bot_id not in self.server.bot_paths:
                    raise ValueError(f"unknown bot: {bot_id}")
                session_id = uuid.uuid4().hex
                seed = None if self.server.seed is None else self.server.seed + len(self.server.sessions)
                session = GameSession(
                    mode=mode,
                    human_seat=human_seat,
                    bot_id=bot_id,
                    bot_path=self.server.bot_paths[bot_id],
                    decision_timeout=parse_decision_timeout(data.get("decision_timeout", DEFAULT_DECISION_TIMEOUT)),
                    seed=seed,
                    turn_limit=self.server.turn_limit,
                )
                with self.server.lock:
                    self.server.sessions[session_id] = session
                self.send_json(session.view(session_id))
                return
            if self.path == "/api/action":
                session = self.server.sessions.get(str(data.get("session", "")))
                if not session:
                    self.send_error_json("session not found", HTTPStatus.NOT_FOUND)
                    return
                session.apply_human_action(str(data.get("action", "")))
                self.send_json(session.view(str(data.get("session", ""))))
                return
            if self.path == "/api/bot":
                session_id = str(data.get("session", ""))
                session = self.server.sessions.get(session_id)
                if not session:
                    self.send_error_json("session not found", HTTPStatus.NOT_FOUND)
                    return
                bot_id = str(data.get("bot", session.bot_id))
                if bot_id not in self.server.bot_paths:
                    raise ValueError(f"unknown bot: {bot_id}")
                session.set_bot(bot_id, self.server.bot_paths[bot_id])
                self.send_json(session.view(session_id))
                return
            if self.path == "/api/advance":
                session = self.server.sessions.get(str(data.get("session", "")))
                if not session:
                    self.send_error_json("session not found", HTTPStatus.NOT_FOUND)
                    return
                timeout = parse_optional_decision_timeout(data.get("decision_timeout"))
                session.advance_bots(timeout)
                self.send_json(session.view(str(data.get("session", ""))))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_error_json(str(exc), HTTPStatus.BAD_REQUEST)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def send_static(self, filename: str) -> None:
        path = (HERE / filename).resolve()
        try:
            path.relative_to(HERE)
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_bytes(path.read_bytes(), STATIC_TYPES.get(path.suffix, "application/octet-stream"))

    def send_json(self, payload: dict[str, Any]) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def send_error_json(self, message: str, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}, ensure_ascii=False).encode("utf-8"))

    def send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def parse_decision_timeout(value: Any) -> float:
    timeout = float(value)
    if timeout not in ALLOWED_DECISION_TIMEOUTS:
        allowed = ", ".join(f"{item:g}" for item in sorted(ALLOWED_DECISION_TIMEOUTS))
        raise ValueError(f"decision_timeout must be one of: {allowed}")
    return timeout


def parse_optional_decision_timeout(value: Any) -> float | None:
    if value is None:
        return None
    return parse_decision_timeout(value)


def format_action(actor: int, action: str, source: str) -> str:
    who = f"Player {actor + 1}" if source == "human" else f"Player {actor + 1} bot"
    return f"{who}: {action}"


def discover_bot_paths() -> dict[str, Path]:
    bot_paths: dict[str, Path] = {}
    if not BASELINE_DIR.is_dir():
        return bot_paths

    for path in sorted(BASELINE_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        bot_paths[bot_id_for_path(path)] = path

    for pattern in ("bot.py", "bot_*.py"):
        for path in sorted(BASELINE_DIR.rglob(pattern)):
            if "__pycache__" in path.parts:
                continue
            bot_paths[bot_id_for_path(path)] = path

    return bot_paths


def bot_id_for_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(HERE.parent)
        return f"/{relative.with_suffix('').as_posix()}"
    except ValueError:
        return f"/custom/{path.stem}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the BoardArena Luqiangqi browser UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--bot", type=Path, default=None, help="optional bot file to add to the browser list")
    parser.add_argument("--default-bot", default=DEFAULT_BOT_ID)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--turn-limit", type=int, default=DEFAULT_TURN_LIMIT)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    bot_paths = discover_bot_paths()
    if args.bot is not None:
        bot_paths[bot_id_for_path(args.bot)] = args.bot
    if not bot_paths:
        raise FileNotFoundError(f"no bot files found under {BASELINE_DIR}")
    default_bot_id = args.default_bot if args.default_bot in bot_paths else next(iter(bot_paths))
    server, port = bind_server(args, bot_paths, default_bot_id)
    url = f"http://{args.host}:{port}/"
    print(f"BoardArena lqq UI: {url}")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    server.serve_forever()
    return 0


def bind_server(args: argparse.Namespace, bot_paths: dict[str, Path], default_bot_id: str) -> tuple[LqqServer, int]:
    for port in range(args.port, args.port + 20):
        try:
            return (
                LqqServer(
                    (args.host, port),
                    bot_paths=bot_paths,
                    default_bot_id=default_bot_id,
                    seed=args.seed,
                    turn_limit=args.turn_limit,
                ),
                port,
            )
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise OSError(errno.EADDRINUSE, f"ports {args.port}-{args.port + 19} are in use")


if __name__ == "__main__":
    raise SystemExit(main())
