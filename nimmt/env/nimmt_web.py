#!/usr/bin/env python3
"""Local browser interface for BoardArena simplified 6 nimmt!."""

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

from nimmt_env import MAX_PLAYERS, MIN_PLAYERS, BotTimeoutError, NimmtEnv, SystemBot, choose_action_with_timeout, load_bot


HERE = Path(__file__).resolve().parent
DEFAULT_BOTS = {
    "/baseline/bot_random": HERE.parent / "baseline" / "bot_random.py",
    "/baseline/bot_greedy": HERE.parent / "baseline" / "bot_greedy.py",
}
DEFAULT_BOT_ID = "/baseline/bot_greedy"
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
        players: int,
        human_seat: int,
        bot_id: str,
        bot_path: Path,
        decision_timeout: float,
        seed: int | None,
    ) -> None:
        if mode not in {"human-human", "human-bot"}:
            raise ValueError("mode must be human-human or human-bot")
        if not MIN_PLAYERS <= players <= MAX_PLAYERS:
            raise ValueError(f"players must be between {MIN_PLAYERS} and {MAX_PLAYERS}")
        if not 0 <= human_seat < players:
            raise ValueError("human_seat must be within player range")

        self.mode = mode
        self.players = players
        self.bot_id = bot_id
        self.decision_timeout = decision_timeout
        self.human_seats = set(range(players)) if mode == "human-human" else {human_seat}
        self.rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
        self.env = NimmtEnv(players=players, seed=seed)
        self.pending: dict[int, str] = {}
        self.log: list[dict[str, Any]] = []
        self.forfeit: dict[str, Any] | None = None
        self.bots: dict[int, Any] = {}

        if mode == "human-bot":
            for seat in range(players):
                if seat not in self.human_seats:
                    self.bots[seat] = load_bot(bot_path) if bot_path.exists() else SystemBot(name=f"system_{seat}")

    def apply_human_action(self, player: int, action: str, decision_timeout: float | None = None) -> None:
        if decision_timeout is not None:
            self.decision_timeout = decision_timeout
        if self.forfeit is not None or self.env.game.finished():
            return
        if player not in self.human_seats:
            raise ValueError("not a human seat")
        if player in self.pending:
            raise ValueError("that player has already selected a card this turn")
        legal = self.env.legal_actions(player)
        if action not in legal:
            raise ValueError("illegal action")
        self.pending[player] = action
        self._finish_turn_if_ready()

    def _finish_turn_if_ready(self) -> None:
        if self.forfeit is not None or self.env.game.finished():
            return
        if any(player not in self.pending for player in self.human_seats):
            return

        actions: dict[int, str] = dict(self.pending)
        for player in range(self.players):
            if player in actions:
                continue
            bot = self.bots[player]
            bot_state = self.env.state(player)
            try:
                action = choose_action_with_timeout(bot, bot_state, self.decision_timeout)
            except BotTimeoutError as exc:
                self.forfeit = {
                    "winner": None,
                    "status": "timeout",
                    "error": f"player {player}: {exc}",
                }
                return
            except Exception as exc:  # noqa: BLE001
                self.forfeit = {
                    "winner": None,
                    "status": "bot_exception",
                    "error": f"player {player}: {type(exc).__name__}: {exc}",
                }
                return
            if action not in bot_state["legal_actions"]:
                self.forfeit = {
                    "winner": None,
                    "status": "invalid_action",
                    "error": f"invalid action from bot seat {player}: {action!r}",
                }
                return
            actions[player] = action

        _, _, terminated, _, info = self.env.step(actions)
        for player in range(self.players):
            self.log.append(
                {
                    "turn": self.env.game.turn,
                    "seat": player,
                    "action": actions[player],
                    "text": f"P{player + 1}: {actions[player]}",
                }
            )
        for event in info.get("events", []):
            self.log.append({"turn": self.env.game.turn, "seat": -1, "action": "", "text": event})
        if terminated:
            winners = ", ".join(f"P{player + 1}" for player in info["winners"])
            self.log.append({"turn": self.env.game.turn, "seat": -1, "action": "", "text": f"Winners: {winners}"})
        self.pending = {}

    def view(self, session_id: str) -> dict[str, Any]:
        state = self.env.state()
        next_human = self._next_human()
        legal = self.env.legal_actions(next_human) if next_human is not None else []
        if self.forfeit is not None:
            state["phase"] = "game_over"
            state["status"] = self.forfeit["status"]
            state["error"] = self.forfeit["error"]

        state.update(
            {
                "session": session_id,
                "mode": self.mode,
                "bot_id": self.bot_id,
                "decision_timeout": self.decision_timeout,
                "human_seats": sorted(self.human_seats),
                "pending": dict(self.pending),
                "next_human": next_human,
                "legal_actions": legal,
                "status_text": self.status_text(state, next_human),
                "log": self.log[-160:],
            }
        )
        return state

    def status_text(self, state: dict[str, Any], next_human: int | None) -> str:
        if state["phase"] == "game_over":
            winners = state.get("winners") or []
            return "Winners: " + ", ".join(f"P{player + 1}" for player in winners)
        if next_human is None:
            return "Resolving turn"
        return f"Player {next_human + 1} choose a card"

    def _next_human(self) -> int | None:
        if self.env.game.finished():
            return None
        for player in sorted(self.human_seats):
            if player not in self.pending:
                return player
        return None


class NimmtServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        bot_paths: dict[str, Path],
        default_bot_id: str,
        seed: int | None,
    ) -> None:
        super().__init__(server_address, Handler)
        self.bot_paths = bot_paths
        self.default_bot_id = default_bot_id
        self.seed = seed
        self.sessions: dict[str, GameSession] = {}
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: NimmtServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_static("index.html")
            return
        if parsed.path in ("/game.js", "/styles.css"):
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
                players = int(data.get("players", 4))
                human_seat = int(data.get("human_seat", 0))
                bot_id = str(data.get("bot", self.server.default_bot_id))
                if bot_id not in self.server.bot_paths:
                    raise ValueError(f"unknown bot: {bot_id}")
                session_id = uuid.uuid4().hex
                seed = None if self.server.seed is None else self.server.seed + len(self.server.sessions)
                session = GameSession(
                    mode=mode,
                    players=players,
                    human_seat=human_seat,
                    bot_id=bot_id,
                    bot_path=self.server.bot_paths[bot_id],
                    decision_timeout=parse_decision_timeout(data.get("decision_timeout", DEFAULT_DECISION_TIMEOUT)),
                    seed=seed,
                )
                with self.server.lock:
                    self.server.sessions[session_id] = session
                self.send_json(session.view(session_id))
                return
            if self.path == "/api/action":
                session_id = str(data.get("session", ""))
                session = self.server.sessions.get(session_id)
                if not session:
                    self.send_error_json("session not found", HTTPStatus.NOT_FOUND)
                    return
                timeout = parse_optional_decision_timeout(data.get("decision_timeout"))
                session.apply_human_action(int(data.get("player", -1)), str(data.get("action", "")), timeout)
                self.send_json(session.view(session_id))
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the BoardArena nimmt browser UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--bot", type=Path, default=None, help="optional path overriding /baseline/bot_greedy")
    parser.add_argument("--default-bot", choices=tuple(DEFAULT_BOTS), default=DEFAULT_BOT_ID)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    bot_paths = dict(DEFAULT_BOTS)
    if args.bot is not None:
        bot_paths["/baseline/bot_greedy"] = args.bot
    server, port = bind_server(args, bot_paths)
    url = f"http://{args.host}:{port}/"
    print(f"BoardArena nimmt UI: {url}")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    server.serve_forever()
    return 0


def bind_server(args: argparse.Namespace, bot_paths: dict[str, Path]) -> tuple[NimmtServer, int]:
    for port in range(args.port, args.port + 20):
        try:
            return (
                NimmtServer(
                    (args.host, port),
                    bot_paths=bot_paths,
                    default_bot_id=args.default_bot,
                    seed=args.seed,
                ),
                port,
            )
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise OSError(errno.EADDRINUSE, f"ports {args.port}-{args.port + 19} are in use")


if __name__ == "__main__":
    raise SystemExit(main())
