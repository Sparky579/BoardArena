#!/usr/bin/env python3
"""Local browser interface for BoardArena chess."""

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

import chess

from chess_env import DEFAULT_MAX_PLIES, ChessEnv, SystemBot, load_bot


HERE = Path(__file__).resolve().parent
DEFAULT_BOTS = {
    "/gpt5p5/bot_easy": HERE.parent / "baseline" / "gpt5p5" / "bot_easy.py",
    "/gpt5p5/bot_hard": HERE.parent / "baseline" / "gpt5p5" / "bot_hard.py",
}
DEFAULT_BOT_ID = "/gpt5p5/bot_hard"
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
        seed: int | None,
        max_plies: int,
    ) -> None:
        if mode not in {"human-human", "human-bot"}:
            raise ValueError("mode must be human-human or human-bot")
        if human_seat not in (0, 1):
            raise ValueError("human_seat must be 0 or 1")

        self.mode = mode
        self.bot_id = bot_id
        self.human_seats = {0, 1} if mode == "human-human" else {human_seat}
        self.rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
        self.env = ChessEnv(seed=seed, max_plies=max_plies)
        self.log: list[dict[str, Any]] = []
        self.forfeit: dict[str, Any] | None = None
        self.bots: dict[int, Any] = {}

        if mode == "human-bot":
            bot_seat = 1 - human_seat
            self.bots[bot_seat] = load_bot(bot_path) if bot_path.exists() else SystemBot(self.rng)

    def apply_human_action(self, action: str) -> None:
        if self.forfeit is not None:
            return
        state = self.env.state()
        actor = state["actor"]
        if actor not in self.human_seats or state["phase"] == "game_over":
            raise ValueError("现在不是人类玩家回合")
        if action not in state["legal_actions"]:
            raise ValueError("非法动作")
        self._push_action(action, source="human")

    def advance_bots(self) -> None:
        guard = 0
        while self.forfeit is None:
            state = self.env.state()
            if state["phase"] == "game_over" or state["actor"] in self.human_seats:
                return

            guard += 1
            if guard > 64:
                raise RuntimeError("自动行动超过步数限制")

            actor = state["actor"]
            bot = self.bots[actor]
            try:
                action = bot.choose_action(state)
            except Exception as exc:  # noqa: BLE001 - user bot errors become session result.
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
        state = self.env.state()
        bot_turn = (
            self.forfeit is None
            and self.mode == "human-bot"
            and state["phase"] != "game_over"
            and state["actor"] not in self.human_seats
        )
        human_turn = (
            self.forfeit is None
            and state["phase"] != "game_over"
            and state["actor"] in self.human_seats
        )
        if not human_turn:
            state["legal_actions"] = []

        if self.forfeit is not None:
            state["phase"] = "game_over"
            state["winner"] = self.forfeit["winner"]
            state["status"] = self.forfeit["status"]
            state["result"] = "1-0" if self.forfeit["winner"] == 0 else "0-1"

        state.update(
            {
                "session": session_id,
                "mode": self.mode,
                "bot_id": self.bot_id,
                "bot_name": self.bot_id,
                "human_seats": sorted(self.human_seats),
                "human_turn": human_turn,
                "bot_turn": bot_turn,
                "status_text": self.status_text(state),
                "log": self.log[-120:],
            }
        )
        return state

    def status_text(self, state: dict[str, Any]) -> str:
        if state["phase"] == "game_over":
            winner = state["winner"]
            status = state["status"]
            if winner is not None:
                side = "白方" if winner == 0 else "黑方"
                if status == "checkmate":
                    return f"{side}将死获胜"
                return f"{side}获胜"
            labels = {
                "stalemate": "无合法走法，和棋",
                "insufficient_material": "子力不足，和棋",
                "seventyfive_moves": "75 回合规则，和棋",
                "fivefold_repetition": "五次重复局面，和棋",
                "fifty_moves": "50 回合规则，和棋",
                "threefold_repetition": "三次重复局面，和棋",
                "turn_limit": "达到步数上限，和棋",
            }
            return labels.get(status, "对局结束")

        side = "白方" if state["actor"] == 0 else "黑方"
        if self.mode == "human-bot" and state["actor"] not in self.human_seats:
            return f"{side} Bot 思考中"
        suffix = "，被将军" if state["check"] else ""
        return f"{side}行动{suffix}"

    def _push_action(self, action: str, *, source: str) -> None:
        actor = self.env.actor
        move = chess.Move.from_uci(action)
        san = self.env.board.san(move)
        _, _, terminated, truncated, info = self.env.step(action)
        side = "白方" if actor == 0 else "黑方"
        who = side if source == "human" else f"{side} Bot"
        self.log.append({"ply": self.env.plies, "seat": actor, "action": action, "san": san, "text": f"{who}: {san}"})
        if terminated or truncated:
            status = info.get("status", "turn_limit" if truncated else "ok")
            winner = info.get("winner")
            if winner is None:
                self.log.append({"ply": self.env.plies, "seat": -1, "action": "", "san": "", "text": f"结束: {status}"})
            else:
                side = "白方" if winner == 0 else "黑方"
                self.log.append({"ply": self.env.plies, "seat": winner, "action": "", "san": "", "text": f"结束: {side}胜"})


class ChessServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        bot_paths: dict[str, Path],
        default_bot_id: str,
        seed: int | None,
        max_plies: int,
    ) -> None:
        super().__init__(server_address, Handler)
        self.bot_paths = bot_paths
        self.default_bot_id = default_bot_id
        self.seed = seed
        self.max_plies = max_plies
        self.sessions: dict[str, GameSession] = {}
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: ChessServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_static("index.html")
            return
        if parsed.path in ("/game.js", "/styles.css"):
            self.send_static(parsed.path.lstrip("/"))
            return
        if parsed.path == "/api/state":
            qs = parse_qs(parsed.query)
            session_id = qs.get("session", [""])[0]
            session = self.server.sessions.get(session_id)
            if not session:
                self.send_error_json("session not found", HTTPStatus.NOT_FOUND)
                return
            self.send_json(session.view(session_id))
            return
        if parsed.path == "/api/bots":
            self.send_json({"bots": sorted(self.server.bot_paths), "default": self.server.default_bot_id})
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
                    seed=seed,
                    max_plies=self.server.max_plies,
                )
                with self.server.lock:
                    self.server.sessions[session_id] = session
                self.send_json(session.view(session_id))
                return

            if self.path == "/api/action":
                session_id = str(data.get("session", ""))
                action = str(data.get("action", ""))
                session = self.server.sessions.get(session_id)
                if not session:
                    self.send_error_json("session not found", HTTPStatus.NOT_FOUND)
                    return
                session.apply_human_action(action)
                self.send_json(session.view(session_id))
                return

            if self.path == "/api/advance":
                session_id = str(data.get("session", ""))
                session = self.server.sessions.get(session_id)
                if not session:
                    self.send_error_json("session not found", HTTPStatus.NOT_FOUND)
                    return
                session.advance_bots()
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
        path = HERE / filename
        if not path.exists():
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
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - stdlib hook name.
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the BoardArena chess browser UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--bot", type=Path, default=None, help="optional path overriding /gpt5p5/bot_hard")
    parser.add_argument("--default-bot", choices=tuple(DEFAULT_BOTS), default=DEFAULT_BOT_ID)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    args.bot_paths = dict(DEFAULT_BOTS)
    if args.bot is not None:
        args.bot_paths["/gpt5p5/bot_hard"] = args.bot

    server, port = bind_server(args)
    url = f"http://{args.host}:{port}/"
    print(f"BoardArena chess UI: {url}")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    server.serve_forever()
    return 0


def bind_server(args: argparse.Namespace) -> tuple[ChessServer, int]:
    for port in range(args.port, args.port + 20):
        try:
            return (
                ChessServer(
                    (args.host, port),
                    bot_paths=args.bot_paths,
                    default_bot_id=args.default_bot,
                    seed=args.seed,
                    max_plies=args.max_plies,
                ),
                port,
            )
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise

    raise OSError(
        errno.EADDRINUSE,
        f"ports {args.port}-{args.port + 19} are already in use; pass --port to choose another one",
    )


if __name__ == "__main__":
    raise SystemExit(main())
