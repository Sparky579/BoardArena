#!/usr/bin/env python3
"""Local browser UI for playing simplified Skull against trained policies."""

from __future__ import annotations

import argparse
import errno
import json
import random
import sys
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASELINE_DIR = ROOT / "baseline"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bot_id_for_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(BASELINE_DIR.resolve())
        return f"/{relative.with_suffix('').as_posix()}"
    except ValueError:
        return f"/{path.stem}"


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

from skull_cfr import (
    FLOWER,
    SKULL,
    State,
    load_policy,
    policy_action,
    weighted_choice,
)


Policy = Dict[str, Dict[str, float]]

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


class GameSession:
    def __init__(
        self,
        human: int,
        policy_mode: str,
        policy: Policy,
        recall: str,
        seed: Optional[int] = None,
    ) -> None:
        self.human = human
        self.cpu = 1 - human
        self.policy_mode = policy_mode
        self.policy = policy
        self.recall = recall
        self.rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
        self.state = State()
        self.log: List[Dict[str, object]] = []
        self.advance_cpu()

    def apply_human_action(self, action: str) -> None:
        if self.state.winner() is not None:
            return
        if self.state.actor != self.human or self.state.phase == "remove":
            raise ValueError("现在不是你的回合")
        if action not in self.state.legal_actions():
            raise ValueError("非法动作")
        self.log_action("你", action)
        self.state = self.state.apply_action(action)
        self.advance_cpu()

    def advance_cpu(self) -> None:
        guard = 0
        while self.state.winner() is None:
            guard += 1
            if guard > 128:
                raise RuntimeError("自动行动超过步数限制")
            if self.state.phase == "remove":
                loser = self.state.pending_loser
                card = weighted_choice(self.rng, self.state.chance_outcomes())
                self.state = self.state.apply_chance(card)
                if loser == self.human:
                    self.log.append(
                        {"text": f"你叫号失败，随机失去一张{card_name(card)}。", "important": True}
                    )
                else:
                    self.log.append(
                        {"text": "CPU 叫号失败，随机失去一张隐藏牌。", "important": True}
                    )
                continue

            if self.state.actor == self.human:
                return

            action = policy_action(
                self.state, self.policy, self.recall, self.rng, greedy=False
            )
            self.log_action("CPU", action)
            self.state = self.state.apply_action(action)

        winner = self.state.winner()
        if winner == self.human:
            self.log.append({"text": "你赢了。", "important": True})
        else:
            self.log.append({"text": "CPU 赢了。", "important": True})

    def log_action(self, who: str, action: str) -> None:
        if who == "CPU" and action in ("PLAY_F", "PLAY_S"):
            self.log.append({"text": "CPU: 放下一张牌", "important": False})
            return
        self.log.append({"text": f"{who}: {action_label(action)}", "important": False})

    def view(self, session_id: str) -> Dict[str, object]:
        state = self.state
        human = self.human
        cpu = self.cpu
        legal = (
            state.legal_actions()
            if state.winner() is None and state.actor == human and state.phase != "remove"
            else []
        )
        winner = state.winner()
        return {
            "session": session_id,
            "winner": winner,
            "policy_mode": self.policy_mode,
            "recall": self.recall,
            "status": self.status_text(),
            "hint": self.hint_text(legal),
            "legal_actions": legal,
            "human": {
                "id": human,
                "score": state.scores[human],
                "total_cards": state.total_cards(human),
                "hand": {
                    "flowers": state.hands[human][0],
                    "skulls": state.hands[human][1],
                },
                "pile": list(state.piles[human]),
            },
            "cpu": {
                "id": cpu,
                "score": state.scores[cpu],
                "total_cards": state.total_cards(cpu),
                "pile_count": len(state.piles[cpu]),
                "pile": list(state.piles[cpu]),
            },
            "log": self.log[-80:],
        }

    def status_text(self) -> str:
        winner = self.state.winner()
        if winner is not None:
            return "你赢了" if winner == self.human else "CPU 赢了"
        if self.state.phase == "play":
            if all(len(pile) > 0 for pile in self.state.piles):
                return "放牌阶段：双方都已放牌，现在可以继续放牌或叫号"
            return "放牌阶段：双方都至少放一张牌后才能叫号"
        if self.state.phase == "bid":
            high = "你" if self.state.high_bidder == self.human else "CPU"
            return f"叫号阶段：当前 {self.state.current_bid}，最高叫号 {high}"
        return "结算中"

    def hint_text(self, legal: Sequence[str]) -> str:
        if self.state.winner() is not None:
            return "先拿到 2 分，或让对手没有牌，即获胜。"
        if not legal:
            return "CPU 正在行动。"
        if self.state.phase == "play":
            if any(action.startswith("BID_") for action in legal):
                return "可以继续暗放一张牌，也可以开始叫号。"
            return "还不能叫号：双方都至少放过一张牌后才允许叫号。"
        return "可以加叫，也可以放弃让当前最高叫号结算。"


def action_label(action: str) -> str:
    if action == "PLAY_F":
        return "放花"
    if action == "PLAY_S":
        return "放骷髅"
    if action == "PASS":
        return "放弃"
    if action.startswith("BID_"):
        return f"叫 {action[4:]}"
    return action


def card_name(card: str) -> str:
    if card == FLOWER:
        return "花"
    if card == SKULL:
        return "骷髅"
    return card


class SkullServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        policies: Dict[str, tuple[Policy, str]],
        bot_paths: dict[str, Path],
        seed: Optional[int],
    ) -> None:
        super().__init__(server_address, Handler)
        self.policies = policies
        self.bot_paths = bot_paths
        self.seed = seed
        self.sessions: Dict[str, GameSession] = {}
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: SkullServer

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
        if parsed.path == "/api/policies":
            self.send_json({"policies": sorted(self.server.policies)})
            return
        if parsed.path == "/api/bots":
            self.send_json({"bots": list(self.server.bot_paths), "default": next(iter(self.server.bot_paths)) if self.server.bot_paths else None})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            data = self.read_json()
            if self.path == "/api/new":
                human = int(data.get("human", 0))
                if human not in (0, 1):
                    raise ValueError("human must be 0 or 1")
                policy_mode = str(data.get("policy_mode", "compact"))
                if policy_mode not in self.server.policies:
                    raise ValueError(f"unknown policy mode: {policy_mode}")
                policy, recall = self.server.policies[policy_mode]
                session_id = uuid.uuid4().hex
                seed = None if self.server.seed is None else self.server.seed + len(self.server.sessions)
                session = GameSession(
                    human=human,
                    policy_mode=policy_mode,
                    policy=policy,
                    recall=recall,
                    seed=seed,
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

            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_error_json(str(exc), HTTPStatus.BAD_REQUEST)

    def read_json(self) -> Dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def send_json(self, payload: Dict[str, object]) -> None:
        self.send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def send_error_json(self, message: str, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}, ensure_ascii=False).encode("utf-8"))

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
        stat = path.stat()
        etag = f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        self.send_bytes(
            path.read_bytes(),
            STATIC_TYPES.get(path.suffix, "application/octet-stream"),
            cache_control="no-cache",
            etag=etag,
        )

    def send_bytes(
        self,
        body: bytes,
        content_type: str,
        *,
        cache_control: str | None = None,
        etag: str | None = None,
    ) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache_control is not None:
            self.send_header("Cache-Control", cache_control)
        if etag is not None:
            self.send_header("ETag", etag)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "skull_policy.json")
    parser.add_argument("--compact-policy", type=Path, default=ROOT / "skull_policy.json")
    parser.add_argument("--perfect-policy", type=Path, default=ROOT / "skull_policy_perfect.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-open", "--no-browser", dest="no_open", action="store_true")
    return parser


def load_named_policies(args: argparse.Namespace) -> Dict[str, tuple[Policy, str]]:
    compact_path = args.compact_policy or args.policy
    policies: Dict[str, tuple[Policy, str]] = {}
    policies["compact"] = load_policy(compact_path)
    if args.perfect_policy.exists():
        policies["perfect"] = load_policy(args.perfect_policy)
    return policies


def main() -> int:
    args = build_parser().parse_args()
    policies = load_named_policies(args)
    bot_paths = discover_bot_paths()
    server, port = bind_server(args, policies, bot_paths)
    url = f"http://{args.host}:{port}/"
    loaded = ", ".join(f"{name}:{recall}" for name, (_, recall) in policies.items())
    print(f"BoardArena skull UI: {url}")
    print(f"Loaded policies: {loaded}")
    if bot_paths:
        print(f"Discovered bots: {', '.join(bot_paths)}")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    return 0


def bind_server(
    args: argparse.Namespace,
    policies: Dict[str, tuple[Policy, str]],
    bot_paths: dict[str, Path],
) -> tuple[SkullServer, int]:
    for port in range(args.port, args.port + 20):
        try:
            return SkullServer((args.host, port), policies, bot_paths, args.seed), port
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise OSError(errno.EADDRINUSE, f"ports {args.port}-{args.port + 19} are in use")


if __name__ == "__main__":
    raise SystemExit(main())
