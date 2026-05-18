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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skull_cfr import (
    FLOWER,
    SKULL,
    State,
    load_policy,
    policy_action,
    weighted_choice,
)


Policy = Dict[str, Dict[str, float]]


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Skull CFR</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f3ea;
      --ink: #1e2524;
      --muted: #66706d;
      --line: #d9d0c0;
      --panel: #fffdf7;
      --red: #ad3434;
      --green: #2f7d54;
      --blue: #315f9f;
      --shadow: 0 10px 28px rgba(32, 28, 18, .12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #fff8df 0, transparent 30rem), var(--bg);
      color: var(--ink);
    }
    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1;
      letter-spacing: 0;
    }
    .top-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .policy-select {
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 40px;
      padding: 8px 34px 8px 12px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-weight: 750;
    }
    .toggle.active {
      background: #f0c84b;
      border-color: #d1a72f;
      color: #1e2524;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      min-height: 40px;
      padding: 9px 13px;
      border-radius: 8px;
      font-weight: 700;
      cursor: pointer;
      transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
    }
    button:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 6px 18px rgba(30, 37, 36, .12);
      border-color: #b9ad98;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: .45;
    }
    .primary { background: var(--ink); color: #fff; border-color: var(--ink); }
    .danger { color: #fff; background: var(--red); border-color: var(--red); }
    .good { color: #fff; background: var(--green); border-color: var(--green); }
    .blue { color: #fff; background: var(--blue); border-color: var(--blue); }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(300px, .8fr);
      gap: 16px;
      align-items: start;
    }
    .table {
      background: linear-gradient(135deg, #2d604b, #244b42);
      min-height: 560px;
      border: 1px solid #1d3e35;
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 18px;
      display: grid;
      grid-template-rows: 1fr auto 1fr;
      gap: 18px;
      color: #f9fbf7;
    }
    .player {
      display: grid;
      grid-template-columns: minmax(128px, 180px) 1fr;
      gap: 16px;
      align-items: center;
    }
    .player.cpu { align-items: start; }
    .name {
      font-size: 13px;
      color: rgba(255,255,255,.75);
      text-transform: uppercase;
      font-weight: 800;
      letter-spacing: .08em;
    }
    .score {
      font-size: 36px;
      font-weight: 900;
      line-height: 1;
      margin-top: 8px;
    }
    .meta {
      margin-top: 10px;
      display: grid;
      gap: 4px;
      color: rgba(255,255,255,.82);
      font-size: 14px;
    }
    .pile {
      min-height: 150px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      padding: 12px;
      border: 1px dashed rgba(255,255,255,.28);
      border-radius: 8px;
      background: rgba(255,255,255,.06);
    }
    .pile-area {
      display: grid;
      gap: 8px;
    }
    .pile-title {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      color: rgba(255,255,255,.88);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0;
    }
    .pile-note {
      color: rgba(255,255,255,.64);
      font-size: 12px;
      font-weight: 650;
    }
    .player.human .pile {
      min-height: 180px;
      border: 2px solid rgba(255,255,255,.56);
      background: rgba(255,255,255,.12);
    }
    .center {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 12px;
      color: rgba(255,255,255,.86);
    }
    .status {
      text-align: center;
      padding: 12px 18px;
      border-radius: 8px;
      background: rgba(0,0,0,.22);
      min-width: min(360px, 100%);
      font-weight: 750;
    }
    .card {
      width: 72px;
      aspect-ratio: 5 / 7;
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 4px;
      font-size: 30px;
      font-weight: 900;
      border: 2px solid rgba(255,255,255,.72);
      box-shadow: 0 8px 16px rgba(0,0,0,.18);
      flex: 0 0 auto;
      user-select: none;
    }
    .player.human .card { width: 84px; }
    .card-symbol { line-height: 1; }
    .card-name {
      font-size: 13px;
      line-height: 1;
      font-weight: 850;
    }
    .back { background: repeating-linear-gradient(45deg, #26313a, #26313a 7px, #384652 7px, #384652 14px); color: #fff; }
    .flower { background: #fff8ec; color: var(--green); }
    .skull { background: #fff2ef; color: var(--red); }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .section {
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    .section:last-child { border-bottom: 0; }
    h2 {
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0;
    }
    .actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .actions button { width: 100%; }
    .log {
      height: 330px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 14px;
      line-height: 1.35;
    }
    .entry {
      padding: 8px 10px;
      border-radius: 8px;
      background: #f1eadb;
      color: #27302f;
    }
    .entry.important {
      background: #e4efe8;
      border: 1px solid #c3dccb;
    }
    .small {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    @media (max-width: 840px) {
      header { align-items: flex-start; flex-direction: column; }
      .top-actions { justify-content: flex-start; }
      .grid { grid-template-columns: 1fr; }
      .table { min-height: 520px; }
      .player { grid-template-columns: 1fr; }
      .center { grid-template-columns: 1fr; }
      .status { min-width: 0; }
      .card { width: 58px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Skull CFR</h1>
      <div class="top-actions">
        <select id="policyMode" class="policy-select" title="选择 CPU 策略">
          <option value="compact">Compact AI</option>
          <option value="perfect">Perfect AI</option>
        </select>
        <button id="newP0" class="primary">新局：我先手</button>
        <button id="newP1">新局：CPU 先手</button>
        <button id="cheatToggle" class="toggle" type="button">作弊模式：关</button>
      </div>
    </header>
    <div class="grid">
      <section class="table">
        <div class="player cpu">
          <div>
            <div class="name">CPU</div>
            <div class="score" id="cpuScore">0</div>
            <div class="meta">
              <div id="cpuCards">总牌 4</div>
              <div id="cpuPileCount">牌堆 0</div>
              <div id="cpuPolicy">策略 compact</div>
            </div>
          </div>
          <div class="pile-area">
            <div class="pile-title">
              <span>CPU 牌堆</span>
              <span class="pile-note" id="cpuPileMode">隐藏</span>
            </div>
            <div class="pile" id="cpuPile"></div>
          </div>
        </div>
        <div class="center">
          <div></div>
          <div class="status" id="status">加载中</div>
          <div></div>
        </div>
        <div class="player human">
          <div>
            <div class="name">你</div>
            <div class="score" id="humanScore">0</div>
            <div class="meta">
              <div id="humanHand">花 3 / 骷髅 1</div>
              <div id="humanCards">总牌 4</div>
            </div>
          </div>
          <div class="pile-area">
            <div class="pile-title">
              <span>你的牌堆</span>
              <span class="pile-note">只有你知道自己放了什么</span>
            </div>
            <div class="pile" id="humanPile"></div>
          </div>
        </div>
      </section>
      <aside class="panel">
        <div class="section">
          <h2>操作</h2>
          <div class="actions" id="actions"></div>
          <p class="small" id="hint"></p>
          <p class="small">新规则：双方都至少放过一张牌后，才允许开始叫号。</p>
        </div>
        <div class="section">
          <h2>记录</h2>
          <div class="log" id="log"></div>
        </div>
      </aside>
    </div>
  </main>
  <script>
    let sessionId = null;
    let cheatMode = false;
    let lastData = null;

    const el = id => document.getElementById(id);
    const actionNames = {
      PLAY_F: "放花",
      PLAY_S: "放骷髅",
      PASS: "放弃"
    };

    function label(action) {
      if (actionNames[action]) return actionNames[action];
      if (action.startsWith("BID_")) return `叫 ${action.slice(4)}`;
      return action;
    }

    function cardNode(kind, hidden=false) {
      const div = document.createElement("div");
      div.className = hidden ? "card back" : `card ${kind === "S" ? "skull" : "flower"}`;
      if (!hidden) {
        const symbol = document.createElement("span");
        symbol.className = "card-symbol";
        symbol.textContent = kind === "S" ? "☠" : "✿";
        const name = document.createElement("span");
        name.className = "card-name";
        name.textContent = kind === "S" ? "骷髅" : "花";
        div.append(symbol, name);
      }
      return div;
    }

    async function api(path, body=null) {
      const res = await fetch(path, {
        method: body ? "POST" : "GET",
        headers: body ? {"Content-Type": "application/json"} : {},
        body: body ? JSON.stringify(body) : null
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function newGame(human) {
      const data = await api("/api/new", {human, policy_mode: el("policyMode").value});
      sessionId = data.session;
      render(data);
    }

    async function act(action) {
      if (!sessionId) return;
      const data = await api("/api/action", {session: sessionId, action});
      render(data);
    }

    function render(data) {
      lastData = data;
      sessionId = data.session;
      document.body.classList.toggle("cheat", cheatMode);
      el("cheatToggle").textContent = cheatMode ? "作弊模式：开" : "作弊模式：关";
      el("cheatToggle").classList.toggle("active", cheatMode);
      el("policyMode").value = data.policy_mode || el("policyMode").value;
      el("cpuScore").textContent = data.cpu.score;
      el("humanScore").textContent = data.human.score;
      el("cpuCards").textContent = `总牌 ${data.cpu.total_cards}`;
      el("cpuPileCount").textContent = `牌堆 ${data.cpu.pile_count}`;
      el("cpuPileMode").textContent = cheatMode ? "已透视" : "隐藏";
      el("cpuPolicy").textContent = `策略 ${data.policy_mode}`;
      el("humanHand").textContent = `花 ${data.human.hand.flowers} / 骷髅 ${data.human.hand.skulls}`;
      el("humanCards").textContent = `总牌 ${data.human.total_cards}`;
      el("status").textContent = data.status;

      const cpuPile = el("cpuPile");
      cpuPile.innerHTML = "";
      if (cheatMode) {
        for (const c of data.cpu.pile) cpuPile.appendChild(cardNode(c));
      } else {
        for (let i = 0; i < data.cpu.pile_count; i++) cpuPile.appendChild(cardNode("?", true));
      }

      const humanPile = el("humanPile");
      humanPile.innerHTML = "";
      for (const c of data.human.pile) humanPile.appendChild(cardNode(c));

      const actions = el("actions");
      actions.innerHTML = "";
      for (const action of data.legal_actions) {
        const b = document.createElement("button");
        b.textContent = label(action);
        b.className = action === "PLAY_S" ? "danger" : action === "PLAY_F" ? "good" : action.startsWith("BID_") ? "blue" : "";
        b.onclick = () => act(action);
        actions.appendChild(b);
      }
      if (!data.legal_actions.length) {
        const b = document.createElement("button");
        b.textContent = data.winner === null ? "等待 CPU" : "重新开始";
        b.className = "primary";
        b.onclick = () => newGame(data.human.id);
        actions.appendChild(b);
      }
      el("hint").textContent = data.hint;

      const log = el("log");
      log.innerHTML = "";
      for (const item of data.log.slice().reverse()) {
        const div = document.createElement("div");
        div.className = "entry" + (item.important ? " important" : "");
        div.textContent = item.text;
        log.appendChild(div);
      }
    }

    el("newP0").onclick = () => newGame(0);
    el("newP1").onclick = () => newGame(1);
    el("cheatToggle").onclick = () => {
      cheatMode = !cheatMode;
      if (lastData) render(lastData);
    };
    newGame(0).catch(err => {
      el("status").textContent = err.message;
    });
  </script>
</body>
</html>
"""


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
        seed: Optional[int],
    ) -> None:
        super().__init__(server_address, Handler)
        self.policies = policies
        self.seed = seed
        self.sessions: Dict[str, GameSession] = {}
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: SkullServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
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

    def send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
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
    server, port = bind_server(args, policies)
    url = f"http://{args.host}:{port}/"
    loaded = ", ".join(f"{name}:{recall}" for name, (_, recall) in policies.items())
    print(f"BoardArena skull UI: {url}")
    print(f"Loaded policies: {loaded}")
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
) -> tuple[SkullServer, int]:
    for port in range(args.port, args.port + 20):
        try:
            return SkullServer((args.host, port), policies, args.seed), port
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise OSError(errno.EADDRINUSE, f"ports {args.port}-{args.port + 19} are in use")


if __name__ == "__main__":
    raise SystemExit(main())
