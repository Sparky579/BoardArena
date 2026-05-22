#!/usr/bin/env python3
"""Tkinter human-vs-bot UI for BoardArena UNO."""

from __future__ import annotations

import argparse
import queue
import random
import sys
import threading
import traceback
from pathlib import Path
from tkinter import BOTH, DISABLED, END, LEFT, NORMAL, RIGHT, TOP, X, Button, Entry, Frame, Label, Listbox, OptionMenu
from tkinter import StringVar, Tk, Toplevel, messagebox
from typing import Any

from env.uno_env import COLORS, DEFAULT_MAX_PLIES, UnoEnv, card_label, load_bot


HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE / "baseline"
DEFAULT_BOT_ID = "/gpt/bot_hard"

CARD_COLORS = {
    "red": ("#d83a35", "#ffffff"),
    "yellow": ("#e4b92f", "#1d1d1d"),
    "green": ("#2f9d58", "#ffffff"),
    "blue": ("#2475d1", "#ffffff"),
    None: ("#24272f", "#ffffff"),
}


def discover_bot_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if not BASELINE_DIR.is_dir():
        return paths
    for pattern in ("*.py", "bot.py", "bot_*.py"):
        for path in sorted(BASELINE_DIR.rglob(pattern)):
            if "__pycache__" in path.parts or path.name.startswith("_"):
                continue
            bot_id = "/" + path.relative_to(BASELINE_DIR).with_suffix("").as_posix()
            paths[bot_id] = path
    return paths


class UnoTkApp:
    def __init__(self, root: Tk, *, bot_id: str | None, human_seat: int, seed: int | None) -> None:
        self.root = root
        self.root.title("BoardArena UNO")
        self.root.minsize(980, 680)

        self.bot_paths = discover_bot_paths()
        if not self.bot_paths:
            raise FileNotFoundError(f"no UNO bot files found under {BASELINE_DIR}")
        default_bot = bot_id if bot_id in self.bot_paths else DEFAULT_BOT_ID
        if default_bot not in self.bot_paths:
            default_bot = next(iter(self.bot_paths))

        self.bot_id_var = StringVar(value=default_bot)
        self.human_seat_var = StringVar(value=str(human_seat))
        self.seed_var = StringVar(value="" if seed is None else str(seed))
        self.status_var = StringVar(value="Ready")
        self.info_var = StringVar(value="")
        self.current_color_var = StringVar(value="")

        self.env: UnoEnv | None = None
        self.bot: Any | None = None
        self.forfeit: dict[str, Any] | None = None
        self.bot_worker: threading.Thread | None = None
        self.bot_queue: queue.Queue[tuple[bool, Any]] = queue.Queue()
        self.hand_buttons: list[Button] = []

        self._build_layout()
        self.new_game()

    @property
    def human_seat(self) -> int:
        return int(self.human_seat_var.get())

    @property
    def bot_seat(self) -> int:
        return 1 - self.human_seat

    def _build_layout(self) -> None:
        top = Frame(self.root, padx=12, pady=10)
        top.pack(side=TOP, fill=X)

        Label(top, text="Bot").pack(side=LEFT)
        OptionMenu(top, self.bot_id_var, *self.bot_paths.keys()).pack(side=LEFT, padx=(6, 14))
        Label(top, text="Human seat").pack(side=LEFT)
        OptionMenu(top, self.human_seat_var, "0", "1").pack(side=LEFT, padx=(6, 14))
        Label(top, text="Seed").pack(side=LEFT)
        Entry(top, textvariable=self.seed_var, width=10).pack(side=LEFT, padx=(6, 14))
        Button(top, text="New game", command=self.new_game).pack(side=LEFT)

        status = Frame(self.root, padx=12, pady=8)
        status.pack(side=TOP, fill=X)
        Label(status, textvariable=self.status_var, font=("Segoe UI", 14, "bold")).pack(side=LEFT)
        Label(status, textvariable=self.info_var, font=("Segoe UI", 10)).pack(side=RIGHT)

        center = Frame(self.root, padx=12, pady=12)
        center.pack(side=TOP, fill=X)
        self.opponent_label = Label(center, text="", width=24, anchor="w", font=("Segoe UI", 12))
        self.opponent_label.pack(side=LEFT)
        self.top_card_button = Button(center, text="?", width=10, height=5, state=DISABLED, font=("Segoe UI", 20, "bold"))
        self.top_card_button.pack(side=LEFT, padx=20)
        Label(center, text="Current color").pack(side=LEFT)
        self.color_label = Label(center, textvariable=self.current_color_var, width=12, font=("Segoe UI", 12, "bold"))
        self.color_label.pack(side=LEFT, padx=(8, 22))
        self.draw_button = Button(center, text="Draw", width=10, command=lambda: self.apply_human_action("draw"))
        self.draw_button.pack(side=LEFT, padx=4)
        self.pass_button = Button(center, text="Pass", width=10, command=lambda: self.apply_human_action("pass"))
        self.pass_button.pack(side=LEFT, padx=4)

        hand_area = Frame(self.root, padx=12, pady=12)
        hand_area.pack(side=TOP, fill=X)
        Label(hand_area, text="Your hand", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.hand_frame = Frame(hand_area)
        self.hand_frame.pack(fill=X, pady=(8, 0))

        bottom = Frame(self.root, padx=12, pady=12)
        bottom.pack(side=TOP, fill=BOTH, expand=True)
        Label(bottom, text="Log", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.log_box = Listbox(bottom, height=12)
        self.log_box.pack(fill=BOTH, expand=True, pady=(8, 0))

    def new_game(self) -> None:
        try:
            seed = self._parse_seed()
            self.bot = load_bot(self.bot_paths[self.bot_id_var.get()])
            self.env = UnoEnv(seed=seed, max_plies=DEFAULT_MAX_PLIES)
            self.forfeit = None
            self.log_box.delete(0, END)
            self._log(f"New game: human=P{self.human_seat}, bot=P{self.bot_seat}, bot={self.bot_id_var.get()}")
            self.render()
            self.maybe_start_bot()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("New game failed", str(exc))

    def apply_human_action(self, action: str) -> None:
        if self.env is None or self.forfeit is not None:
            return
        state = self.env.state(self.human_seat)
        if self.env.actor != self.human_seat or action not in state["legal_actions"]:
            return
        self._apply_action(action, source="human")
        self.render()
        self.maybe_start_bot()

    def maybe_start_bot(self) -> None:
        if self.env is None or self.bot is None or self.forfeit is not None:
            return
        if self.env.is_done() or self.env.actor != self.bot_seat:
            return
        if self.bot_worker is not None and self.bot_worker.is_alive():
            return
        state = self.env.state(self.bot_seat)
        self.status_var.set(f"Bot P{self.bot_seat} thinking...")
        self._set_controls_enabled(False)
        self.bot_worker = threading.Thread(target=self._bot_choose_worker, args=(state,), daemon=True)
        self.bot_worker.start()
        self.root.after(60, self._poll_bot_queue)

    def _bot_choose_worker(self, state: dict[str, Any]) -> None:
        try:
            self.bot_queue.put((True, self.bot.choose_action(state)))
        except BaseException as exc:  # noqa: BLE001
            self.bot_queue.put((False, exc))

    def _poll_bot_queue(self) -> None:
        if self.env is None:
            return
        try:
            ok, value = self.bot_queue.get_nowait()
        except queue.Empty:
            self.root.after(60, self._poll_bot_queue)
            return

        if ok:
            action = value
            legal = self.env.state(self.bot_seat)["legal_actions"]
            if isinstance(action, str) and action in legal:
                self._apply_action(action, source="bot")
            else:
                self.forfeit = {
                    "winner": self.human_seat,
                    "status": "invalid_action",
                    "error": f"bot returned {action!r}",
                }
                self._log(f"Bot invalid action: {action!r}")
        else:
            self.forfeit = {
                "winner": self.human_seat,
                "status": "bot_exception",
                "error": f"{type(value).__name__}: {value}",
            }
            self._log("Bot exception:")
            self._log("".join(traceback.format_exception_only(type(value), value)).strip())

        self.render()
        self.maybe_start_bot()

    def _apply_action(self, action: str, *, source: str) -> None:
        if self.env is None:
            return
        actor = self.env.actor
        self.env.step(action)
        label = self.env.last_action or action
        if self.env.last_draw_count:
            label += f" ({self.env.last_draw_count} drawn)"
        self._log(f"P{actor} {source}: {label}")
        if self.env.is_done():
            winner = self.env.winner
            if winner is None:
                self._log(f"Game over: {self.env.terminal_status}, draw")
            else:
                self._log(f"Game over: P{winner} wins ({self.env.terminal_status})")

    def render(self) -> None:
        if self.env is None:
            return
        state = self.env.state(self.human_seat)
        top_card = state["top_card"]
        bg, fg = CARD_COLORS.get(top_card["current_color"], CARD_COLORS[None])
        self.top_card_button.configure(text=top_card["label"], bg=bg, fg=fg, activebackground=bg)
        self.current_color_var.set(state["current_color"])
        color_bg, color_fg = CARD_COLORS.get(state["current_color"], CARD_COLORS[None])
        self.color_label.configure(bg=color_bg, fg=color_fg)
        self.opponent_label.configure(text=f"Opponent P{self.bot_seat}: {state['opponent_hand_count']} cards")
        self.info_var.set(
            f"P0 cards {state['hand_counts'][0]} | P1 cards {state['hand_counts'][1]} | "
            f"draw {state['draw_pile_count']} | discard {state['discard_pile_count']}"
        )

        self._render_hand(state)
        self._render_status(state)
        self._set_controls_enabled(self._human_can_act(state))

    def _render_hand(self, state: dict[str, Any]) -> None:
        for button in self.hand_buttons:
            button.destroy()
        self.hand_buttons = []

        can_act = self._human_can_act(state)
        for card in state["hand"]:
            actions = list(card["legal_actions"])
            bg, fg = CARD_COLORS.get(card["color"], CARD_COLORS[None])
            button = Button(
                self.hand_frame,
                text=card["label"],
                width=7,
                height=4,
                bg=bg,
                fg=fg,
                activebackground=bg,
                font=("Segoe UI", 13, "bold"),
                command=lambda card=card, actions=actions: self._choose_card_action(card, actions),
            )
            button.pack(side=LEFT, padx=4, pady=4)
            if not can_act or not actions:
                button.configure(state=DISABLED)
            self.hand_buttons.append(button)

    def _render_status(self, state: dict[str, Any]) -> None:
        if self.forfeit is not None:
            self.status_var.set(f"P{self.forfeit['winner']} wins ({self.forfeit['status']})")
        elif state["phase"] == "game_over":
            if state["winner"] is None:
                self.status_var.set(f"Game over: {state['status']}")
            else:
                self.status_var.set(f"P{state['winner']} wins ({state['status']})")
        elif self.env and self.env.actor == self.human_seat:
            self.status_var.set(f"Your turn (P{self.human_seat})")
        elif self.env:
            self.status_var.set(f"Bot turn (P{self.bot_seat})")

    def _set_controls_enabled(self, enabled: bool) -> None:
        if self.env is None:
            return
        state = self.env.state(self.human_seat)
        legal = set(state["legal_actions"]) if enabled else set()
        self.draw_button.configure(state=NORMAL if "draw" in legal else DISABLED)
        self.pass_button.configure(state=NORMAL if "pass" in legal else DISABLED)
        for button in self.hand_buttons:
            if not enabled:
                button.configure(state=DISABLED)

    def _human_can_act(self, state: dict[str, Any]) -> bool:
        return (
            self.env is not None
            and self.forfeit is None
            and state["phase"] != "game_over"
            and self.env.actor == self.human_seat
        )

    def _choose_card_action(self, card: dict[str, Any], actions: list[str]) -> None:
        if len(actions) == 1:
            self.apply_human_action(actions[0])
            return
        if len(actions) > 1:
            self._show_color_picker(card, actions)

    def _show_color_picker(self, card: dict[str, Any], actions: list[str]) -> None:
        dialog = Toplevel(self.root)
        dialog.title(f"Choose color for {card['label']}")
        dialog.resizable(False, False)
        Label(dialog, text=f"Choose color for {card['label']}", padx=12, pady=10).pack(fill=X)
        row = Frame(dialog, padx=10, pady=10)
        row.pack(fill=X)
        by_color = {action.split(":")[2]: action for action in actions}
        for color in COLORS:
            bg, fg = CARD_COLORS[color]
            button = Button(
                row,
                text=color.title(),
                width=10,
                bg=bg,
                fg=fg,
                command=lambda action=by_color[color]: self._pick_color_action(dialog, action),
            )
            button.pack(side=LEFT, padx=4)
        dialog.transient(self.root)
        dialog.grab_set()

    def _pick_color_action(self, dialog: Toplevel, action: str) -> None:
        dialog.destroy()
        self.apply_human_action(action)

    def _parse_seed(self) -> int | None:
        text = self.seed_var.get().strip()
        if not text:
            return None
        return int(text)

    def _log(self, text: str) -> None:
        self.log_box.insert(END, text)
        self.log_box.yview_moveto(1.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BoardArena UNO Tkinter human-vs-bot UI")
    parser.add_argument("--bot", default=DEFAULT_BOT_ID, help="bot id from uno/baseline, for example /gpt/bot_hard")
    parser.add_argument("--human-seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Tk()
    UnoTkApp(root, bot_id=args.bot, human_seat=args.human_seat, seed=args.seed)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
