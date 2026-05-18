"""Gym-style chess environment and bot battle API.

Rules are delegated to python-chess so legal move generation, castling,
en-passant, promotion, check, checkmate, stalemate, and draw rules stay aligned
with standard chess instead of a local reimplementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import random
import sys
import threading
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import chess
except ImportError as exc:  # pragma: no cover - exercised only without deps.
    raise ImportError(
        "BoardArena/chess requires python-chess. Install it with: "
        "python -m pip install -r BoardArena/chess/requirements.txt"
    ) from exc


SUPPORTED_PLAYERS = 2
DEFAULT_MAX_PLIES = 512

_MATCH_LOGS: dict[str, list[str]] = {}


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class IllegalActionError(ValueError):
    """Raised when an action is not a legal UCI move in the current position."""


class BotTimeoutError(TimeoutError):
    """Raised when a bot does not return an action before the deadline."""


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def as_tuple(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self.observation, self.reward, self.terminated, self.truncated, self.info


class ChessEnv:
    """A small alternating-turn Gym-style environment for two-player chess."""

    metadata = {"render_modes": ["ansi"], "players": SUPPORTED_PLAYERS}

    def __init__(
        self,
        *,
        fen: str = chess.STARTING_FEN,
        seed: int | None = None,
        max_plies: int | None = DEFAULT_MAX_PLIES,
        claim_draw: bool = True,
    ) -> None:
        self.initial_fen = fen
        self.max_plies = max_plies
        self.claim_draw = claim_draw
        self.rng = random.Random(seed)
        self.board = chess.Board(fen)
        self.plies = 0
        self.last_move: chess.Move | None = None
        self.san_history: list[str] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        fen: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        self.board = chess.Board(fen or self.initial_fen)
        self.plies = 0
        self.last_move = None
        self.san_history = []
        return self.state(), {"fen": self.board.fen()}

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.is_done():
            outcome = self._outcome_info()
            return self.state(), 0.0, True, False, outcome

        legal = self.legal_actions()
        if action not in legal:
            raise IllegalActionError(f"{action!r} is not legal in the current position")

        actor = self.actor
        move = chess.Move.from_uci(action)
        san = self.board.san(move)
        self.board.push(move)
        self.last_move = move
        self.san_history.append(san)
        self.plies += 1

        terminated = self.is_done()
        truncated = bool(self.max_plies is not None and self.plies >= self.max_plies and not terminated)
        info = self._outcome_info() if terminated or truncated else {}
        reward = self._reward_for(actor, info)
        return self.state(), reward, terminated, truncated, info

    @property
    def actor(self) -> int:
        return 0 if self.board.turn == chess.WHITE else 1

    def legal_actions(self) -> list[str]:
        return [move.uci() for move in self.board.legal_moves]

    def is_done(self) -> bool:
        return self.board.is_game_over(claim_draw=self.claim_draw)

    def state(self, player_id: int | None = None) -> dict[str, Any]:
        outcome = self.board.outcome(claim_draw=self.claim_draw)
        winner = _winner_id(outcome)
        status = _termination_status(outcome)
        actor = self.actor
        phase = "game_over" if outcome is not None else "turn"
        legal = [] if outcome is not None else self.legal_actions()

        return {
            "player_id": actor if player_id is None else player_id,
            "num_players": SUPPORTED_PLAYERS,
            "phase": phase,
            "actor": actor,
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "legal_actions": legal,
            "fen": self.board.fen(),
            "board_fen": self.board.board_fen(),
            "pieces": self._pieces(),
            "castling_rights": self.board.castling_xfen(),
            "en_passant_square": chess.square_name(self.board.ep_square) if self.board.ep_square is not None else None,
            "halfmove_clock": self.board.halfmove_clock,
            "fullmove_number": self.board.fullmove_number,
            "plies": self.plies,
            "check": self.board.is_check(),
            "last_move": self.last_move.uci() if self.last_move else None,
            "san_history": list(self.san_history),
            "winner": winner,
            "status": status,
            "result": self.board.result(claim_draw=self.claim_draw) if outcome is not None else "*",
        }

    def render(self) -> str:
        return str(self.board)

    def _pieces(self) -> list[dict[str, str]]:
        pieces: list[dict[str, str]] = []
        for square, piece in self.board.piece_map().items():
            pieces.append(
                {
                    "square": chess.square_name(square),
                    "type": piece.symbol().lower(),
                    "color": "white" if piece.color == chess.WHITE else "black",
                    "symbol": piece.symbol(),
                }
            )
        return sorted(pieces, key=lambda item: item["square"])

    def _outcome_info(self) -> dict[str, Any]:
        outcome = self.board.outcome(claim_draw=self.claim_draw)
        if outcome is None:
            if self.max_plies is not None and self.plies >= self.max_plies:
                return {
                    "status": "turn_limit",
                    "winner": None,
                    "result": "*",
                    "termination": "turn_limit",
                }
            return {}
        return {
            "status": _termination_status(outcome),
            "winner": _winner_id(outcome),
            "result": self.board.result(claim_draw=self.claim_draw),
            "termination": outcome.termination.name.lower(),
        }

    @staticmethod
    def _reward_for(actor: int, info: dict[str, Any]) -> float:
        winner = info.get("winner")
        if winner is None:
            return 0.0
        return 1.0 if winner == actor else -1.0


class SystemBot:
    name = "system"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_action(self, state: dict[str, Any]) -> str:
        legal = state["legal_actions"]
        if not legal:
            raise ValueError("system bot received no legal actions")

        board = chess.Board(state["fen"])
        scored: list[tuple[int, str]] = []
        for action in legal:
            move = chess.Move.from_uci(action)
            score = 0
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured is None and board.is_en_passant(move):
                    captured = chess.Piece(chess.PAWN, not board.turn)
                mover = board.piece_at(move.from_square)
                score += 10 * PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
                score -= PIECE_VALUES.get(mover.piece_type, 0) // 10 if mover else 0
            if move.promotion:
                score += PIECE_VALUES.get(move.promotion, 0)
            board.push(move)
            if board.is_checkmate():
                score += 100_000
            elif board.is_check():
                score += 25
            board.pop()
            scored.append((score, action))

        best_score = max(score for score, _ in scored)
        best = [action for score, action in scored if score == best_score]
        return self.rng.choice(sorted(best))


class CallableBot:
    def __init__(self, choose_action: Callable[[dict[str, Any]], str], name: str) -> None:
        self.choose_action = choose_action
        self.name = name


def choose_action_with_timeout(
    bot: Any,
    state: dict[str, Any],
    decision_timeout: float | None,
) -> Any:
    if decision_timeout is None:
        return bot.choose_action(state)

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            result_queue.put((True, bot.choose_action(state)))
        except BaseException as exc:  # noqa: BLE001 - preserve existing bot failure semantics.
            result_queue.put((False, exc))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(decision_timeout)
    if thread.is_alive():
        raise BotTimeoutError(f"choose_action exceeded {decision_timeout:g} seconds")

    ok, value = result_queue.get_nowait()
    if ok:
        return value
    raise value


def validate_decision_timeout(decision_timeout: float | None) -> None:
    if decision_timeout is not None and decision_timeout <= 0:
        raise ValueError("decision_timeout must be positive seconds or None")


def load_bot(bot_path: str | Path) -> Any:
    path = Path(bot_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"bot file not found: {path}")

    module_name = f"chess_user_bot_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import bot file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, "Bot"):
        bot = module.Bot()
        if not hasattr(bot, "choose_action"):
            raise AttributeError("Bot must define choose_action(state)")
        if not hasattr(bot, "name"):
            bot.name = path.stem
        return bot

    if hasattr(module, "choose_action"):
        return CallableBot(module.choose_action, getattr(module, "name", path.stem))

    raise AttributeError("bot.py must define choose_action(state) or class Bot")


def battle_once(
    bot_path: str | Path,
    *,
    players: int = SUPPORTED_PLAYERS,
    seat: int = 0,
    seed: int | None = None,
    keep_log: bool = True,
    max_plies: int = DEFAULT_MAX_PLIES,
    fen: str = chess.STARTING_FEN,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("chess only supports players=2")
    if seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    validate_decision_timeout(decision_timeout)

    game_seed = seed if seed is not None else random.randrange(1 << 30)
    rng = random.Random(game_seed)
    game_id = uuid.uuid4().hex[:12]
    log: list[str] = [f"G:{game_id}:N2:SEED{game_seed}:FEN:{fen}"]

    developer_bot = load_bot(bot_path)
    bots: list[Any] = [SystemBot(rng), SystemBot(rng)]
    bots[seat] = developer_bot
    bot_names = [getattr(bot, "name", f"bot_{index}") for index, bot in enumerate(bots)]

    env = ChessEnv(fen=fen, seed=game_seed, max_plies=max_plies)
    status = "ok"
    error: str | None = None

    while True:
        state = env.state(env.actor)
        actor = state["actor"]
        legal_actions = state["legal_actions"]

        if state["phase"] == "game_over":
            break
        if env.max_plies is not None and env.plies >= env.max_plies:
            status = "turn_limit"
            break
        if not legal_actions:
            status = "no_legal_actions"
            break

        try:
            action = choose_action_with_timeout(bots[actor], state, decision_timeout)
        except BotTimeoutError as exc:
            status = "timeout"
            error = f"player {actor} timed out: {exc}"
            log.append(f"T{env.plies}:ERR:P{actor}:TIMEOUT")
            break
        except Exception as exc:  # noqa: BLE001 - bot exceptions are match results.
            status = "bot_exception"
            error = f"{type(exc).__name__}: {exc}"
            log.append(f"T{env.plies}:ERR:P{actor}:EXCEPTION:{type(exc).__name__}")
            break

        if not isinstance(action, str) or action not in legal_actions:
            status = "invalid_action"
            error = f"invalid action from player {actor}: {action!r}"
            log.append(f"T{env.plies}:ERR:P{actor}:INVALID:{action}")
            break

        san = env.board.san(chess.Move.from_uci(action))
        _, _, terminated, truncated, info = env.step(action)
        log.append(f"T{env.plies}:P{actor}:MOVE:{action}:SAN:{san}:FEN:{env.board.fen()}")
        if terminated or truncated:
            if truncated:
                status = "turn_limit"
            else:
                status = info.get("status", "ok")
            break

    final_state = env.state()
    winner = final_state["winner"]
    if status in {"bot_exception", "invalid_action", "timeout"}:
        loser = env.actor
        winner = 1 - loser
    elif status == "turn_limit":
        winner = None
    elif status not in {"ok", "turn_limit"} and final_state["status"] is not None:
        status = final_state["status"]

    log.append(
        "END:"
        f"{status}:WINNER:{winner}:RESULT:{final_state['result']}:"
        f"PLIES:{env.plies}:FEN:{env.board.fen()}"
    )

    if keep_log:
        _MATCH_LOGS[game_id] = log

    developer_win = winner == seat
    return {
        "game_id": game_id,
        "winner": winner,
        "status": status,
        "result": final_state["result"],
        "plies": env.plies,
        "fen": env.board.fen(),
        "bot_names": bot_names,
        "developer_seat": seat,
        "developer_win": developer_win,
        "error": error,
    }


def battle_many(
    bot_path: str | Path,
    *,
    games: int = 100,
    players: int = SUPPORTED_PLAYERS,
    seat: int = 0,
    seed: int | None = None,
    alternate_seats: bool = True,
    keep_logs: bool = False,
    max_plies: int = DEFAULT_MAX_PLIES,
    fen: str = chess.STARTING_FEN,
    decision_timeout: float | None = None,
) -> dict[str, Any]:
    if players != SUPPORTED_PLAYERS:
        raise ValueError("chess only supports players=2")
    if games < 1:
        raise ValueError("games must be >= 1")
    validate_decision_timeout(decision_timeout)

    wins_by_seat = [0, 0]
    statuses: Counter[str] = Counter()
    developer_wins = 0
    developer_losses = 0
    draws = 0
    game_ids: list[str] = []

    for index in range(games):
        game_seat = (seat + index) % 2 if alternate_seats else seat
        game_seed = None if seed is None else seed + index
        result = battle_once(
            bot_path,
            players=players,
            seat=game_seat,
            seed=game_seed,
            keep_log=keep_logs,
            max_plies=max_plies,
            fen=fen,
            decision_timeout=decision_timeout,
        )
        statuses[result["status"]] += 1
        if result["winner"] in (0, 1):
            wins_by_seat[result["winner"]] += 1
        else:
            draws += 1
        if result["developer_win"]:
            developer_wins += 1
        else:
            developer_losses += 1
        if keep_logs:
            game_ids.append(result["game_id"])

    summary = {
        "games": games,
        "players": players,
        "wins_by_seat": wins_by_seat,
        "draws": draws,
        "developer_wins": developer_wins,
        "developer_losses": developer_losses,
        "developer_win_rate": developer_wins / games,
        "statuses": dict(statuses),
    }
    if keep_logs:
        summary["game_ids"] = game_ids
    return summary


def get_match_log(game_id: str) -> list[str]:
    return list(_MATCH_LOGS.get(game_id, []))


def write_sample_bot(path: str | Path = "bot.py") -> None:
    sample = '''def choose_action(state):
    legal = state["legal_actions"]
    captures = []
    for action in legal:
        # UCI moves are strings like e2e4, e7e8q, or e1g1.
        if len(action) >= 4:
            captures.append(action)
    return sorted(captures or legal)[0]
'''
    Path(path).write_text(sample, encoding="utf-8")


def _winner_id(outcome: chess.Outcome | None) -> int | None:
    if outcome is None or outcome.winner is None:
        return None
    return 0 if outcome.winner == chess.WHITE else 1


def _termination_status(outcome: chess.Outcome | None) -> str | None:
    if outcome is None:
        return None
    return outcome.termination.name.lower()


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chess Gym-style env and bot battle referee")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample-bot", help="write a sample bot.py")
    sample_parser.add_argument("--output", default="bot.py")

    battle_parser = subparsers.add_parser("battle", help="run one or many games")
    battle_parser.add_argument("--bot", required=True)
    battle_parser.add_argument("--players", type=int, default=SUPPORTED_PLAYERS)
    battle_parser.add_argument("--games", type=int, default=1)
    battle_parser.add_argument("--seat", type=int, default=0)
    battle_parser.add_argument("--seed", type=int, default=None)
    battle_parser.add_argument("--keep-logs", action="store_true")
    battle_parser.add_argument("--fixed-seat", action="store_true")
    battle_parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    battle_parser.add_argument("--fen", default=chess.STARTING_FEN)
    battle_parser.add_argument("--decision-timeout", type=float, default=None)

    args = parser.parse_args(argv)

    try:
        if args.command == "sample-bot":
            write_sample_bot(args.output)
            print(f"wrote {args.output}")
            return 0

        if args.games == 1:
            _print_json(
                battle_once(
                    args.bot,
                    players=args.players,
                    seat=args.seat,
                    seed=args.seed,
                    keep_log=args.keep_logs,
                    max_plies=args.max_plies,
                    fen=args.fen,
                    decision_timeout=args.decision_timeout,
                )
            )
        else:
            _print_json(
                battle_many(
                    args.bot,
                    games=args.games,
                    players=args.players,
                    seat=args.seat,
                    seed=args.seed,
                    alternate_seats=not args.fixed_seat,
                    keep_logs=args.keep_logs,
                    max_plies=args.max_plies,
                    fen=args.fen,
                    decision_timeout=args.decision_timeout,
                )
            )
        return 0
    except Exception:  # noqa: BLE001 - command line should show the failure.
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
