"""Import chess games from text.

Supports two formats:

  1. Standard PGN (tags + space-separated SAN move list, optional move
     numbers, optional result token at the end).

  2. The web UI's per-line replay format used by chess_web.py's log:

         黑方 Bot: Qe7#
         白方: Ke8
         黑方 Bot: Kd6
         ...
         白方: d4         <-- opening move

     Each line is one half-move in SAN; lines may be prefixed with
     "白方 [Bot]: " / "黑方 [Bot]: " / "白:" / "黑:" etc. The list is in
     **reverse chronological order** (most recent move first), so it
     must be reversed before being applied. Side prefixes (when present)
     are used to sanity-check the alternation after reversal.

Both parsers return a tuple ``(board, moves)`` where:
  - ``board`` is a final-position ``chess.Board`` with the moves played
    (so ``board.fen()`` is the resulting FEN and ``board.move_stack``
    contains the chess.Move objects in play order).
  - ``moves`` is a list of UCI strings in chronological order.

If parsing fails (bad SAN, illegal move, color mismatch), the parsers
raise ``ImportError`` with a 1-indexed line/move number in the message.
"""

from __future__ import annotations

import io
import re

import chess
import chess.pgn


# ---------- PGN ----------


def parse_pgn(text: str) -> tuple[chess.Board, list[str]]:
    """Parse standard PGN text.

    Picks up the first game in the input (so multi-game PGN files work
    — we just use the first). Tags like [FEN ".."] are honored as the
    starting position.
    """
    stream = io.StringIO(text)
    game = chess.pgn.read_game(stream)
    if game is None:
        raise ImportError("PGN 解析失败：找不到对局")

    board = game.board()
    moves: list[str] = []
    for i, move in enumerate(game.mainline_moves(), start=1):
        if move not in board.legal_moves:
            raise ImportError(f"PGN 第 {i} 步非法：{move.uci()}")
        moves.append(move.uci())
        board.push(move)
    return board, moves


# ---------- replay-log format (per-line, reverse chronological) ----------


_PREFIX_RE = re.compile(
    r"""^\s*
    (?:
        (?P<side>白方|黑方|白|黑|White|Black)
        \s*
        (?:Bot|bot|玩家)?
        \s*[:：]?
    )?
    \s*(?P<san>.+?)\s*$
    """,
    re.VERBOSE,
)

# Pure SAN check: filters out non-move lines (headers, status notes).
_SAN_LIKE = re.compile(
    r"""^
    (?:O-O(?:-O)?[+#]?
     | [KQRBN][a-h1-8]?x?[a-h][1-8][+#]?
     | [a-h]?x[a-h][1-8](?:=[QRBN])?[+#]?
     | [a-h][1-8](?:=[QRBN])?[+#]?)
    $""",
    re.VERBOSE,
)

_SIDE_WHITE = {"白方", "白", "White"}
_SIDE_BLACK = {"黑方", "黑", "Black"}


def _parse_replay_lines(text: str) -> list[tuple[str | None, str]]:
    """Return list of (side, san) in *file order* (still reverse chrono).

    side is "white" / "black" / None. Lines that don't look like a move
    are skipped so the parser is robust to noise (status banners,
    separators, blank lines).
    """
    out: list[tuple[str | None, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("#", "//")):
            continue
        m = _PREFIX_RE.match(line)
        if not m:
            continue
        side_raw = m.group("side")
        san_part = m.group("san").strip()
        if san_part in {"1-0", "0-1", "1/2-1/2", "*"}:
            continue
        san_part = re.sub(r"\{[^}]*\}", "", san_part).strip()
        san_part = re.sub(r"\s+[!?]+$", "", san_part).strip()
        if not san_part or not _SAN_LIKE.match(san_part):
            continue
        if side_raw in _SIDE_WHITE:
            side: str | None = "white"
        elif side_raw in _SIDE_BLACK:
            side = "black"
        else:
            side = None
        out.append((side, san_part))
    return out


def parse_replay_log(text: str) -> tuple[chess.Board, list[str]]:
    """Parse the per-line replay format. Auto-detects ordering:

      - If the first prefixed line is 白方 (white), the input is
        chronological (top = ply 1, bottom = last move). This matches
        the on-screen move-log UI, so copy-paste from the running
        session "just works".
      - If the first prefixed line is 黑方 (black), the input is
        reverse-chronological (top = last move, bottom = opening).
        This matches the older sample shape we shipped with.
      - If no prefix is present at all, treat as chronological.
    """
    lines = _parse_replay_lines(text)
    if not lines:
        raise ImportError("复盘文本解析失败：未找到任何走子")

    first_side = next((side for side, _ in lines if side is not None), None)
    if first_side == "black":
        plays = list(reversed(lines))
    else:
        plays = list(lines)

    board = chess.Board()
    moves: list[str] = []
    for i, (side, san) in enumerate(plays, start=1):
        expected_side = "white" if board.turn == chess.WHITE else "black"
        if side is not None and side != expected_side:
            raise ImportError(
                f"复盘第 {i} 步标记为 {side} 但当前应当是 {expected_side}：{san}"
            )
        try:
            move = board.parse_san(san)
        except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError,
                chess.AmbiguousMoveError) as exc:
            raise ImportError(f"复盘第 {i} 步 SAN 无法解析：{san}（{exc}）") from None
        moves.append(move.uci())
        board.push(move)
    return board, moves


# ---------- top-level dispatch ----------


def parse_game(text: str) -> tuple[chess.Board, list[str], str]:
    """Auto-detect format and parse.

    Returns ``(board, moves, format)`` where ``format`` is ``"pgn"`` or
    ``"replay"``. The detector looks for PGN tag lines (``[Event "..."]``
    style) or a numeric ``1.`` move-number prefix; everything else is
    treated as replay-log format.
    """
    sniff = text.lstrip()
    if sniff.startswith("[") or re.match(r"^1\.\s", sniff):
        try:
            return (*parse_pgn(text), "pgn")
        except ImportError:
            # Fall through to replay parser; some PGN-ish inputs are
            # actually mis-detected (e.g. SAN starting with "1." followed
            # by something weird). The replay parser is more forgiving.
            pass
    return (*parse_replay_log(text), "replay")
