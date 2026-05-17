# 国际象棋 Bot 对战接口

本文档描述 `env/chess_env.py` 提供的标准化接口。Bot 需要根据当前局面返回一个合法 UCI 动作。

## bot.py 标准格式

最小可用版本：

```python
def choose_action(state):
    return state["legal_actions"][0]
```

也可以写成类：

```python
class Bot:
    name = "my_chess_bot"

    def choose_action(self, state):
        return state["legal_actions"][0]
```

约束：

- `choose_action(state)` 必须返回一个字符串动作。
- 返回值必须在 `state["legal_actions"]` 里。
- 动作使用 UCI 格式，例如 `e2e4`、`e1g1`、`e7e8q`。
- 国际象棋是完全信息游戏，bot 可以看到完整棋盘、FEN、合法动作和历史 SAN。
- 如果 bot 抛异常或返回非法动作，本局会以 `bot_exception` 或 `invalid_action` 结束，并判该 bot 负。

## state 字段

`choose_action(state)` 收到的是一个 `dict`：

```python
{
    "player_id": 0,
    "num_players": 2,
    "phase": "turn",
    "actor": 0,
    "turn": "white",
    "legal_actions": ["g1h3", "g1f3", "b1c3", "b1a3", "e2e4"],
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "board_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    "pieces": [
        {"square": "a1", "type": "r", "color": "white", "symbol": "R"}
    ],
    "castling_rights": "KQkq",
    "en_passant_square": None,
    "halfmove_clock": 0,
    "fullmove_number": 1,
    "plies": 0,
    "check": False,
    "last_move": None,
    "san_history": [],
    "winner": None,
    "status": None,
    "result": "*"
}
```

字段说明：

- `player_id`：当前收到状态的 bot 座位号，白方为 `0`，黑方为 `1`。
- `num_players`：固定为 `2`。
- `phase`：`turn` 或 `game_over`。
- `actor`：当前行动玩家，白方 `0`，黑方 `1`。
- `turn`：`white` 或 `black`。
- `legal_actions`：当前行动玩家的所有合法 UCI 动作。
- `fen`：完整 FEN。
- `board_fen`：仅棋盘部分的 FEN。
- `pieces`：棋子列表，便于 UI 或不想解析 FEN 的 bot 使用。
- `castling_rights`：当前易位权利。
- `en_passant_square`：可吃过路兵目标格，没有则为 `None`。
- `halfmove_clock`：50 回合规则计数。
- `fullmove_number`：完整回合数。
- `plies`：已经执行的半回合数。
- `check`：当前行动方是否被将军。
- `last_move`：上一手 UCI 动作。
- `san_history`：SAN 记谱历史。
- `winner`：白胜为 `0`，黑胜为 `1`，和棋或未结束为 `None`。
- `status`：终局状态，未结束为 `None`。
- `result`：`1-0`、`0-1`、`1/2-1/2` 或 `*`。

## Python API

### Gym 风格环境

```python
from env.chess_env import ChessEnv

env = ChessEnv(seed=1)
state, info = env.reset()
state, reward, terminated, truncated, info = env.step("e2e4")
```

### 单次对战

```python
from env.chess_env import battle_once

result = battle_once(
    "baseline/gpt5p5/bot_hard.py",
    players=2,
    seat=0,
    seed=1,
    keep_log=True,
)
print(result)
```

内置 baseline 包含 `bot_easy.py`、`bot_medium.py` 和 `bot_hard.py`。其中 `bot_medium.py` 是旧 hard 的保留版本，`bot_hard.py` 是更强的传统搜索版本。

返回示例：

```python
{
    "game_id": "a1b2c3d4e5f6",
    "winner": 0,
    "status": "checkmate",
    "result": "1-0",
    "plies": 73,
    "fen": "8/8/8/8/8/8/8/8 b - - 0 37",
    "bot_names": ["gpt5p5_hard", "system"],
    "developer_seat": 0,
    "developer_win": True,
    "error": None
}
```

### 批量对战

```python
from env.chess_env import battle_many

summary = battle_many(
    "baseline/gpt5p5/bot_easy.py",
    games=100,
    seed=1,
    alternate_seats=True,
)
print(summary)
```

返回字段：

- `games`：总局数。
- `players`：固定为 `2`。
- `wins_by_seat`：白方、黑方胜场。
- `draws`：和棋数。
- `developer_wins`：开发者 bot 胜场。
- `developer_losses`：开发者 bot 未胜场，包含和棋。
- `developer_win_rate`：开发者 bot 胜率。
- `statuses`：终局状态统计。
- `game_ids`：仅当 `keep_logs=True` 时返回可查询日志的 id。

### 查询对战日志

```python
from env.chess_env import battle_once, get_match_log

result = battle_once("baseline/gpt5p5/bot_hard.py", keep_log=True)
log = get_match_log(result["game_id"])
for item in log:
    print(item)
```

日志是简单字符串编码：

- `G:<game_id>:N2:SEED<seed>:FEN:<fen>`：一局开始。
- `T<t>:P<p>:MOVE:<uci>:SAN:<san>:FEN:<fen>`：玩家走子。
- `T<t>:ERR:P<p>:INVALID:<action>`：玩家返回非法动作。
- `T<t>:ERR:P<p>:EXCEPTION:<type>`：玩家 bot 抛异常。
- `END:<status>:WINNER:<p>:RESULT:<result>:PLIES:<plies>:FEN:<fen>`：一局结束。

## 命令行接口

生成示例 bot：

```bash
python env/chess_env.py sample-bot --output bot.py
```

单局对战：

```bash
python env/chess_env.py battle --bot baseline/gpt5p5/bot_hard.py --games 1 --seat 0 --seed 1 --keep-logs
```

批量对战：

```bash
python env/chess_env.py battle --bot baseline/gpt5p5/bot_easy.py --games 100 --seed 1
```

默认批量对战会轮换开发者 bot 的座位，减少先手影响。如需固定开发者 bot 在 `seat=0`：

```bash
python env/chess_env.py battle --bot baseline/gpt5p5/bot_easy.py --games 100 --fixed-seat
```
