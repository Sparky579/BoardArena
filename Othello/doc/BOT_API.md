# 黑白棋 Bot 对战接口

本文档描述 `env/othello_env.py` 提供的标准化接口。Bot 需要根据当前局面返回一个合法动作。

## bot.py 标准格式

最小可用版本：

```python
def choose_action(state):
    return state["legal_actions"][0]
```

也可以写成类：

```python
class Bot:
    name = "my_othello_bot"

    def choose_action(self, state):
        return state["legal_actions"][0]
```

约束：

- `choose_action(state)` 必须返回一个字符串动作。
- 返回值必须在 `state["legal_actions"]` 里。
- 普通落子使用坐标字符串，例如 `d3`。
- 强制跳过使用 `PASS`，但只有 `PASS` 出现在 `legal_actions` 时才合法。
- 黑白棋是完全信息游戏，bot 可以看到完整棋盘、合法动作、每个合法动作会翻哪些子，以及历史。
- 如果 bot 抛异常或返回非法动作，本局会以 `bot_exception` 或 `invalid_action` 结束，并判该 bot 负。
- 如果设置了单步决策限时，且 bot 的一次 `choose_action(state)` 调用超时，本局会以 `timeout` 结束，并判该 bot 负。

## state 字段

`choose_action(state)` 收到的是一个 `dict`：

```python
{
    "player_id": 0,
    "num_players": 2,
    "phase": "turn",
    "actor": 0,
    "turn": "black",
    "legal_actions": ["c4", "d3", "e6", "f5"],
    "legal_flips": {"c4": ["d4"], "d3": ["d4"], "e6": ["e5"], "f5": ["e5"]},
    "board": [
        "........",
        "........",
        "........",
        "...BW...",
        "...WB...",
        "........",
        "........",
        "........"
    ],
    "board_size": 8,
    "pieces": [
        {"square": "d4", "color": "white", "symbol": "W"}
    ],
    "disc_counts": {"black": 2, "white": 2},
    "scores": [2, 2],
    "empty_count": 60,
    "plies": 0,
    "pass_count": 0,
    "last_move": None,
    "last_flips": [],
    "history": [],
    "winner": None,
    "status": None,
    "result": "*"
}
```

字段说明：

- `player_id`：当前收到状态的 bot 座位号，黑方为 `0`，白方为 `1`。
- `num_players`：固定为 `2`。
- `phase`：`turn` 或 `game_over`。
- `actor`：当前行动玩家，黑方 `0`，白方 `1`。
- `turn`：`black` 或 `white`。
- `legal_actions`：当前行动玩家的所有合法动作。
- `legal_flips`：普通落子的翻子结果；`PASS` 不会出现在该字段中。
- `board`：8 行字符串，从 rank 8 到 rank 1；`.` 为空，`B` 为黑子，`W` 为白子。
- `pieces`：棋子列表，便于 UI 或不想解析 `board` 的 bot 使用。
- `disc_counts`：黑白棋子数。
- `scores`：按 `[black, white]` 排列的棋子数。
- `empty_count`：空格数量。
- `plies`：已经执行的动作数，包含 `PASS`。
- `pass_count`：连续 `PASS` 数。
- `last_move`：上一手动作。
- `last_flips`：上一手翻转的棋子坐标。
- `history`：动作历史。
- `winner`：黑胜为 `0`，白胜为 `1`，平局或未结束为 `None`。
- `status`：终局状态，未结束为 `None`。
- `result`：`1-0`、`0-1`、`1/2-1/2` 或 `*`。

## Python API

内置 baseline 包含 `baseline/gpt5p5/bot_easy.py` 和 `baseline/gpt5p5/bot_hard.py`。`bot_hard.py` 是纯 CPU 传统搜索 bot，内部按 2 秒单步限制预留余量。

### Gym 风格环境

```python
from env.othello_env import OthelloEnv

env = OthelloEnv(seed=1)
state, info = env.reset()
state, reward, terminated, truncated, info = env.step("d3")
```

### 单次对战

```python
from env.othello_env import battle_once

result = battle_once(
    "baseline/gpt5p5/bot_easy.py",
    players=2,
    seat=0,
    seed=1,
    keep_log=True,
    decision_timeout=2.0,
)
print(result)
```

返回示例：

```python
{
    "game_id": "a1b2c3d4e5f6",
    "winner": 0,
    "status": "disc_majority",
    "result": "1-0",
    "plies": 60,
    "scores": [38, 26],
    "board": ["BBBBBBBB", "..."],
    "bot_names": ["gpt5p5_othello_easy", "system"],
    "developer_seat": 0,
    "developer_win": True,
    "error": None
}
```

### 批量对战

```python
from env.othello_env import battle_many

summary = battle_many(
    "baseline/gpt5p5/bot_easy.py",
    games=100,
    seed=1,
    alternate_seats=True,
    decision_timeout=2.0,
)
print(summary)
```

返回字段：

- `games`：总局数。
- `players`：固定为 `2`。
- `wins_by_seat`：黑方、白方胜场。
- `draws`：平局数。
- `developer_wins`：开发者 bot 胜场。
- `developer_losses`：开发者 bot 未胜场，包含平局。
- `developer_win_rate`：开发者 bot 胜率。
- `statuses`：终局状态统计，包含 `timeout`、`invalid_action`、`bot_exception` 等异常结束状态。
- `game_ids`：仅当 `keep_logs=True` 时返回可查询日志的 id。

### 查询对战日志

```python
from env.othello_env import battle_once, get_match_log

result = battle_once("baseline/gpt5p5/bot_easy.py", keep_log=True)
log = get_match_log(result["game_id"])
for item in log:
    print(item)
```

日志是简单字符串编码：

- `G:<game_id>:N2:SEED<seed>:GAME:OTHELLO`：一局开始。
- `T<t>:P<p>:MOVE:<square>:FLIPS:<squares>:BOARD:<rows>`：玩家落子。
- `T<t>:P<p>:PASS`：玩家强制跳过。
- `T<t>:ERR:P<p>:INVALID:<action>`：玩家返回非法动作。
- `T<t>:ERR:P<p>:EXCEPTION:<type>`：玩家 bot 抛异常。
- `T<t>:ERR:P<p>:TIMEOUT`：玩家 bot 单步决策超时。
- `END:<status>:WINNER:<p>:RESULT:<result>:PLIES:<plies>:SCORE:<b>-<w>:BOARD:<rows>`：一局结束。

## 命令行接口

生成示例 bot：

```bash
python env/othello_env.py sample-bot --output bot.py
```

单局对战：

```bash
python env/othello_env.py battle --bot baseline/gpt5p5/bot_easy.py --games 1 --seat 0 --seed 1 --keep-logs
```

批量对战：

```bash
python env/othello_env.py battle --bot baseline/gpt5p5/bot_easy.py --games 100 --seed 1
```

限制每次 `choose_action(state)` 最多思考 2 秒，超时自动判负：

```bash
python env/othello_env.py battle --bot baseline/gpt5p5/bot_easy.py --games 100 --seed 1 --decision-timeout 2
```

默认批量对战会轮换开发者 bot 的座位，减少先手影响。如需固定开发者 bot 在 `seat=0`：

```bash
python env/othello_env.py battle --bot baseline/gpt5p5/bot_easy.py --games 100 --fixed-seat
```
