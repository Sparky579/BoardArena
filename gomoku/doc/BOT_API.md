# 五子棋 Bot 对战接口

本文档描述 `env/gomoku_env.py` 提供的标准化接口。Bot 需要根据当前局面返回一个合法动作。

## bot.py 标准格式

最小可用版本：

```python
def choose_action(state):
    return state["legal_actions"][0]
```

也可以写成类：

```python
class Bot:
    name = "my_gomoku_bot"

    def choose_action(self, state):
        return state["legal_actions"][0]
```

约束：

- `choose_action(state)` 必须返回一个字符串动作。
- 返回值必须在 `state["legal_actions"]` 里。
- 落子使用坐标字符串，例如 `h8`、`a1`、`o15`。
- 没有 `PASS`，必须落子。
- 五子棋是完全信息游戏，bot 可以看到完整棋盘、合法动作与历史。`legal_actions` 已经过滤掉黑方禁手。
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
    "legal_actions": ["a1", "a2", "...", "o15"],
    "board": [
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "..............."
    ],
    "board_size": 15,
    "pieces": [],
    "stone_counts": {"black": 0, "white": 0},
    "scores": [0, 0],
    "empty_count": 225,
    "plies": 0,
    "last_move": None,
    "last_player": None,
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
- `legal_actions`：当前行动玩家的所有合法动作；**已经过滤掉黑方禁手**。
- `board`：15 行字符串，从 rank 15 到 rank 1；`.` 为空，`B` 为黑子，`W` 为白子。
- `pieces`：棋子列表，便于 UI 或不想解析 `board` 的 bot 使用。
- `stone_counts` / `scores`：黑白棋子数。
- `empty_count`：空点数量。
- `plies`：已经执行的动作数。
- `last_move`：上一手动作。
- `last_player`：上一手玩家的座位号。
- `history`：动作历史。
- `winner`：黑胜为 `0`，白胜为 `1`，平局或未结束为 `None`。
- `status`：终局状态，未结束为 `None`。
- `result`：`1-0`、`0-1`、`1/2-1/2` 或 `*`。

## Python API

内置 baseline 含 `baseline/claude_opus4p7/bot_easy.py` 和 `baseline/claude_opus4p7/bot_hard.py`。Hard bot 是纯 CPU 传统搜索 bot，按 2 秒单步限制预留余量。

### Gym 风格环境

```python
from env.gomoku_env import GomokuEnv

env = GomokuEnv(seed=1)
state, info = env.reset()
state, reward, terminated, truncated, info = env.step("h8")
```

### 单次对战

```python
from env.gomoku_env import battle_once

result = battle_once(
    "baseline/claude_opus4p7/bot_hard.py",
    players=2,
    seat=0,
    seed=1,
    keep_log=True,
    decision_timeout=2.0,
)
print(result)
```

### 批量对战

```python
from env.gomoku_env import battle_many

summary = battle_many(
    "baseline/claude_opus4p7/bot_hard.py",
    games=20,
    seed=1,
    alternate_seats=True,
    decision_timeout=2.0,
)
```

### 两个任意 Bot 路径互相对战

```python
from env.gomoku_env import battle_bots_many

summary = battle_bots_many(
    "baseline/claude_opus4p7/bot_easy.py",
    "baseline/claude_opus4p7/bot_hard.py",
    games=20, seed=1, alternate_seats=True, decision_timeout=2.0,
)
```

### 查询对战日志

```python
from env.gomoku_env import battle_once, get_match_log

result = battle_once("baseline/claude_opus4p7/bot_hard.py", keep_log=True)
log = get_match_log(result["game_id"])
for item in log:
    print(item)
```

日志格式：

- `G:<game_id>:N2:SEED<seed>:GAME:GOMOKU`：一局开始。
- `T<t>:P<p>:MOVE:<square>:BOARD:<rows>`：玩家落子。
- `T<t>:ERR:P<p>:INVALID:<action>`：玩家返回非法动作。
- `T<t>:ERR:P<p>:EXCEPTION:<type>`：玩家 bot 抛异常。
- `T<t>:ERR:P<p>:TIMEOUT`：玩家 bot 单步决策超时。
- `END:<status>:WINNER:<p>:RESULT:<result>:PLIES:<plies>:SCORE:<b>-<w>:BOARD:<rows>`：一局结束。

## 命令行接口

生成示例 bot：

```bash
python env/gomoku_env.py sample-bot --output bot.py
```

单局对战：

```bash
python env/gomoku_env.py battle --bot baseline/claude_opus4p7/bot_hard.py --games 1 --seed 1 --keep-logs
```

批量对战：

```bash
python env/gomoku_env.py battle --bot baseline/claude_opus4p7/bot_hard.py --games 20 --seed 1 --decision-timeout 2
```

两个 Bot 互相对战：

```bash
python env/gomoku_env.py duel --bot0 baseline/claude_opus4p7/bot_easy.py --bot1 baseline/claude_opus4p7/bot_hard.py --games 20 --decision-timeout 2
```
