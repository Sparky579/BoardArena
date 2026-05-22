# UNO Bot 对战接口

本文档描述 `env/uno_env.py` 提供的标准化接口。Bot 需要根据当前观察返回一个合法字符串动作。

## bot.py 标准格式

最小可用版本：

```python
def choose_action(state):
    return state["legal_actions"][0]
```

也可以写成类：

```python
class Bot:
    name = "my_uno_bot"

    def choose_action(self, state):
        return state["legal_actions"][0]
```

约束：

- `choose_action(state)` 必须返回一个字符串动作。
- 返回值必须在 `state["legal_actions"]` 中。
- Bot 只能看到自己的 `hand`、双方手牌数量、弃牌堆顶牌、当前颜色和公开历史。
- Bot 抛异常或返回非法动作时，本局会以 `bot_exception` 或 `invalid_action` 结束，并判该 Bot 负。
- 如果设置了单步决策限时，Bot 超时会以 `timeout` 结束并判负。

## 动作格式

- `play:<card_id>`：打出普通颜色牌。
- `play:<card_id>:<color>`：打出 `Wild` 或 `Wild Draw Four` 并声明颜色。
- `draw`：没有可出牌时摸 1 张。
- `pass`：无法出牌且无法摸牌时跳过。

颜色字符串固定为：

```python
"red", "yellow", "green", "blue"
```

## state 字段

`choose_action(state)` 收到的是一个 `dict`：

```python
{
    "player_id": 0,
    "num_players": 2,
    "phase": "turn",
    "actor": 0,
    "turn": "player_0",
    "legal_actions": ["play:12", "play:101:red", "play:101:blue"],
    "hand": [
        {
            "id": 12,
            "color": "red",
            "kind": "number",
            "value": 7,
            "label": "R7",
            "current_color": "red",
            "legal_actions": ["play:12"]
        }
    ],
    "hand_count": 7,
    "hand_counts": [7, 7],
    "opponent_hand_count": 7,
    "top_card": {
        "id": 3,
        "color": "blue",
        "kind": "number",
        "value": 7,
        "label": "B7",
        "current_color": "blue"
    },
    "current_color": "blue",
    "draw_pile_count": 93,
    "discard_pile_count": 1,
    "can_draw": False,
    "can_pass": False,
    "plies": 0,
    "last_action": None,
    "last_draw_count": 0,
    "history": [],
    "winner": None,
    "status": None,
    "result": "*"
}
```

字段说明：

- `player_id`：收到该观察的玩家座位。
- `actor`：当前行动玩家。
- `legal_actions`：当前玩家可选动作。非行动玩家观察中为空。
- `hand`：当前观察玩家自己的手牌。
- `hand_counts`：两个座位的手牌数。
- `top_card`：弃牌堆顶牌。若顶牌是万能牌，`current_color` 表示声明颜色。
- `current_color`：当前需要匹配的颜色。
- `draw_pile_count` / `discard_pile_count`：摸牌堆和弃牌堆数量。
- `history`：公开动作历史，不包含对手具体手牌。
- `winner`：玩家 `0` 获胜为 `0`，玩家 `1` 获胜为 `1`，未结束或平局为 `None`。
- `status`：终局状态，未结束为 `None`。
- `result`：`1-0`、`0-1`、`1/2-1/2` 或 `*`。

## Python API

内置 baseline 包含 `baseline/gpt/bot_easy.py` 和 `baseline/gpt/bot_hard.py`。`bot_hard.py` 是不依赖第三方库的 MCTS bot，会在每次行动时基于公开信息抽样隐藏牌并进行快速 rollout；建议批量评测时设置 `--decision-timeout 2`。

### Gym 风格环境

```python
from env.uno_env import UnoEnv

env = UnoEnv(seed=1)
state, info = env.reset()
state, reward, terminated, truncated, info = env.step(state["legal_actions"][0])
```

### 单次对战

```python
from env.uno_env import battle_once

result = battle_once(
    "baseline/gpt/bot_hard.py",
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
    "status": "empty_hand",
    "result": "1-0",
    "plies": 41,
    "hand_counts": [0, 8],
    "current_color": "red",
    "bot_names": ["gpt_hard", "system"],
    "developer_seat": 0,
    "developer_win": True,
    "error": None
}
```

### 批量对战

```python
from env.uno_env import battle_many

summary = battle_many(
    "baseline/gpt/bot_easy.py",
    games=100,
    seed=1,
    alternate_seats=True,
    decision_timeout=2.0,
)
print(summary)
```

## 命令行接口

生成示例 bot：

```bash
python env/uno_env.py sample-bot --output bot.py
```

单局对战：

```bash
python env/uno_env.py battle --bot baseline/gpt/bot_hard.py --games 1 --seat 0 --seed 1 --keep-logs
```

批量对战：

```bash
python env/uno_env.py battle --bot baseline/gpt/bot_easy.py --games 100 --seed 1
```

固定开发者 bot 座位：

```bash
python env/uno_env.py battle --bot baseline/gpt/bot_easy.py --games 100 --fixed-seat
```

限制单步决策时间：

```bash
python env/uno_env.py battle --bot baseline/gpt/bot_hard.py --games 100 --decision-timeout 2
```
