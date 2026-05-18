# nimmt 简化版 Bot 对战接口

本文档描述 `nimmt_multi.py` 提供的标准化接口。它和仓库里其他小游戏一样，Bot 只需要提供 `choose_action(state)`，并返回 `state["legal_actions"]` 里的一个字符串动作。

## 规则范围

- 支持玩家数：`2..6`。
- N 名玩家时，牌堆是 `1..10N`，每名玩家发 `10` 张，所有牌全部发完。
- 只玩这一轮，共 `10` 墩同步出牌。
- 没有额外的初始牌。结算时前 `4` 个开行的牌会建立 4 行。
- 每墩所有玩家先同时选择一张牌，再按牌面从小到大结算。
- 放牌规则遵循 6 nimmt：放到行顶小于该牌且最接近该牌的行；如果成为第 6 张，则吃掉原行并以该牌开新行。
- 如果牌比当前所有行顶都小，Bot 必须在出牌动作中同时指定要吃哪一行。
- 游戏结束后直接比较牛头数，牛头数最低者胜；平分时 `winners` 会包含所有并列玩家。

## 牛头数

牛头数按原版常见规则计算：

- `55` 是 `7` 牛。
- 其他 `11` 的倍数是 `5` 牛。
- 其他 `10` 的倍数是 `3` 牛。
- 其他 `5` 的倍数是 `2` 牛。
- 其余牌是 `1` 牛。

## bot.py 标准格式

最小随机 Bot：

```python
import random


def choose_action(state):
    return random.choice(state["legal_actions"])
```

也可以写成类：

```python
import random


class Bot:
    name = "my_nimmt_bot"

    def choose_action(self, state):
        return random.choice(state["legal_actions"])
```

约束：

- `choose_action(state)` 必须返回字符串动作。
- 返回值必须在 `state["legal_actions"]` 里。
- Bot 只能看到自己的手牌和公共行、比分、历史，不会看到其他玩家手牌。
- 如果 Bot 抛异常或返回非法动作，本局会以 `bot_exception` 或 `invalid_action` 结束，并判该 Bot 负。

## state 字段

示例：

```python
{
    "player_id": 0,
    "num_players": 4,
    "phase": "play",
    "actor": 0,
    "legal_actions": ["PLAY_3_TAKE_0", "PLAY_3_TAKE_1", "PLAY_18"],
    "hand": [3, 18, 29],
    "hand_sizes": [3, 3, 3, 3],
    "rows": [[8, 12], [15], [21, 24, 28], [31]],
    "row_bulls": [2, 2, 4, 1],
    "scores": [0, 4, 2, 1],
    "turn": 7,
    "cards_per_player": 10,
    "deck_max": 40,
    "bull_values": {3: 1, 18: 1, 29: 1},
    "history": []
}
```

字段说明：

- `player_id`：当前 Bot 的座位号。
- `num_players`：玩家数，范围 `2..6`。
- `phase`：固定为 `"play"`。
- `actor`：和 `player_id` 相同；本游戏每墩同步出牌。
- `legal_actions`：当前 Bot 可返回的所有合法动作。
- `hand`：自己的剩余手牌。
- `hand_sizes`：所有玩家剩余手牌数量。
- `rows`：当前公共 4 行，开局未补足 4 行时可能少于 4 行。
- `row_bulls`：每行当前牛头数。
- `scores`：所有玩家已吃到的牛头数。
- `turn`：已完成墩数，范围 `0..10`。
- `deck_max`：本局最大牌号，即 `10N`。
- `bull_values`：自己手牌中每张牌的牛头数。
- `history`：已结算墩的公开历史。

## 动作说明

- `PLAY_17`：打出 `17`。
- `PLAY_3_TAKE_2`：打出 `3`，如果需要吃牛，则吃第 `2` 行。

当 4 行已经建立，且某张手牌小于所有当前行顶牌时，`legal_actions` 不会给出 `PLAY_<card>`，而会给出：

```text
PLAY_<card>_TAKE_0
PLAY_<card>_TAKE_1
PLAY_<card>_TAKE_2
PLAY_<card>_TAKE_3
```

也就是说，Bot 必须在决定出哪张牌的同时决定“如果要吃牛，要吃哪堆”。

## Python API

### 单次对战

```python
from nimmt_multi import battle_once

result = battle_once(
    "bot_random.py",
    players=4,
    seat=0,
    seed=1,
    keep_log=True,
)
print(result)
```

返回示例：

```python
{
    "game_id": "a1b2c3d4e5f6",
    "winner": 2,
    "winners": [2],
    "status": "ok",
    "turns": 10,
    "scores": [8, 12, 5, 9],
    "rows": [[1, 6, 10], [18], [24, 29], [37, 40]],
    "bot_names": ["random_nimmt", "system_1", "system_2", "system_3"],
    "developer_seat": 0,
    "developer_win": False,
    "error": None
}
```

### 批量对战

```python
from nimmt_multi import battle_many

summary = battle_many(
    "bot_random.py",
    games=1000,
    players=6,
    seed=1,
    alternate_seats=True,
    keep_logs=False,
)
print(summary)
```

返回字段：

- `games`：总局数。
- `players`：每局玩家数。
- `wins_by_seat`：各座位成为并列最低分者的次数。
- `ties`：出现并列最低分的局数。
- `developer_wins`：开发者 Bot 成为最低分者的次数。
- `developer_losses`：开发者 Bot 未成为最低分者的次数。
- `developer_win_rate`：开发者 Bot 胜率。
- `statuses`：`ok`、`invalid_action`、`bot_exception` 等状态统计。
- `game_ids`：仅当 `keep_logs=True` 时返回可查询日志的 id。

### 查询对战日志

```python
from nimmt_multi import battle_once, get_match_log

result = battle_once("bot_random.py", players=4, seed=1, keep_log=True)
for item in get_match_log(result["game_id"]):
    print(item)
```

日志项示例：

- `G:<game_id>:N<players>:SEED<seed>`：一局开始。
- `H:P<p>:<cards>`：玩家初始手牌。
- `T<t>:P<p>:PLAY:<card>`：玩家选择出牌。
- `T<t>:P<p>:OPEN:R<r>:<card>`：开局补建一行。
- `T<t>:P<p>:PLACE:R<r>:<card>`：放入某行。
- `T<t>:P<p>:SIXTH:R<r>:B<bulls>:<old_row>><card>`：成为第 6 张，吃掉该行。
- `T<t>:P<p>:TAKE:R<r>:B<bulls>:<old_row>><card>`：牌太小，吃掉指定行。
- `END:<status>:WINNERS:<players>:SCORES:<scores>:ROWS:<rows>`：一局结束。

## 命令行接口

生成示例随机 Bot：

```powershell
python .\nimmt_multi.py sample-bot --output .\bot_random.py
```

单局对战：

```powershell
python .\nimmt_multi.py battle --bot .\bot_random.py --players 4 --games 1 --seat 0 --seed 1 --keep-logs
```

批量对战：

```powershell
python .\nimmt_multi.py battle --bot .\bot_random.py --players 6 --games 1000 --seed 1
```

默认批量对战会轮换开发者 Bot 的座位。若要固定在 `seat=0`：

```powershell
python .\nimmt_multi.py battle --bot .\bot_random.py --players 6 --games 100 --fixed-seat
```
