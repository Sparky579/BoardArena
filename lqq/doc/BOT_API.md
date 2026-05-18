# 路墙棋 Bot 对战接口

本文档描述 `lqq_multi.py` 提供的标准化接口。接口形状对齐 Skull 多人 Bot 对战接口，但路墙棋只支持 `2` 人对战。

## 规则范围

- 支持玩家数：固定 `2`。
- 棋盘为 `9 x 9`。
- 每方初始 `10` 堵墙。
- 每回合只能移动一格，或放置一堵两格长的墙。
- 墙不能重叠、交叉，也不能完全堵死任一方通往终点的路线。
- 玩家 `0` 从下方向上走，玩家 `1` 从上方向下走。
- 先到达对面底线者获胜。

## bot.py 标准格式

最小可用版本：

```python
def choose_action(state):
    legal = state["legal_actions"]
    player_id = state["player_id"]
    row = state["positions"][player_id][0]
    goal = state["goal_rows"][player_id]

    forward = "MOVE_UP" if goal < row else "MOVE_DOWN"
    if forward in legal:
        return forward

    moves = [a for a in legal if a.startswith("MOVE_")]
    if moves:
        return moves[0]

    return legal[0]
```

也可以写成类：

```python
class Bot:
    name = "my_lqq_bot"

    def choose_action(self, state):
        return state["legal_actions"][0]
```

约束：

- `choose_action(state)` 必须返回一个字符串动作。
- 返回值必须在 `state["legal_actions"]` 里。
- 路墙棋没有暗牌，bot 可以看到双方位置、剩余墙数和已放置的墙。
- 如果 bot 抛异常或返回非法动作，本局会以 `bot_exception` 或 `invalid_action` 结束，并判该 bot 负。
- 如果极端局面下当前玩家没有任何合法动作，本局会以 `no_legal_actions` 结束，并判对手胜。

## state 字段

`choose_action(state)` 收到的是一个 `dict`：

```python
{
    "player_id": 0,
    "num_players": 2,
    "phase": "turn",
    "actor": 0,
    "legal_actions": ["MOVE_UP", "MOVE_LEFT", "MOVE_RIGHT", "WALL_H_0_0"],
    "board_size": 9,
    "positions": [[8, 4], [0, 4]],
    "goal_rows": [0, 8],
    "walls_remaining": [10, 10],
    "walls": [{"dir": "H", "row": 3, "col": 4}],
    "turn": 0
}
```

字段说明：

- `player_id`：当前收到状态的 bot 座位号。
- `num_players`：固定为 `2`。
- `phase`：固定为 `"turn"`。
- `actor`：当前行动玩家。
- `legal_actions`：当前行动玩家的所有合法动作。
- `board_size`：棋盘大小，固定为 `9`。
- `positions`：双方棋子坐标，格式为 `[row, col]`。
- `goal_rows`：双方目标行，玩家 `0` 为 `0`，玩家 `1` 为 `8`。
- `walls_remaining`：双方剩余墙数。
- `walls`：已放置墙列表。
- `turn`：已执行动作数。

## 动作说明

- `MOVE_UP`：向上移动一格。
- `MOVE_DOWN`：向下移动一格。
- `MOVE_LEFT`：向左移动一格。
- `MOVE_RIGHT`：向右移动一格。
- `WALL_H_r_c`：在原点 `(r, c)` 放置横墙。
- `WALL_V_r_c`：在原点 `(r, c)` 放置竖墙。

墙坐标 `r` 和 `c` 的合法范围是 `0..7`。例如 `WALL_H_3_4` 表示在第 `3` 行、第 `4` 列原点放置横墙。

## Python API

### 单次对战

```python
from lqq_multi import battle_once

result = battle_once(
    "bot.py",
    players=2,
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
    "winner": 0,
    "status": "ok",
    "turns": 17,
    "positions": [[0, 4], [8, 3]],
    "walls_remaining": [10, 10],
    "bot_names": ["bot", "system"],
    "developer_seat": 0,
    "developer_win": True,
    "error": None
}
```

### 批量对战

```python
from lqq_multi import battle_many

summary = battle_many(
    "bot.py",
    games=1000,
    players=2,
    seed=1,
    alternate_seats=True,
    keep_logs=False,
)
print(summary)
```

返回字段：

- `games`：总局数。
- `players`：固定为 `2`。
- `wins_by_seat`：两个座位的胜场。
- `developer_wins`：开发者 bot 胜场。
- `developer_losses`：开发者 bot 负场。
- `developer_win_rate`：开发者 bot 胜率。
- `statuses`：`ok`、`turn_limit`、`invalid_action`、`bot_exception`、`no_legal_actions` 等状态统计。
- `game_ids`：仅当 `keep_logs=True` 时返回可查询日志的 id。

### 查询对战日志

```python
from lqq_multi import battle_once, get_match_log

result = battle_once("bot.py", keep_log=True)
log = get_match_log(result["game_id"])
for item in log:
    print(item)
```

日志是简单字符串编码，每一项表示一个局内事件：

- `G:<game_id>:N2:SEED<seed>`：一局开始。
- `T<t>:P<p>:MOVE:<r1>,<c1>><r2>,<c2>`：玩家移动。
- `T<t>:P<p>:WALL:H:<r>:<c>`：玩家放置横墙。
- `T<t>:P<p>:WALL:V:<r>:<c>`：玩家放置竖墙。
- `T<t>:ERR:P<p>:INVALID:<action>`：玩家返回非法动作。
- `T<t>:ERR:P<p>:EXCEPTION:<type>`：玩家 bot 抛异常。
- `T<t>:ERR:P<p>:NO_LEGAL_ACTIONS`：玩家没有任何合法动作。
- `END:<status>:WINNER:<p>:POS:<positions>:WALLS:<walls_remaining>`：一局结束。

## 命令行接口

生成示例 bot：

```powershell
python .\lqq_multi.py sample-bot
```

单局对战：

```powershell
python .\lqq_multi.py battle --bot .\bot.py --players 2 --games 1 --seat 0 --seed 1 --keep-logs
```

批量对战：

```powershell
python .\lqq_multi.py battle --bot .\bot.py --players 2 --games 1000 --seed 1
```

默认批量对战会轮换开发者 bot 的座位，减少座位优势影响。如需固定开发者 bot 在 `seat=0`：

```powershell
python .\lqq_multi.py battle --bot .\bot.py --players 2 --games 100 --fixed-seat
```
