# Skull 多人 Bot 对战接口

本文档描述 `skull_multi.py` 提供的标准化接口。它是独立于原有 `skull_cfr.py` 的多人裁判模块，支持 2 到 6 人对战、开发者 `bot.py`、单局/批量对战、以及局内日志查询。

## 规则范围

- 支持玩家数：`2..6`。
- 每名玩家初始手牌：`3` 张花 `F`，`1` 张骷髅 `S`。
- 每轮先暗放牌；仍有手牌时可以放 `PLAY_F` 或 `PLAY_S`。
- 所有未淘汰玩家都至少放过一张牌后，允许叫号 `BID_n`。
- 叫号阶段可 `PASS` 或加叫更大的 `BID_n`。
- 多人局中，除最高叫号者外的其他未弃权玩家都 `PASS` 后，最高叫号者开始挑战。
- 挑战者会先自动翻完自己牌堆中需要翻的牌；如果还没达到叫号数，由 bot 选择 `FLIP_i` 翻某个对手的牌堆顶牌。
- 翻到 `S` 挑战失败，挑战者随机损失自己的一张牌；全部翻到 `F` 挑战成功并得 1 分。
- 先到 2 分或其他玩家全部无牌时获胜。

## bot.py 标准格式

最小可用版本：

```python
def choose_action(state):
    legal = state["legal_actions"]
    if state["phase"] == "challenge":
        return legal[0]
    if "PLAY_F" in legal:
        return "PLAY_F"
    bids = [a for a in legal if a.startswith("BID_")]
    if bids:
        return bids[0]
    return "PASS" if "PASS" in legal else legal[0]
```

也可以写成类：

```python
class Bot:
    name = "my_bot"

    def choose_action(self, state):
        return state["legal_actions"][0]
```

约束：

- `choose_action(state)` 必须返回一个字符串动作。
- 返回值必须在 `state["legal_actions"]` 里。
- bot 只能看到自己的手牌和公共信息，看不到其他玩家暗牌内容。
- 如果 bot 抛异常或返回非法动作，本局会以 `bot_exception` 或 `invalid_action` 结束，并判该 bot 负。
- 如果设置了单步决策限时，且 bot 的一次 `choose_action(state)` 调用超时，本局会以 `timeout` 结束，并判该 bot 负。

## state 字段

`choose_action(state)` 收到的是一个 `dict`：

```python
{
    "player_id": 0,
    "num_players": 2,
    "phase": "play",
    "actor": 0,
    "legal_actions": ["PLAY_F", "PLAY_S"],
    "scores": [0, 0],
    "hand": {"flowers": 3, "skulls": 1},
    "own_pile": [],
    "pile_sizes": [0, 0],
    "total_cards": [4, 4],
    "current_bid": 0,
    "high_bidder": -1,
    "passed": [False, False],
    "challenge_remaining": 0,
    "flipped_counts": [0, 0],
    "turn": 0
}
```

动作说明：

- `PLAY_F`：暗放一张花。
- `PLAY_S`：暗放一张骷髅。
- `BID_1`、`BID_2` ...：叫号或加叫。
- `PASS`：叫号阶段放弃继续加叫。
- `FLIP_0` ... `FLIP_5`：挑战阶段翻指定玩家的一张牌，不能翻自己，且只能翻仍有未翻牌的牌堆。

## Python API

### 单次对战

```python
from skull_multi import battle_once

result = battle_once(
    "bot.py",
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
    "status": "ok",
    "turns": 37,
    "scores": [2, 0],
    "bot_names": ["bot", "system_1"],
    "developer_seat": 0,
    "developer_win": True,
    "error": None
}
```

### 批量对战

```python
from skull_multi import battle_many

summary = battle_many(
    "bot.py",
    games=1000,
    players=2,
    seed=1,
    alternate_seats=True,
    keep_logs=False,
    decision_timeout=2.0,
)
print(summary)
```

返回字段：

- `games`：总局数。
- `players`：每局玩家数。
- `wins_by_seat`：各座位胜场。
- `developer_wins`：开发者 bot 胜场。
- `developer_losses`：开发者 bot 负场。
- `developer_win_rate`：开发者 bot 胜率。
- `statuses`：`ok`、`turn_limit`、`invalid_action`、`bot_exception`、`timeout` 等状态统计。
- `game_ids`：仅当 `keep_logs=True` 时返回可查询日志的 id。

### 查询对战日志

```python
from skull_multi import battle_once, get_match_log

result = battle_once("bot.py", keep_log=True)
log = get_match_log(result["game_id"])
for item in log:
    print(item)
```

日志是简单字符串编码，每一项表示一个局内事件：

- `G:<game_id>:N<players>:SEED<seed>`：一局开始。
- `T<t>:P<p>:PLAY_F` / `PLAY_S`：玩家暗放一张牌。
- `T<t>:P<p>:B<n>`：玩家叫号到 `n`。
- `T<t>:P<p>:PASS`：玩家在叫号阶段 pass。
- `T<t>:R:P<p>:F` / `S`：挑战阶段翻开玩家 `p` 的一张花/骷髅。
- `T<t>:OK:P<p>`：玩家 `p` 挑战成功得分。
- `T<t>:BAD:P<p>`：玩家 `p` 挑战失败。
- `T<t>:X:P<p>:F` / `S`：玩家 `p` 随机损失一张花/骷髅。
- `T<t>:ERR:P<p>:INVALID:<action>`：玩家返回非法动作。
- `T<t>:ERR:P<p>:EXCEPTION:<type>`：玩家 bot 抛异常。
- `T<t>:ERR:P<p>:TIMEOUT`：玩家 bot 单步决策超时。
- `END:<status>:WINNER:<p>:SCORES:<scores>`：一局结束。

## 命令行接口

生成示例 bot：

```powershell
python .\skull_multi.py sample-bot
```

单局对战：

```powershell
python .\skull_multi.py battle --bot .\bot.py --players 2 --games 1 --seat 0 --seed 1 --keep-logs
```

批量对战：

```powershell
python .\skull_multi.py battle --bot .\bot.py --players 2 --games 1000 --seed 1
```

限制每次 `choose_action(state)` 最多思考 2 秒，超时自动判负：

```powershell
python .\skull_multi.py battle --bot .\bot.py --players 2 --games 100 --seed 1 --decision-timeout 2
```

多人局：

```powershell
python .\skull_multi.py battle --bot .\bot.py --players 6 --games 100 --seed 1
```

默认批量对战会轮换开发者 bot 的座位，减少座位优势影响。如需固定开发者 bot 在 `seat=0`：

```powershell
python .\skull_multi.py battle --bot .\bot.py --players 4 --games 100 --fixed-seat
```
