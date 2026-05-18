# nimmt 简化版

这是一个面向 Bot 对战的简化版 `6 nimmt! / 谁是牛头王` 裁判。

## 简化规则

- 支持 `2..6` 人。
- N 人局使用 `1..10N` 的牌，全部发完，每人 `10` 张。
- 只玩一轮，所有人正好出完 10 张牌。
- 每墩同步出牌，按牌面从小到大结算。
- 没有额外初始牌，开局结算时先补足 4 行。
- 之后按原版行规则放牌；第 6 张吃原行；牌比所有行顶都小时吃一行。
- “吃哪行”的选择被提前到出牌动作中：例如 `PLAY_3_TAKE_2`。
- 最后直接比较牛头数，牛头数最低者胜。

牛头数遵循常见原版规则：`55` 是 7 牛，其他 `11` 倍数 5 牛，其他 `10` 倍数 3 牛，其他 `5` 倍数 2 牛，其余 1 牛。

## 快速开始

单局随机 Bot 对战：

```powershell
python .\nimmt_multi.py battle --bot .\bot_random.py --players 4 --games 1 --seed 1 --keep-logs
```

批量对战：

```powershell
python .\nimmt_multi.py battle --bot .\bot_random.py --players 6 --games 1000 --seed 1
```

限制单步决策最多 2 秒：

```powershell
python .\nimmt_multi.py battle --bot .\bot_random.py --players 6 --games 100 --seed 1 --decision-timeout 2
```

生成一个新的随机 Bot 文件：

```powershell
python .\nimmt_multi.py sample-bot --output .\bot_random.py
```

更多接口说明见 `BOT_API.md`。
