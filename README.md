# BoardArena

BoardArena 是一个棋类/桌游 Bot 对战实验仓库。目前包含七个子项目：

- `lqq/`：路墙棋，支持本地网页对战和 2 人 Bot 裁判接口。
- `nimmt/`：简化版 6 nimmt!，支持 2 到 6 人 Bot 对战、批量评测和日志查询。
- `skull/`：Skull，支持多人 Bot 对战、批量评测和日志查询。
- `chess/`：国际象棋，支持 Gym 风格环境、本地网页人人/人机对战和 2 人 Bot 裁判接口。
- `Othello/`：黑白棋，支持 Gym 风格环境、本地网页人人/人机对战和 2 人 Bot 裁判接口。
- `go_9x9/`：九路围棋，支持 Gym 风格环境、本地网页人人/人机对战和 2 人 Bot 裁判接口。
- `uno/`：二人 UNO，支持 Gym 风格环境、本地网页人人/人机对战和 2 人 Bot 裁判接口。

## 项目结构

```text
BoardArena/
├── lqq/
│   ├── env/
│   ├── baseline/
│   ├── doc/
│   ├── lqq_multi.py
├── nimmt/
│   ├── env/
│   ├── baseline/
│   ├── doc/
│   └── nimmt_multi.py
├── skull/
│   ├── env/
│   ├── baseline/
│   ├── doc/
│   ├── skull_multi.py
│   └── skull_cfr.py
├── chess/
│   ├── env/
│   ├── baseline/
│   └── doc/
├── Othello/
│   ├── env/
│   ├── baseline/
│   └── doc/
├── go_9x9/
│   ├── env/
│   ├── baseline/
│   └── doc/
├── uno/
│   ├── env/
│   ├── baseline/
│   └── doc/
└── README.md
```

## 路墙棋

打开网页对战：

```powershell
start .\lqq\env\index.html
```

生成示例 Bot：

```powershell
python .\lqq\lqq_multi.py sample-bot --output .\lqq\bot.py
```

运行单局 Bot 对战：

```powershell
python .\lqq\lqq_multi.py battle --bot .\lqq\bot.py --players 2 --games 1 --seed 1 --keep-logs
```

运行批量对战：

```powershell
python .\lqq\lqq_multi.py battle --bot .\lqq\bot.py --players 2 --games 1000 --seed 1
```

更多规则和接口见：

- `lqq/doc/README.md`
- `lqq/doc/GAME_RULE.md`
- `lqq/doc/BOT_API.md`

## nimmt

生成示例 Bot：

```powershell
python .\nimmt\nimmt_multi.py sample-bot --output .\nimmt\bot_random.py
```

运行单局 Bot 对战：

```powershell
python .\nimmt\nimmt_multi.py battle --bot .\nimmt\bot_random.py --players 4 --games 1 --seed 1 --keep-logs
```

运行批量对战：

```powershell
python .\nimmt\nimmt_multi.py battle --bot .\nimmt\bot_random.py --players 6 --games 1000 --seed 1
```

更多说明见：

- `nimmt/doc/README.md`
- `nimmt/doc/GAME_RULE.md`
- `nimmt/doc/BOT_API.md`

## Skull

运行单局 Bot 对战：

```powershell
python .\skull\skull_multi.py battle --bot .\skull\bot.py --players 2 --games 1 --seed 1 --keep-logs
```

运行批量对战：

```powershell
python .\skull\skull_multi.py battle --bot .\skull\bot.py --players 6 --games 100 --seed 1
```

更多说明见：

- `skull/doc/README.md`
- `skull/doc/GAME_RULE.md`
- `skull/doc/BOT_API.md`

## Chess

安装依赖：

```powershell
python -m pip install -r .\chess\requirements.txt
```

打开浏览器对战：

```powershell
python .\chess\env\chess_web.py
```

不要直接用 Live Server 打开 `chess\env\index.html`，该页面依赖 `chess_web.py` 提供的本地 JSON API。
如果默认端口被占用，服务会自动尝试后续端口，也可以传入 `--port 8021`。
人机模式右侧可选择 `/gpt5p5/bot_easy` 或 `/gpt5p5/bot_hard`，并可调整动画速度。

运行单局 Bot 对战：

```powershell
python .\chess\env\chess_env.py battle --bot .\chess\baseline\gpt5p5\bot_hard.py --games 1 --seed 1 --keep-logs
```

运行批量对战：

```powershell
python .\chess\env\chess_env.py battle --bot .\chess\baseline\gpt5p5\bot_easy.py --games 100 --seed 1
```

更多说明见：

- `chess/doc/README.md`
- `chess/doc/GAME_RULE.md`
- `chess/doc/BOT_API.md`

## Othello

打开浏览器对战：

```powershell
python .\Othello\env\othello_web.py
```

不要直接用 Live Server 打开 `Othello\env\index.html`，该页面依赖 `othello_web.py` 提供的本地 JSON API。
如果默认端口被占用，服务会自动尝试后续端口，也可以传入 `--port 8031`。
人机模式右侧可选择 `/gpt5p5/bot_easy` 或 `/gpt5p5/bot_hard`。

运行单局 Bot 对战：

```powershell
python .\Othello\env\othello_env.py battle --bot .\Othello\baseline\gpt5p5\bot_easy.py --games 1 --seed 1 --keep-logs
```

运行批量对战：

```powershell
python .\Othello\env\othello_env.py battle --bot .\Othello\baseline\gpt5p5\bot_easy.py --games 100 --seed 1
```

运行 hard bot 时建议设置 2 秒单步限制：

```powershell
python .\Othello\env\othello_env.py battle --bot .\Othello\baseline\gpt5p5\bot_hard.py --games 10 --seed 1 --decision-timeout 2
```

更多说明见：

- `Othello/doc/README.md`
- `Othello/doc/GAME_RULE.md`
- `Othello/doc/BOT_API.md`

## go_9x9

打开浏览器对战：

```powershell
python .\go_9x9\env\go_web.py
```

不要直接用 Live Server 打开 `go_9x9\env\index.html`，该页面依赖 `go_web.py` 提供的本地 JSON API。
如果默认端口被占用，服务会自动尝试后续端口，也可以传入 `--port 8041`。
人机模式右侧可选择 `/gpt5p5/bot_easy` 或 `/gpt5p5/bot_hard`。

运行单局 Bot 对战：

```powershell
python .\go_9x9\env\go_env.py battle --bot .\go_9x9\baseline\gpt5p5\bot_easy.py --games 1 --seed 1 --keep-logs
```

运行批量对战：

```powershell
python .\go_9x9\env\go_env.py battle --bot .\go_9x9\baseline\gpt5p5\bot_easy.py --games 100 --seed 1
```

运行 hard bot 时建议设置 2 秒单步限制：

```powershell
python .\go_9x9\env\go_env.py battle --bot .\go_9x9\baseline\gpt5p5\bot_hard.py --games 10 --seed 1 --decision-timeout 2
```

更多说明见：

- `go_9x9/doc/README.md`
- `go_9x9/doc/GAME_RULE.md`
- `go_9x9/doc/BOT_API.md`

## UNO

打开浏览器对战：

```powershell
python .\uno\env\uno_web.py
```

不要直接用 Live Server 打开 `uno\env\index.html`，该页面依赖 `uno_web.py` 提供的本地 JSON API。
如果默认端口被占用，服务会自动尝试后续端口，也可以传入 `--port 8061`。
人机模式右侧可选择 `/gpt/bot_easy` 或 `/gpt/bot_hard`。

运行单局 Bot 对战：

```powershell
python .\uno\env\uno_env.py battle --bot .\uno\baseline\gpt\bot_hard.py --games 1 --seed 1 --keep-logs
```

运行批量对战：

```powershell
python .\uno\env\uno_env.py battle --bot .\uno\baseline\gpt\bot_easy.py --games 100 --seed 1
```

UNO 的 `bot_hard.py` 使用 determinized MCTS。运行 hard bot 时建议设置 2 秒单步限制：

```powershell
python .\uno\env\uno_env.py battle --bot .\uno\baseline\gpt\bot_hard.py --games 100 --seed 1 --decision-timeout 2
```

更多说明见：

- `uno/doc/README.md`
- `uno/doc/GAME_RULE.md`
- `uno/doc/BOT_API.md`

## ELO Benchmark

各游戏 Bot 的 ELO 评分汇总。Othello 采用 Edax 锚点法（绝对 ELO），其余游戏采用组内 round-robin 相对 ELO（初始值 1500）。

### 路墙棋（lqq）

Double round-robin，每对 2 局（先后手各一），`decision_timeout = 1.5s`。数据来源：[`lqq/arena/elo_results_v4.json`](lqq/arena/elo_results_v4.json)。

| 排名 | Bot | ELO | W | L | D |
|:---:|-----|----:|--:|--:|--:|
| 1 | gemini-cli/bot_v2.1 | 1582 | 8 | 2 | 0 |
| 2 | gemini-cli/bot_v2 | 1545 | 7 | 3 | 0 |
| 3 | claude-opus4p7/bot_hard_mcts | 1530 | 6 | 4 | 0 |
| 4 | claude-new/bot_v2 | 1500 | 5 | 5 | 0 |
| 5 | gpt-hard/bot | 1476 | 4 | 6 | 0 |
| 6 | gemini-cli/bot_v3 | 1367 | 0 | 10 | 0 |

### Gomoku

每对 100 局，`decision_timeout = 3.0s`。数据来源：[`gomoku/tourney_results.json`](gomoku/tourney_results.json)。

| 排名 | Bot | ELO | W | L | D |
|:---:|-----|----:|--:|--:|--:|
| 1 | claude_opus4p7/bot_hard | 1661 | 185 | 15 | 0 |
| 2 | deepseek_v4/bot_hard | 1529 | 115 | 85 | 0 |
| 3 | claude_opus4p7/bot_easy | 1311 | 0 | 200 | 0 |

### Othello

锚点：Edax（1300 / 1500 / 1700 / 1900 / 2100 / 2300 / 2500），每对 4 局，格式 W-L-D。数据来源：[`Othello/elo_results.json`](Othello/elo_results.json)。

| Bot | ELO | vs 1300 | vs 1500 | vs 1700 | vs 1900 | vs 2100 | vs 2300 | vs 2500 |
|-----|----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| gpt5p5/bot_hard | 2819 | 4-0-0 | 4-0-0 | 4-0-0 | 4-0-0 | 4-0-0 | 4-0-0 | 4-0-0 |
| claude_opus4p7/bot_hard | 2819 | 4-0-0 | 4-0-0 | 4-0-0 | 4-0-0 | 4-0-0 | 4-0-0 | 4-0-0 |

> 运行方式见 [`Othello/run_elo.py`](Othello/run_elo.py)。

### nimmt

每对 100 局（2 人对战），`decision_timeout = 3.0s`。数据来源：[`nimmt/tourney_results.json`](nimmt/tourney_results.json)。

| 排名 | Bot | ELO | W | L | D |
|:---:|-----|----:|--:|--:|--:|
| 1 | bot_greedy | 1562 | 193 | 95 | 12 |
| 2 | bot_easy | 1555 | 190 | 103 | 7 |
| 3 | bot_hard | 1465 | 118 | 174 | 8 |
| 4 | bot_random | 1418 | 82 | 211 | 7 |

### Skull

每对 100 局，`decision_timeout = 3.0s`。数据来源：[`skull/tourney_results.json`](skull/tourney_results.json)。

| 排名 | Bot | ELO | W | L | D |
|:---:|-----|----:|--:|--:|--:|
| 1 | bot_simple | 1688 | 200 | 0 | 0 |
| 2 | bot_easy | 1406 | 0 | 100 | 100 |
| 2 | bot_hard | 1406 | 0 | 100 | 100 |

> `bot_easy` 与 `bot_hard` 的 100 局对战全部平局。

### UNO

每对 100 局，`decision_timeout = 3.0s`。数据来源：[`uno/tourney_results.json`](uno/tourney_results.json)。

| 排名 | Bot | ELO | W | L | D |
|:---:|-----|----:|--:|--:|--:|
| 1 | gpt/bot_hard | 1509 | 53 | 47 | 0 |
| 2 | gpt/bot_easy | 1491 | 47 | 53 | 0 |

## Bot 接口约定

每个游戏目录下都有自己的 `doc/BOT_API.md`。通用约定是：

- Bot 文件通常命名为 `bot.py`。
- Bot 需要提供 `choose_action(state)` 函数，或提供带 `choose_action` 方法的 `Bot` 类。
- `choose_action(state)` 必须返回 `state["legal_actions"]` 中的一个字符串动作。
- Bot 抛异常或返回非法动作时，裁判会结束本局并判该 Bot 负。
- 对战命令支持 `--decision-timeout 秒` 限制单步决策耗时；超时会以 `timeout` 状态结束并判该 Bot 负。

## Git

本仓库远程地址：

```text
git@github.com:Sparky579/BoardArena.git
```
