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
