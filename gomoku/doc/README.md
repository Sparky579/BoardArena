# BoardArena gomoku

BoardArena 的五子棋（Renju）子项目，15×15 棋盘，**黑方禁手生效**（长连 / 四四 / 三三）。规则核心由 `env/gomoku_env.py` 直接实现，不依赖第三方库。

## 目录结构

```text
gomoku/
├── env/
│   ├── gomoku_env.py
│   ├── gomoku_web.py
│   ├── index.html
│   ├── game.js
│   ├── styles.css
│   └── __init__.py
├── baseline/
│   └── claude_opus4p7/
│       ├── bot_easy.py
│       └── bot_hard.py
└── doc/
    ├── README.md
    ├── GAME_RULE.md
    └── BOT_API.md
```

## Gym 风格环境

```python
from env.gomoku_env import GomokuEnv

env = GomokuEnv(seed=1)
state, info = env.reset()
state, reward, terminated, truncated, info = env.step("h8")
```

`step(action)` 接收坐标字符串，例如 `h8`、`a1`、`o15`。没有 `PASS`。

返回值形状对齐 Gymnasium：

```python
(observation, reward, terminated, truncated, info)
```

## 浏览器 UI

启动本地服务：

```bash
python BoardArena/gomoku/env/gomoku_web.py
```

默认访问：

```text
http://127.0.0.1:8050/
```

不要直接用 VS Code 的 Live Server 打开 `env/index.html`。该页面需要 `gomoku_web.py` 提供 `/api/new`、`/api/action` 和 `/api/advance`，Live Server 只能提供静态文件。

如果 `8050` 被占用，服务会自动尝试后续端口。也可以手动指定：

```bash
python BoardArena/gomoku/env/gomoku_web.py --port 8051
```

支持人人对战或人机对战。右侧下拉框可以选择 `/claude_opus4p7/bot_easy` 或 `/claude_opus4p7/bot_hard`。人机对战默认使用：

```text
BoardArena/gomoku/baseline/claude_opus4p7/bot_hard.py
```

棋盘上所有 "你可以落子"的空点会有半透明提示；黑方被禁手过滤掉的点不会显示提示。

## Bot 对战

单局：

```bash
python BoardArena/gomoku/env/gomoku_env.py battle --bot BoardArena/gomoku/baseline/claude_opus4p7/bot_easy.py --games 1 --seed 1 --keep-logs
```

批量：

```bash
python BoardArena/gomoku/env/gomoku_env.py battle --bot BoardArena/gomoku/baseline/claude_opus4p7/bot_hard.py --games 20 --seed 1 --decision-timeout 2
```

两个 Bot 互相对战：

```bash
python BoardArena/gomoku/env/gomoku_env.py duel --bot0 BoardArena/gomoku/baseline/claude_opus4p7/bot_easy.py --bot1 BoardArena/gomoku/baseline/claude_opus4p7/bot_hard.py --games 20 --seed 1 --decision-timeout 2
```

更多规则和接口见：

- `doc/GAME_RULE.md`
- `doc/BOT_API.md`
