# BoardArena go_9x9

这是 BoardArena 的九路围棋子项目。规则核心由 `env/go_env.py` 直接实现，不依赖第三方库；项目提供 Gym 风格环境、Bot 对战接口和本地浏览器对战 UI。

## 目录结构

```text
go_9x9/
├── env/
│   ├── go_env.py
│   ├── go_web.py
│   ├── index.html
│   ├── game.js
│   └── styles.css
├── baseline/
│   └── gpt5p5/
│       ├── bot_easy.py
│       └── bot_hard.py
└── doc/
    ├── README.md
    ├── GAME_RULE.md
    └── BOT_API.md
```

## Gym 风格环境

```python
from env.go_env import Go9x9Env

env = Go9x9Env(seed=1)
state, info = env.reset()
state, reward, terminated, truncated, info = env.step("e5")
```

`step(action)` 接收棋盘坐标字符串，例如 `e5`。跳过使用 `PASS`，且 `PASS` 始终是合法动作。

返回值形状对齐 Gymnasium：

```python
(observation, reward, terminated, truncated, info)
```

## 浏览器 UI

启动本地服务：

```bash
python BoardArena/go_9x9/env/go_web.py
```

默认访问：

```text
http://127.0.0.1:8040/
```

不要直接用 VS Code 的 Live Server 打开 `env/index.html`。这个页面需要 `go_web.py` 提供 `/api/new`、`/api/action` 和 `/api/advance`，Live Server 只能提供静态文件，不能运行 Python 规则环境。

如果 `8040` 被占用，服务会自动尝试后续端口。也可以手动指定端口：

```bash
python BoardArena/go_9x9/env/go_web.py --port 8041
```

可切换人人对战或人机对战。人机对战默认使用：

```text
BoardArena/go_9x9/baseline/gpt5p5/bot_hard.py
```

右侧下拉框可以选择 `/gpt5p5/bot_easy` 或 `/gpt5p5/bot_hard`。

## Bot 对战

单局：

```bash
python BoardArena/go_9x9/env/go_env.py battle --bot BoardArena/go_9x9/baseline/gpt5p5/bot_easy.py --games 1 --seed 1 --keep-logs
```

批量：

```bash
python BoardArena/go_9x9/env/go_env.py battle --bot BoardArena/go_9x9/baseline/gpt5p5/bot_easy.py --games 100 --seed 1
```

Hard bot 建议配合 2 秒单步限制运行：

```bash
python BoardArena/go_9x9/env/go_env.py battle --bot BoardArena/go_9x9/baseline/gpt5p5/bot_hard.py --games 10 --seed 1 --decision-timeout 2
```

指定贴目：

```bash
python BoardArena/go_9x9/env/go_env.py battle --bot BoardArena/go_9x9/baseline/gpt5p5/bot_hard.py --games 10 --komi 6.5
```

生成最小示例 bot：

```bash
python BoardArena/go_9x9/env/go_env.py sample-bot --output BoardArena/go_9x9/baseline/my_bot.py
```

更多规则和接口见：

- `doc/GAME_RULE.md`
- `doc/BOT_API.md`

