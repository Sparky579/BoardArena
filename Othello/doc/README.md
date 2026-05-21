# BoardArena Othello

这是 BoardArena 的黑白棋/Othello 子项目。规则核心由 `env/othello_env.py` 直接实现，不依赖第三方库；项目提供 Gym 风格环境、Bot 对战接口和本地浏览器对战 UI。

## 目录结构

```text
Othello/
├── env/
│   ├── othello_env.py
│   ├── othello_web.py
│   ├── index.html
│   ├── game.js
│   └── styles.css
├── baseline/
│   └── gpt5p5/
│       └── bot_easy.py
└── doc/
    ├── README.md
    ├── GAME_RULE.md
    └── BOT_API.md
```

## Gym 风格环境

```python
from env.othello_env import OthelloEnv

env = OthelloEnv(seed=1)
state, info = env.reset()
state, reward, terminated, truncated, info = env.step("d3")
```

`step(action)` 接收棋盘坐标字符串，例如 `d3`。如果当前玩家没有任何合法落子但对手有合法落子，唯一合法动作是 `PASS`。

返回值形状对齐 Gymnasium：

```python
(observation, reward, terminated, truncated, info)
```

## 浏览器 UI

启动本地服务：

```bash
python BoardArena/Othello/env/othello_web.py
```

默认访问：

```text
http://127.0.0.1:8030/
```

不要直接用 VS Code 的 Live Server 打开 `env/index.html`。这个页面需要 `othello_web.py` 提供 `/api/new`、`/api/action` 和 `/api/advance`，Live Server 只能提供静态文件，不能运行 Python 规则环境。

如果 `8030` 被占用，服务会自动尝试后续端口。也可以手动指定端口：

```bash
python BoardArena/Othello/env/othello_web.py --port 8031
```

可切换人人对战或人机对战。人机对战默认使用：

```text
BoardArena/Othello/baseline/gpt5p5/bot_easy.py
```

## Bot 对战

单局：

```bash
python BoardArena/Othello/env/othello_env.py battle --bot BoardArena/Othello/baseline/gpt5p5/bot_easy.py --games 1 --seed 1 --keep-logs
```

批量：

```bash
python BoardArena/Othello/env/othello_env.py battle --bot BoardArena/Othello/baseline/gpt5p5/bot_easy.py --games 100 --seed 1
```

限制单步决策最多 2 秒：

```bash
python BoardArena/Othello/env/othello_env.py battle --bot BoardArena/Othello/baseline/gpt5p5/bot_easy.py --games 100 --seed 1 --decision-timeout 2
```

生成最小示例 bot：

```bash
python BoardArena/Othello/env/othello_env.py sample-bot --output BoardArena/Othello/baseline/my_bot.py
```

更多规则和接口见：

- `doc/GAME_RULE.md`
- `doc/BOT_API.md`

