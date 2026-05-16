# BoardArena Chess

这是一个严格按国际象棋规则运行的 BoardArena 子项目。规则核心由 `python-chess` 提供，项目本身负责封装 Gym 风格环境、Bot 对战接口和本地浏览器对战 UI。

## 目录结构

```text
chess/
├── env/
│   ├── chess_env.py
│   ├── chess_web.py
│   ├── index.html
│   ├── game.js
│   └── styles.css
├── baseline/
│   └── gpt5p5/
│       └── bot.py
├── doc/
│   ├── README.md
│   ├── GAME_RULE.md
│   └── BOT_API.md
└── requirements.txt
```

## 安装依赖

```bash
python -m pip install -r BoardArena/chess/requirements.txt
```

如果当前工作目录已经是 `BoardArena/chess`：

```bash
python -m pip install -r requirements.txt
```

## Gym 风格环境

```python
from env.chess_env import ChessEnv

env = ChessEnv(seed=1)
state, info = env.reset()
state, reward, terminated, truncated, info = env.step("e2e4")
```

`step(action)` 接收 UCI 字符串动作，例如：

- `e2e4`：普通移动。
- `e1g1`：白方短易位。
- `e7e8q`：升变为后。

返回值形状对齐 Gymnasium：

```python
(observation, reward, terminated, truncated, info)
```

## 浏览器 UI

启动本地服务：

```bash
python BoardArena/chess/env/chess_web.py
```

默认访问：

```text
http://127.0.0.1:8020/
```

不要直接用 VS Code 的 Live Server 打开 `env/index.html`。这个页面需要 `chess_web.py` 提供 `/api/new` 和 `/api/action`，Live Server 只能提供静态文件，不能运行 Python 规则环境。

如果 `8020` 被占用，服务会自动尝试后续端口。也可以手动指定端口：

```bash
python BoardArena/chess/env/chess_web.py --port 8021
```

可切换人人对战或人机对战。人机对战默认使用：

```text
BoardArena/chess/baseline/gpt5p5/bot.py
```

指定端口或 bot：

```bash
python BoardArena/chess/env/chess_web.py --port 8021 --bot BoardArena/chess/baseline/gpt5p5/bot.py
```

## Bot 对战

单局：

```bash
python BoardArena/chess/env/chess_env.py battle --bot BoardArena/chess/baseline/gpt5p5/bot.py --games 1 --seed 1 --keep-logs
```

批量：

```bash
python BoardArena/chess/env/chess_env.py battle --bot BoardArena/chess/baseline/gpt5p5/bot.py --games 100 --seed 1
```

生成最小示例 bot：

```bash
python BoardArena/chess/env/chess_env.py sample-bot --output BoardArena/chess/baseline/my_bot.py
```

更多规则和接口见：

- `doc/GAME_RULE.md`
- `doc/BOT_API.md`
