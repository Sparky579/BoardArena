# BoardArena UNO

这是 BoardArena 的二人 UNO 子项目，包含纯 Python 规则环境、Bot 对战裁判、批量评测接口、本地浏览器对战 UI 和 Tkinter 人机对战 UI。

## 目录结构

```text
uno/
├── uno_ui.py
├── env/
│   ├── uno_env.py
│   ├── uno_web.py
│   ├── index.html
│   ├── game.js
│   └── styles.css
├── baseline/
│   └── gpt/
│       ├── bot_easy.py
│       └── bot_hard.py
├── doc/
│   ├── README.md
│   ├── GAME_RULE.md
│   └── BOT_API.md
└── requirements.txt
```

## 环境

UNO 子项目不依赖第三方包。可选执行：

```bash
python -m pip install -r BoardArena/uno/requirements.txt
```

## Gym 风格环境

```python
from env.uno_env import UnoEnv

env = UnoEnv(seed=1)
state, info = env.reset()
action = state["legal_actions"][0]
state, reward, terminated, truncated, info = env.step(action)
```

`step(action)` 接收字符串动作：

- `play:<card_id>`：打出普通颜色牌。
- `play:<card_id>:<color>`：打出 Wild 或 Wild Draw Four，并声明颜色。
- `draw`：没有可出牌时摸 1 张，回合结束。
- `pass`：牌堆无法摸牌且没有可出牌时跳过。

返回值形状对齐 Gymnasium：

```python
(observation, reward, terminated, truncated, info)
```

## 浏览器 UI

启动本地服务：

```bash
python BoardArena/uno/env/uno_web.py
```

默认访问：

```text
http://127.0.0.1:8060/
```

不要直接用 Live Server 打开 `uno/env/index.html`，页面需要 `uno_web.py` 提供 JSON API。

如果 `8060` 被占用，服务会自动尝试后续端口，也可以手动指定：

```bash
python BoardArena/uno/env/uno_web.py --port 8061
```

人机模式默认使用：

```text
BoardArena/uno/baseline/gpt/bot_hard.py
```

`bot_hard.py` 使用自包含的 determinized MCTS：每次行动会按公开信息随机补全隐藏手牌和摸牌堆，在多个模拟世界中评估候选动作；根节点先按 UNO 二人局的强制牌优先级收窄候选，再用搜索修正颜色和同级动作选择。

## Python 桌面 UI

启动 Tkinter 人机对战界面：

```bash
python BoardArena/uno/uno_ui.py
```

可指定人类座位、随机种子和 bot：

```bash
python BoardArena/uno/uno_ui.py --human-seat 1 --seed 1 --bot /gpt/bot_hard
```

该界面只在 `uno/` 子项目内工作，使用 `uno/env/uno_env.py` 作为规则裁判，默认从 `uno/baseline/` 自动发现可用 bot。

## Bot 对战

单局：

```bash
python BoardArena/uno/env/uno_env.py battle --bot BoardArena/uno/baseline/gpt/bot_hard.py --games 1 --seed 1 --keep-logs
```

批量：

```bash
python BoardArena/uno/env/uno_env.py battle --bot BoardArena/uno/baseline/gpt/bot_easy.py --games 100 --seed 1
```

限制单步决策最多 2 秒：

```bash
python BoardArena/uno/env/uno_env.py battle --bot BoardArena/uno/baseline/gpt/bot_hard.py --games 100 --seed 1 --decision-timeout 2
```

更多规则和接口说明见：

- `doc/GAME_RULE.md`
- `doc/BOT_API.md`
