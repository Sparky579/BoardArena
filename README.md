# BoardArena

BoardArena 是一个棋类/桌游 Bot 对战实验仓库。目前包含两个子项目：

- `lqq/`：路墙棋，支持本地网页对战和 2 人 Bot 裁判接口。
- `skull/`：Skull，支持多人 Bot 对战、批量评测和日志查询。

## 项目结构

```text
BoardArena/
├── lqq/
│   ├── index.html
│   ├── lqq_multi.py
│   ├── GAME_RULE.MD
│   └── BOT_API.md
├── skull/
│   ├── skull_multi.py
│   ├── skull_cfr.py
│   ├── README.md
│   └── BOT_API.md
└── README.md
```

## 路墙棋

打开网页对战：

```powershell
start .\lqq\index.html
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

- `lqq/GAME_RULE.MD`
- `lqq/BOT_API.md`

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

- `skull/README.md`
- `skull/BOT_API.md`

## Bot 接口约定

每个游戏目录下都有自己的 `BOT_API.md`。通用约定是：

- Bot 文件通常命名为 `bot.py`。
- Bot 需要提供 `choose_action(state)` 函数，或提供带 `choose_action` 方法的 `Bot` 类。
- `choose_action(state)` 必须返回 `state["legal_actions"]` 中的一个字符串动作。
- Bot 抛异常或返回非法动作时，裁判会结束本局并判该 Bot 负。

## Git

本仓库远程地址：

```text
git@github.com:Sparky579/BoardArena.git
```
