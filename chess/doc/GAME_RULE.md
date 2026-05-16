# 国际象棋规则

本文档描述 `BoardArena/chess` 的规则范围。规则判定由 `python-chess` 完成，目标是严格覆盖标准国际象棋规则，而不是在仓库中手写走法生成器。

## 基本设置

- 支持玩家数：固定 `2` 人。
- 玩家 `0`：白方，先手。
- 玩家 `1`：黑方，后手。
- 初始局面：标准国际象棋初始 FEN。
- 棋盘坐标：代数坐标，文件为 `a..h`，横线为 `1..8`。

## 行动格式

所有动作使用 UCI 字符串：

- 普通走子：`e2e4`。
- 吃子：同样使用起点和终点，例如 `e4d5`。
- 短易位：白方 `e1g1`，黑方 `e8g8`。
- 长易位：白方 `e1c1`，黑方 `e8c8`。
- 升变：在末尾追加升变棋子，例如 `e7e8q`、`a2a1n`。

## 规则覆盖

环境使用 `python-chess` 的合法走法列表，因此支持：

- 王、后、车、象、马、兵的标准走法。
- 将军、应将、将死。
- 王车易位，包括王或车移动后不可易位、穿越受攻击格不可易位等限制。
- 吃过路兵。
- 兵升变为后、车、象、马。
- 无子可动逼和。
- 子力不足和棋。
- 50 回合规则、75 回合规则、三次重复、五次重复。

`ChessEnv` 默认 `claim_draw=True`，因此当 50 回合规则或三次重复已经可以声明和棋时，环境会自动结束为和棋。这适合自动评测；如果需要模拟必须由玩家声明的规则，可在创建环境时传入 `claim_draw=False`。

## 胜负

- 将死对手：将死方获胜。
- 逼和、子力不足、重复局面、回合规则等：胜者为 `None`。
- 达到环境 `max_plies` 上限仍未结束：状态为 `turn_limit`，胜者为 `None`。
- Bot 对战中，bot 抛异常或返回非法动作：该 bot 判负。

## 终局状态

常见 `status`：

- `checkmate`
- `stalemate`
- `insufficient_material`
- `seventyfive_moves`
- `fivefold_repetition`
- `fifty_moves`
- `threefold_repetition`
- `turn_limit`
- `invalid_action`
- `bot_exception`
