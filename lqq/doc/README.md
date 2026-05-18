# 路墙棋对战

这是一个纯前端本地双人路墙棋实现，直接打开 `index.html` 即可运行。

## 规则

- 棋盘为 9x9。
- 蓝方从下方中点出发，目标是到达最上方任一格。
- 红方从上方中点出发，目标是到达最下方任一格。
- 每回合只能选择一种行动：向前、后、左、右移动一格，或放置一堵墙。
- 墙为两格长，可横放或竖放。
- 墙不能重叠、交叉，也不能让任一玩家完全没有到达目标边的路线。
- 每方 10 堵墙，先到达对面底线者获胜。

## 操作

- 点击高亮格子移动棋子。
- 点击棋盘墙槽可按槽方向放墙。
- 点击墙槽交叉点时，使用右侧选择的横墙/竖墙方向。
- 可以悔棋或重新开始。

## Bot 对战

单局对战：

```powershell
python .\lqq_multi.py battle --bot .\bot.py --players 2 --games 1 --seed 1 --keep-logs
```

批量对战并限制单步决策最多 2 秒：

```powershell
python .\lqq_multi.py battle --bot .\bot.py --players 2 --games 100 --seed 1 --decision-timeout 2
```

更多接口说明见 `BOT_API.md`。
