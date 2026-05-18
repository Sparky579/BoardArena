# Simplified Skull CFR AI

这是一个两人简化版《Skull / 骷髅牌》AI。项目包含：

- 一个简化规则的游戏状态机；
- 一个采样式信息集 CFR 训练器；
- 一个命令行自战 / 人机对战入口；
- 一个本地浏览器 UI。

目标不是做严格完美求解器，而是在普通 CPU 上快速训练出一个可玩的不完全信息博弈策略。

## 规则

- 两名玩家，每人初始 `3` 张花和 `1` 张骷髅。
- 每轮玩家轮流暗放一张牌。
- **双方都至少放过一张牌后，才允许开始叫号。**
- play 阶段可以继续放牌，也可以叫 `1..桌面总牌数`。
- 进入 bid 阶段后，只能加叫更大的数字，或者 `PASS`。
- 两人局里，一方 `PASS` 后立即结算当前最高叫号。
- 结算时，最高叫号者必须先从自己的牌堆翻，翻完自己的牌还不够，再翻对手牌堆。
- 翻到任意骷髅则叫号失败；全部翻到花则叫号成功。
- 叫号成功得 `1` 分，先得 `2` 分获胜。
- 叫号失败会随机损失自己的一张牌；如果某方没有牌，另一方获胜。

## 算法说明

核心训练器在 `skull_cfr.py` 里的 `CFRTrainer`。

它使用的是一个简化版、采样式的 counterfactual regret minimization 思路：

1. 从初始状态开始模拟一局。
2. 每次迭代分别更新 P0 和 P1。
3. 当轨迹走到当前被更新玩家的信息集时，训练器会枚举这个信息集下所有合法动作。
4. 对每个合法动作做若干次 rollout，估计这个动作的价值。
5. 用动作价值和当前策略价值的差，更新 regret。
6. 用 regret matching 产生当前策略。
7. 把训练过程中产生的策略累积起来，最后导出 average policy。

这个实现不是完整遍历博弈树的 CFR。它为了速度做了几个近似：

- 每条训练轨迹只更新有限个信息集。
- 非当前更新玩家的动作按当前已有策略或 fallback 策略采样。
- 动作价值靠 rollout 估计，不是精确算完整子树。
- 导出的策略是平均策略，因此早期探索过的动作会留下少量残余概率。

这些近似让训练能很快跑起来，但也意味着策略会有采样噪声，尤其在某些稀有信息集上。

## 信息集

项目支持两种 recall 模式。

### compact

默认模式：

```powershell
python .\skull_cfr.py train --recall compact
```

compact 信息集保留：

- 当前分数；
- 自己手牌数量；
- 自己已放出的牌堆内容；
- 对手剩余总牌数；
- 对手牌堆数量；
- 当前阶段；
- 当前叫号；
- 当前最高叫号者是自己、对手还是没人。

compact 不记录完整公开历史，因此不同历史可能被压到同一个信息集里。好处是信息集少、训练样本更集中、收敛更快；坏处是策略表达能力更弱。

### perfect

perfect 模式会额外把公开历史加入信息集：

```powershell
python .\skull_cfr.py train --recall perfect --out skull_policy_perfect.json
```

它更接近标准 CFR 的完美记忆假设，理论表达能力更强。但代价是信息集数量明显增加，同样训练预算下，每个信息集被访问和更新的次数会减少，所以更容易出现训练不足。

## 当前训练参数

默认训练命令：

```powershell
python .\skull_cfr.py train --iterations 10000 --out skull_policy.json
```

当前默认参数：

- `--recall compact`
- `--explore 0.02`
- `--rollouts-per-action 16`
- `--updates-per-trajectory 8`
- `--eval-games 2000`

之前版本里 `rollouts-per-action` 是 `1`，动作价值估计方差很大。现在改成 `16`，每个候选动作会用更多 rollout 估值，训练更慢，但明显更稳。

## 使用

训练默认策略：

```powershell
python .\skull_cfr.py train --iterations 10000 --out skull_policy.json
```

训练 perfect recall 策略：

```powershell
python .\skull_cfr.py train --iterations 100000 --recall perfect --out skull_policy_perfect.json
```

CPU 自战评估：

```powershell
python .\skull_cfr.py battle --policy skull_policy.json --games 10000
```

打印一局 CPU 自战过程：

```powershell
python .\skull_cfr.py battle --policy skull_policy.json --trace
```

人机对战：

```powershell
python .\skull_cfr.py play --policy skull_policy.json --human 0
```

浏览器 UI：

```powershell
python .\env\skull_web.py --policy skull_policy.json
```

打开后访问 `http://127.0.0.1:8000/`。如果端口被占用，可以换端口：

```powershell
python .\env\skull_web.py --policy skull_policy.json --port 8010
```

浏览器 UI 会默认加载：

- `skull_policy.json` 作为 Compact AI；
- `skull_policy_perfect.json` 作为 Perfect AI。

页面右上角可以在新局前选择 `Compact AI` 或 `Perfect AI`。选择只影响新开的局，不会切换正在进行的 session。

多人 Bot 对战：

```powershell
python .\skull_multi.py battle --bot .\bot.py --players 4 --games 100 --seed 1 --decision-timeout 2
```

`--decision-timeout` 用于限制每次 `choose_action(state)` 的最大秒数，超时会以 `timeout` 状态结束并判该 Bot 负。更多接口说明见 `BOT_API.md`。

## 更新记录

### 2026-05-10

规则更新：

- 旧规则：桌上只要有任意一张牌，就允许叫号。
- 新规则：双方都至少放过一张牌后，才允许叫号。

训练稳定性更新：

- `rollouts_per_action` 默认从 `1` 改为 `16`。
- CLI 的 `updates_per_trajectory` 默认从 `4` 改为 `8`，与 `CFRTrainer` 类默认值保持一致。
- 重新训练并更新了 `skull_policy.json`。
- 从头重训了 `skull_policy_perfect.json`：`100000` iterations，`recall=perfect`，`rollouts_per_action=2`，`updates_per_trajectory=8`，最终 `infosets=5560`。
- Web UI 增加 Compact / Perfect 策略选择，并修正页面文案为新规则。

这次改动的背景是一个具体坏例子：

```text
CPU 自己牌堆：骷髅
玩家牌堆：花
当前没人叫号
轮到 CPU
```

在这个 play 阶段，CPU 主动喊 `2` 明显很差。喊 `1` 至少还能诱导玩家加到 `2`，但直接喊 `2` 会让 CPU 自己成为挑战者，玩家只要 `PASS`，CPU 就必须先翻自己的骷髅并失败。

问题不是结算规则写错，而是旧训练设置中每个动作只 rollout 一次，估值噪声太大，早期把 `BID_2` 错误推高。提高 rollout 数后，这类坏动作的概率明显下降。

## 对战记录

以下是一组 compact policy 和 perfect policy 的对战结果：

```text
1000 局
双方交替先手
seed = 2
双方都使用随机采样策略

Perfect AI wins: 464
Policy AI wins:  536

Perfect win rate: 46.400%
Policy win rate:  53.600%

Perfect as P0: 236/500
Perfect as P1: 228/500
Policy as P0:  272/500
Policy as P1:  264/500

Seat wins: P0 508 - P1 492
```

这组结果里，座位优势很小：

```text
P0 508 - P1 492
```

所以 53.6% vs 46.4% 主要不是先后手造成的，而是两个策略本身在当前训练预算下的表现差异。

## 为什么 perfect 反而略差

直觉上，perfect recall 信息更多，应该更强。但在当前实现和训练预算下，它反而可能略差，原因是：

1. **信息集变多，样本被分散了。**

   perfect 会把公开历史放进 key。同一个局面只要历史路径不同，就会进入不同信息集。这样每个信息集被访问的次数变少，regret 更新也更稀疏。

2. **当前训练器是采样式近似，不是完整 CFR。**

   它不会完整遍历所有状态和所有后续分支，而是沿采样轨迹更新少量信息集。信息集越多，越容易出现“某些局面没训练够”的情况。

3. **compact 有一种意外的平滑效果。**

   compact 把相似局面合并了。理论上这会损失信息，但实践里它也等于把训练样本合在一起，让策略更平滑、更稳。训练预算不大时，这种泛化可能比 perfect 的细粒度记忆更有用。

4. **双方使用的是随机采样策略，不是 greedy 策略。**

   `policy_action(..., greedy=False)` 会按概率抽样。average policy 里残留的小概率坏动作仍然可能被抽到。perfect 信息集更多，每个信息集样本更少，小概率噪声可能更明显。

5. **1000 局仍然有统计波动。**

   53.6% 对 46.4% 是一个有意义的信号，但不是严格证明 compact 一定更强。更稳的结论需要更多局数、多个 seed、以及 greedy / sampled 两种模式分别评估。

因此，这组结果更准确的解释是：

> 在当前采样式训练器、当前训练预算、随机采样对战方式下，compact policy 比 perfect policy 更稳；perfect 的理论表达力更强，但还没有被训练预算充分兑现。

## 代码结构

- `State`: 简化骷髅牌状态机，包含合法动作、动作转移、随机抽牌和信息集编码。
- `CFRTrainer`: 采样式递归信息集 CFR 训练器。
- `policy_action`: 从平均策略中采样动作；未见过的信息集使用 fallback 策略。
- `env/skull_web.py`: 本地浏览器 UI 和简单 JSON API，复用同一套状态机和策略。

## 已知限制

- 当前 CFR 是实用近似，不是严格完整求解。
- average policy 会保留早期探索残余概率。
- perfect recall 需要更高训练预算，否则可能不如 compact 稳。
- Web UI 默认使用随机采样策略，因此 AI 偶尔会做低概率坏动作。
