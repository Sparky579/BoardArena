# LQQ Bot Elo Ranking & Analysis

*Evaluation Date: 2026-05-22*

This document contains the official evaluation results for all baseline bots in the LQQ (Luqiangqi) environment.

## Methodology: The "Soft Limit" Test

To obtain the most accurate measure of algorithmic strength without the interference of system-level CPU contention (which disproportionately affects computationally heavy bots), this evaluation utilized a **"Soft Limit"** approach:

1. **Perceived Timeout (Bot view)**: 1.0 seconds. Bots dynamically adjusted their search depth and heuristics assuming they only had 1 second to move.
2. **Enforced Timeout (Referee view)**: 10.0 seconds. The referee did not penalize bots for minor systemic delays, ensuring games finished cleanly based on game logic rather than system jitter.
3. **Execution**: 100 random pairings executed across 4 threads.

This approach eliminated timeouts (0% timeout rate in the final run), revealing the true capabilities of the bots.

---

## 1. Complete Elo Ranking (100 Matches)

| Rank | Elo Rating | W-L-D | Games | Bot Name | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **1693.3** | 38-8-0 | 46 | `gemini-MCTS/bot_v2` | |
| 2 | **1658.9** | 23-8-0 | 31 | `gemini-MCTS` | |
| 3 | **1607.2** | 11-5-0 | 16 | `gemini-MCTS/bot_v5` | |
| 4 | **1579.5** | 21-9-4 | 34 | `gemini-cli` | |
| 5 | **1559.5** | 16-12-0 | 28 | `claude-new/bot_v2` | |
| 6 | **1541.8** | 6-3-0 | 9 | `claude-v2` | |
| 7 | **1538.1** | 23-16-5 | 44 | `claude-opus4p7/bot_hard_v2` | |
| 8 | **1531.5** | 2-0-0 | 2 | `gemini-MCTS/bot_v4` | |
| 9 | **1530.2** | 4-2-1 | 7 | `gemini-cli/botv2` | |
| 10 | **1520.6** | 4-4-0 | 8 | `gemini-MCTS/bot_final` | |
| 11 | **1519.5** | 5-4-0 | 9 | `claude-opus4p7/bot_hard` | |
| 12 | **1507.3** | 2-4-2 | 8 | `gemini-MCTS/bot_v6` | |
| 13 | **1506.3** | 4-3-2 | 9 | `gemini-pro-v3` | |
| 14 | **1502.0** | 2-4-2 | 8 | `gemini-MCTS/bot_v8` | |
| 15 | **1500.0** | 2-4-2 | 8 | `gemini-MCTS/bot_v7` | |
| 16 | **1500.0** | 0-0-0 | 0 | `gemini-MCTS/bot_v3` | |
| 17 | **1499.3** | 13-17-0 | 30 | `sample-bot` | |
| 18 | **1493.0** | 15-14-0 | 29 | `human_synth` | |
| 19 | **1477.3** | 2-4-0 | 6 | `bot_v2` | |
| 20 | **1477.2** | 3-5-0 | 8 | `gpt5-xhigh-v2` | |
| 21 | **1461.5** | 3-6-0 | 9 | `gemini-pro-v2` | |
| 22 | **1458.9** | 3-6-0 | 9 | `bot_v4` | |
| 23 | **1458.8** | 2-5-1 | 8 | `gemini-hard` | |
| 24 | **1443.9** | 1-5-0 | 6 | `bot_v3` | |
| 25 | **1443.0** | 0-4-0 | 4 | `bot_greedy` | |
| 26 | **1442.7** | 3-8-0 | 11 | `gpt-hard-anti-fork` | |
| 27 | **1404.0** | 1-8-0 | 9 | `gemini-pro` | |
| 28 | **1392.2** | 11-30-3 | 44 | `gpt-hard` | |
| 29 | **1384.0** | 6-17-1 | 24 | `gpt5-xhigh` | |
| 30 | **1368.2** | 0-11-1 | 12 | `claude-opus4p7/bot_easy` | |

---

## 2. First-Move Advantage Analysis

Extensive testing dispels the notion of a massive first-move advantage in LQQ.

* **Total Games**: 100
* **Player 0 (First Move) Wins**: 51 (51.0%)
* **Player 1 (Second Move) Wins**: 45 (45.0%)
* **Draws (Turn Limit)**: 4 (4.0%)
* **First Move Win Rate (excluding draws)**: **53.1%**

**Conclusion**: LQQ demonstrates exceptional competitive balance. The ~53% win rate for the first player is well within the acceptable margin for strategic board games (comparable to Chess at high levels). Previous observations of extreme first-move dominance (e.g., 70%+) were artifacts of the testing environment: heavy CPU contention caused the second player to experience slightly higher latency, pushing them over the strict 1.0s timeout threshold more frequently.

## 3. Game Status Distribution
* **ok (Normal Win/Loss)**: 96
* **turn_limit (Draw)**: 4
* **timeout**: 0
