# 板书（中文）— 搜索 1（建模 · 树搜索 · 动态规划 · 一致代价搜索）

## 搜索问题建模
- **状态** \(s\)：对过去的**最小充分**摘要，使得后续可最优决策。
- **起始** \(s_{\text{start}}\)、**IsEnd**(s)、**Actions**(s)、**Succ**(s,a)、**Cost**(s,a)。
- **目标**：起点到终点的总成本最小。

## 树搜索（分支因子 b，最大深度 D，解深度 d）
- **回溯**（任意代价）：时间 \(O(b^D)\)，空间 \(O(D)\)。
- **DFS**（假设所有代价为 0）：遇到第一个终点即停；最坏时间 \(O(b^D)\)，空间 \(O(D)\)。
- **BFS**（假设所有代价相等 \(c\ge0\)）：时间/空间 \(O(b^d)\)。
- **DFS-ID 迭代加深**（与 BFS 同假设）：时间 \(O(b^d)\)，空间 \(O(d)\)。

## 动态规划（状态图无环）
**未来代价**递推：
\[
\mathrm{FutureCost}(s)=
\begin{cases}
0, & \text{若 IsEnd}(s)\\[2mm]
\min\limits_{a\in \mathrm{Actions}(s)}\ [\mathrm{Cost}(s,a)+\mathrm{FutureCost}(\mathrm{Succ}(s,a))],& \text{否则}
\end{cases}
\]
用记忆化避免重复计算 → 当状态量不大时从指数降到多项式。

## 一致代价搜索（UCS）— 非负代价
- 按**过去代价** \( \mathrm{PastCost}(s)\) 递增次序枚举。
- 维护 **Explored/Frontier/Unexplored**；从 frontier 弹出最小优先级。
- 在 \( \mathrm{Cost}(s,a)\ge0\) 下正确；等价于 Dijkstra（在隐式图上）。

## 复杂度要点
- 最坏时间指数，但空间可用 DFS-ID 降为线性。
- 依据代价假设与内存预算选择算法。

*（与讲稿：应用、超越“反射式”、树搜索家族、DP、UCS 内容一致。）*
