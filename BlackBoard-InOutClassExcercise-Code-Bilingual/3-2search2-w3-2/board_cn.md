# 板书（中文）— 搜索 2（UCS 正确性 · A* · 一致/可采纳启发 · 松弛）

## UCS 正确性（图搜索，非负代价）
Frontier/Explored 不变式：弹出 s 时优先级 = PastCost(s)（到 s 的最小代价）。  
边界交叉 + 非负代价 ⇒ 终点进入 explored 时代价最优。

## A* = 在修改代价上的 UCS
启发 \(h(s)\) 近似 FutureCost(s)。  
修改代价：\( \mathrm{Cost}'(s,a)=\mathrm{Cost}(s,a)+h(\mathrm{Succ}(s,a)) - h(s)\)。  
在 \(\mathrm{Cost}'\) 上跑 **UCS** ⇔ 以 \( \mathrm{PastCost}(s)+h(s)\) 排序。

### 一致性（图搜索最优性）
若 \(h\) **一致**：1) \( \mathrm{Cost}'(s,a)\ge 0\)；2) \(h(\text{end})=0\)。

### 望远镜恒等式（A* 正确性）
\(\sum \mathrm{Cost}' = \sum \mathrm{Cost} + h(s_L)-h(s_0)\)。  
因 \(h(\text{end})=0\)、\(h(s_0)\) 常数，最小化修改路经代价等价于原问题。

### A* 的效率
A* 仅探索满足 \( \mathrm{PastCost}(s)+h(s)\le \mathrm{PastCost}(\text{end})\) 的状态；\(h\) 越大且一致 ⇒ 搜索越省。

## 松弛启发（统一框架）
把原问题 \(P\) 放宽成 \(P_{\text{rel}}\)，满足 \( \mathrm{Cost}_{\text{rel}}(s,a)\le \mathrm{Cost}(s,a)\)。  
令 \( h(s)=\mathrm{FutureCost}_{\text{rel}}(s)\) ⇒ \(h\) 一致。  
计算：封闭解（如曼哈顿）、更易搜索、独立子问题；可用 \( \max(h_1,h_2)\) 组合（仍一致）。

## DP vs UCS（回顾）
DP：无环、任意代价、遍历全部 \(N\)；UCS：可有环、需非负、仅到达目标代价阈值以内的状态。
