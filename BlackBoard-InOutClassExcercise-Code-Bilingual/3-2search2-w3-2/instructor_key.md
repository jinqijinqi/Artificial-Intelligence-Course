# Instructor Key — A* & Relaxations on Constrained Transportation

**一致性证明（摘要）**  
松弛问题 \(P_{\text{rel}}\) 满足 \( \mathrm{Cost}_{\text{rel}}\le \mathrm{Cost}\)。  
\(h(s)=\mathrm{FutureCost}_{\text{rel}}(\text{loc}(s))\)。  
由三角不等式：\(h(s)\le \mathrm{Cost}_{\text{rel}}(s,a)+h(\mathrm{Succ}(s,a))\le \mathrm{Cost}(s,a)+h(\mathrm{Succ}(s,a))\)；且 \(h(\text{end})=0\)。  
故 \(h\) 一致；A* 返回最优路径。

**效率（示例 n=60）**  
通常：UCS 展开 ≫ A*(h0) ≫ A*(h_walk) ≳ A*(h_relaxed)；h_max 进一步减少。  
（以实际运行打印展开节点数为准。）

**A* 望远镜公式**  
\(\sum \mathrm{Cost}' = \sum \mathrm{Cost} + h(\text{end})-h(\text{start})\)，故改边代价仅差常数。
