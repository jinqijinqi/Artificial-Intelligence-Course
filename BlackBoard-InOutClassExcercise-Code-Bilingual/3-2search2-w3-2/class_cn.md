# 课堂练习 — 同题：带约束 → 松弛 → A*

**原问题（带约束的交通）**  
状态 \(s=(\text{loc}, \Delta)\)，\(\Delta=\#\text{walk}-\#\text{tram}\ge 0\)。  
起点 \((1,0)\)，终点 \((n,\Delta\ge0)\)。动作：
- walk: \( (loc,\Delta)\to(loc+1,\Delta+1)\)，代价 1  
- tram: \( (loc,\Delta)\to(2\cdot loc,\Delta-1)\)（需 \(\Delta-1\ge0\)），代价 2

**松弛问题**：移除 \(\Delta\ge0\) 约束；状态仅为 **location**。  
计算 \( \mathrm{FutureCost}_{\text{rel}}(\text{loc})\)（在**反向**松弛图上用 UCS）。

**启发**：\(h((\text{loc},\Delta)) := \mathrm{FutureCost}_{\text{rel}}(\text{loc})\)。

## 任务
1) 用“松弛 ⇒ 一致”定理证明 \(h\) 一致。  
2) \(n=30\) 时手算 A* 前 8 次弹出（写 \(g+h\)）并与 UCS 对比。  
3) 列 loc=1..16 的 \( \mathrm{FutureCost}_{\text{rel}}(\text{loc})\) 表。  
4) 进阶：\(h_0=0\)、\(h_{\text{walk}}=n-\text{loc}\)，验证 \(h_{\max}=\max(h,h_{\text{walk}})\) 一致。
