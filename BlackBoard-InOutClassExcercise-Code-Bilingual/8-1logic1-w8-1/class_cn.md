# 课堂练习 — 同题：Rain–Wet–Slippery 知识库

**KB** = { Rain, Rain → Wet, Wet → Slippery }，原子 {Rain, Wet, Slippery}。

任务
1) **推导（仅用肯定前件）**：尽可能推导新公式。出现了哪些？
2) **用 SAT 思考做 Ask/Tell**（手算逻辑，不用代码）：
   a) **Wet** 是否被蕴含？检查 KB ∪ {¬Wet} 是否可满足？
   b) **Rain → Slippery** 是否被蕴含？
   c) **¬Rain** 是否与 KB 矛盾？
3) **或然**：**Snow** 对 KB 是否或然？用“模型集”直觉解释。
4) **M(KB) 收缩**：若 Tell[¬Wet]，系统应给出何种响应（已知/拒绝/学习到新信息）？
提交：一页，含 (i) 推导结果，(ii) 蕴含/矛盾/或然判断与 1–2 行理由。
