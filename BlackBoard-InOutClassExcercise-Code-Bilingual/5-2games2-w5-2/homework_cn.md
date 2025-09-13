# 作业 — 同题程序实现：掷骰到 21 的期望极大

实现：
1) **带机会结点的接口**：`succ_max(s)`、`succ_chance(s, a)`（结果与概率）、`is_terminal(s)`、`utility(s)`。  
2) **expectimax(state, depth, utility_fn, eval_fn)**：支持 MAX/CHANCE，含深度截断与估值。  
3) **风险分析**：比较以下效用下的策略：(a) 风险中性 \(U(x)=x\)；(b) 风险厌恶 \(U(x)=\sqrt{x_+}-\lambda x_-\)，\(\lambda\in\{1,2,4\}\)。  
4) **采样期望极大**：在机会结点用 \(k\in\{2,4,8\}\) 样本；报告与精确值的误差及节点数。  
5) **（选做）** MCTS（UCT）10k 次 rollout；比较在 \(s=18\) 的选招。

提交：代码 + 简短报告（值/节点计数表与“k—误差”图）。
