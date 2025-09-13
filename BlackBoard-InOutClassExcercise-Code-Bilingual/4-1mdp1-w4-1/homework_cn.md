# 作业 — 同题程序实现：骰子 MDP 的策略评估与价值迭代

实现一个小型 MDP 工具并求解骰子游戏：

1) **MDP 接口**：`states()`、`actions(s)`、`transitions(s,a)` → `(s', prob, reward)` 列表、`is_end(s)`、`start_state`。  
2) **policy_evaluation(mdp, policy, gamma, eps)**：返回 \(V^{\pi}\)，并满足 \(\max_s|Δ|\le\) eps；报告迭代次数与计数。  
3) **value_iteration(mdp, gamma, eps)**：返回 \(V^*, \pi^*\)；同时返回 \((\max_s|Δ|, \pi^{(t)})\) 迭代日志。  
4) **实验**：  
   - γ=1：比较 \(V^{\pi=\text{stay}}\)、\(V^{\pi=\text{quit}}\)、以及 \(V^*\)（在 **in** 的取值）。  
   - γ∈{0.0, 0.5, 0.9}：重复，分析最优策略何时翻转。  
5) **（选做）** 实现 **3×4 火山网格世界**（滑移概率 p、步长惩罚 r_step、终止奖励 r_goal/r_lava），做 10/20/50 次价值迭代展示“价值传播”。

**提交**：代码 + 1–2 页小报告（含 \(V\)、贪心策略随迭代的表/图与结论）。
