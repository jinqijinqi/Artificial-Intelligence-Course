# Homework — SAME Problem Programmatically: Policy Evaluation & Value Iteration on Dice MDP

Implement a tiny MDP toolkit and solve the dice game:

1) **MDP interface** with `states()`, `actions(s)`, `transitions(s,a)` → list of `(s', prob, reward)`, `is_end(s)`, `start_state`.  
2) **policy_evaluation(mdp, policy, gamma, eps)** returning \(V^{\pi}\) with \(\max_s|Δ|\le\) eps; report iterations and runtime counts.  
3) **value_iteration(mdp, gamma, eps)** returning \(V^*, \pi^*\); also return an iteration log \((\max_s|Δ|, \pi^{(t)})\).  
4) **Experiments**:  
   - With γ=1: compare \(V^{\pi=\text{stay}}\), \(V^{\pi=\text{quit}}\), and \(V^*\) (values at **in**).  
   - With γ∈{0.0, 0.5, 0.9}: repeat; analyze when the optimal policy flips.  
5) **(Optional)** Add a **3×4 Volcano GridWorld** MDP (slip probability p, step reward r_step, terminal rewards r_goal/r_lava) and run value iteration for 10, 20, 50 iterations to illustrate “value propagation”.

**Deliverables**: code + a short report (1–2 pages) with tables/plots of \(V\), greedy policy vs iteration, and conclusions.
