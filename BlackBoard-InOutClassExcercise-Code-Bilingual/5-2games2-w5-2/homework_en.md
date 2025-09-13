# Homework — SAME Problem Programmatically: Expectimax on Dice-to-21

Implement:
1) **Game API** with chance nodes: `succ_max(s)` (actions), `succ_chance(s, a)` → outcomes with probs, `is_terminal(s)`, `utility(s)`.  
2) **expectimax(state, depth, utility_fn, eval_fn)** supporting MAX and CHANCE; depth-limited with cutoff + Eval.  
3) **Risk studies**: compare policies under (a) risk-neutral \(U(x)=x\), (b) risk-averse \(U(x)=\sqrt{x_+}-\lambda x_-\) for \(\lambda\in\{1,2,4\}\).  
4) **Sampling expectimax**: at chance nodes, use \(k\in\{2,4,8\}\) samples; report value error vs exact and node counts.  
5) **(Optional)** MCTS (UCT) baseline with 10k rollouts; compare move choice at \(s=18\).

Deliverables: code + short report (tables of values, node counts, plots of k vs error).
