# Class — SAME Problem: Dice-to-21 (Expectimax with Chance Nodes)

**Rules**  
Score \(s\in\{0,\dots\}\). On each turn, MAX chooses **roll** or **stop**.  
- **stop**: terminal payoff \(U = s\).  
- **roll**: chance node; add \(X\sim\mathrm{Unif}\{1,\dots,6\}\) to score. If new score \(>21\) (**bust**), terminal payoff \(U=-10\).

**Tasks**
1) **Depth-2 expectimax** from \(s=18\): compute value of `roll` vs `stop` under **risk-neutral** \(U(x)=x\). Which action is better?  
2) **Risk-averse** utility \(U(x)=\sqrt{\max(x,0)} - 2\cdot \max(-x,0)\). Recompute (1). Does the decision change?  
3) **Cutoff \(D=4\) with Eval**: propose \(\phi(s)=(s,\ \mathbb{1}[s\ge 20],\ \mathbb{1}[s\le 15])\) and a linear \(w\). Evaluate a partial tree at \(s=12\).  
4) **Sampling**: at a roll node, estimate expectation with \(k=3\) samples; discuss variance vs bias.

**Deliverables**: your tree sketches and numeric values (show arithmetic).
