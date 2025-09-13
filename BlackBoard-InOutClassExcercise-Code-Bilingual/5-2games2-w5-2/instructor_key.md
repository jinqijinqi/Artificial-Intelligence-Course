# Instructor Key — Dice-to-21

**Task 1 (risk-neutral, s=18, depth=2)**  
- `stop`: value \(=18\).  
- `roll`: immediate outcomes add {1..6}. Bust on {4,5,6} → payoff \(-10\); safe on {1,2,3} → states 19,20,21 are terminal next step under depth=2.  
  - 19: if then `stop`, payoff 19;  
  - 20: `stop`→20;  
  - 21: `stop`→21.  
  Expectation \(=\tfrac{1}{6}(19+20+21) + \tfrac{3}{6}(-10) = \tfrac{60}{6} - \tfrac{30}{6} = 5\).  
  So at depth-2, `stop` (18) > `roll` (5) for risk-neutral.

**Task 2 (risk-averse)**  
Bust penalty weights more; `stop` remains preferred. For lower s (e.g., s=10), risk-neutral may prefer rolling; for large \(\lambda\) risk-averse prefers stopping earlier.

**Task 3 (cutoff + eval)**  
With \(\mathrm{Eval}(s)=1\cdot s + 5\mathbf{1}[s\ge 20] - 3\mathbf{1}[s\le 15]\), students should show how partial trees are evaluated at depth limit.

**Sampling**  
Variance decreases as \(k\) grows; node count drops vs exact expectimax; unbiased for leaves using eval; small bias if eval is imperfect at cutoffs.
