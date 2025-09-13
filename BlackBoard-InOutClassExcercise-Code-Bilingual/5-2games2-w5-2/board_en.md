# Board (EN) — Games II (Chance Nodes · Expectimax · Evaluation · Risk · Sampling)

## Stochastic game trees
Players MAX/MIN plus **chance** nodes with known distribution \(P(o\mid s)\). Terminal utility \(U(s)\) (for MAX).

## Expectimax recursion
\[
V(s)=
\begin{cases}
U(s), & s \text{ terminal}\\
\max_{a\in A(s)} V(\mathrm{Succ}(s,a)), & s \text{ MAX}\\
\min_{a\in A(s)} V(\mathrm{Succ}(s,a)), & s \text{ MIN}\\
\sum_o P(o\mid s)\, V(\mathrm{Succ}(s,o)), & s \text{ CHANCE}
\end{cases}
\]

## Cutoff + evaluation
Depth limit \(D\): replace leaves by \(\mathrm{Eval}(s)=w^\top\phi(s)\).  
Design features aligned with task (threats, distance, risk). Tuning via small grid-search or regression.

## Risk attitudes via utilities
Replace payoff \(x\) by utility \(U(x)\).  
- **Risk-neutral**: \(U(x)=x\).  
- **Risk-averse**: concave \(U(x)=\sqrt{x_+}-\lambda x_-\).  
Decision rule stays **maximize expected utility** at MAX and minimize at MIN.

## Alpha–Beta?
No safe αβ pruning at chance nodes in general (non-linearity of max/min vs expectation).  
Practical heuristics: probability cut, beam on outcomes, **Monte Carlo expectimax** (sampling leaves).

## Monte Carlo & IDS
- **Sampling** at chance nodes: estimate \(\mathbb{E}[V]\) with \(k\) samples.  
- **Iterative deepening**: D=1,2,…; reuse principal variation for move ordering.
