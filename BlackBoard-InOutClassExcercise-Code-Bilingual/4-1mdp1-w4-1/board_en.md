# Board (EN) — MDPs 1 (Overview · Modeling · Policy Evaluation · Value Iteration)

## MDP (finite, episodic) definition
States **S**, start **s₀**, actions **A(s)**.  
Transition **T(s,a,s') = P(s' | s,a)** (∑_{s'} T = 1).  
Reward **R(s,a,s')**.  End **IsEnd(s)**.  Discount **γ∈[0,1]**.

Search as a special case:
- Deterministic: **Succ(s,a)** ⇒ **T(s,a,s') = 1[s'=Succ(s,a)]**.
- Costs vs rewards: **R = −Cost**.

## Policy π, Value and Q-value
Policy **π: S→A**.  
Value (expected discounted utility):  
\[
V^{\pi}(s)=
\begin{cases}
0, & \text{IsEnd}(s)\\
\sum_{s'} T(s,\pi(s),s')\,[R(s,\pi(s),s') + \gamma\,V^{\pi}(s')], & \text{else}
\end{cases}
\]
Q-value:
\[
Q^{\pi}(s,a)=\sum_{s'} T(s,a,s')\,[R(s,a,s') + \gamma\,V^{\pi}(s')].
\]

## Policy evaluation (iterative)
Initialize \(V^{(0)}\equiv 0\); iterate  
\[
V^{(t)}(s)\leftarrow \sum_{s'} T(s,\pi(s),s')\,[R(s,\pi(s),s')+\gamma V^{(t-1)}(s')].
\]
Stop when \(\max_s |V^{(t)}(s)-V^{(t-1)}(s)|\le \varepsilon\).  
Time \(O(t_{\text{PE}}\cdot |S|\cdot S')\) where \(S'=\#\{s':T>0\}\).

## Optimality & Value iteration
\[
Q^*(s,a)=\sum_{s'} T(s,a,s')[R(s,a,s')+\gamma V^*(s')],\quad
V^*(s)=\begin{cases}
0,&\text{IsEnd}(s)\\
\max_a Q^*(s,a),&\text{else.}
\end{cases}
\]
Iterate
\[
V^{(t)}(s)\leftarrow \max_{a}\sum_{s'} T(s,a,s')[R(s,a,s')+\gamma V^{(t-1)}(s')].
\]
Policy: \(\pi^*(s)=\arg\max_a Q^*(s,a)\).  
Time \(O(t_{\text{VI}}\cdot |S|\cdot |A|\cdot S')\).

## Convergence (sufficient)
If **γ<1** or the MDP graph is **acyclic**, policy evaluation and value iteration converge.  
Interpretation: with prob \(1-γ\) terminate each step (absorbing end).
