# Class — SAME Problem: Dice Game (Policy Evaluation & Value Iteration)

**Setup**  
State space \(S=\{\text{in}, \text{end}\}\). Actions at **in**: `stay` or `quit`; **end** has no actions.  
Transitions (γ is specified):  
- `quit`: \(T(\text{in},\text{quit},\text{end})=1\), reward \(R=10\).  
- `stay`: \(T(\text{in},\text{stay},\text{in})=\tfrac{2}{3}\), \(T(\text{in},\text{stay},\text{end})=\tfrac{1}{3}\), reward \(R=4\) on both branches.

## Tasks
1) **Closed-form** (γ=1): For policy π(stay), solve \(V^{\pi}(\text{in}) = \tfrac{1}{3}(4+0) + \tfrac{2}{3}(4+V^{\pi}(\text{in}))\).  
2) **Policy evaluation (iterative)**: with γ=1, initialize \(V^{(0)}\equiv0\) and compute \(V^{(t)}(\text{in})\) for t=1..5; report the max change \(\Delta_t\).  
3) **Value iteration**: initialize \(V^{(0)}\equiv0\); compute \(V^{(t)}\) and the greedy action \(\pi^{(t)}(\text{in})\) for t=1..5. When does policy switch from `quit` to `stay`?  
4) **Discounting**: set γ=0.5. Recompute (1)–(3). Which policy is optimal now?

**Submission**: your recurrence steps, the sequence \(V^{(t)}(\text{in})\), and the first t where \(\pi^{(t)}\) stabilizes.
