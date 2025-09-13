# Board (EN) — Search 2 (UCS Correctness · A* · Consistent/Admissible Heuristics · Relaxations)

## UCS correctness (graph search, non-negative costs)
Frontier/Explored invariant: when a state s is popped from the frontier → priority = PastCost(s) (min cost-to-s).  
Boundary-crossing argument + non-negativity ⇒ when end enters explored, its cost is optimal.

## A* as UCS with modified costs
Heuristic: \(h(s)\) ≈ FutureCost(s).  
Modified edge cost: \( \mathrm{Cost}'(s,a)=\mathrm{Cost}(s,a)+h(\mathrm{Succ}(s,a)) - h(s)\).  
Run **UCS** on \(\mathrm{Cost}'\) ⇒ equivalent to prioritizing \( \mathrm{PastCost}(s)+h(s)\).

### Consistency (needed for graph search optimality)
\(h\) is **consistent** iff  
1) \( \mathrm{Cost}'(s,a)\ge 0\) (triangle inequality) and 2) \(h(\text{end})=0\).

### Telescoping identity (correctness of A*)
For any path \(s_0\xrightarrow{a_1}\dots\xrightarrow{a_L}s_L\):  
\(\sum_i \mathrm{Cost}'(s_{i-1},a_i) = \sum_i \mathrm{Cost}(s_{i-1},a_i) + h(s_L)-h(s_0)\).  
Since \(h(\text{end})=0\) and \(h(s_0)\) is constant, minimizing modified path cost ≡ original.

### Efficiency of A*
A* explores all states with \( \mathrm{PastCost}(s) + h(s) \le \mathrm{PastCost}(\text{end})\). Larger \(h\) ⇒ fewer states (bounded by consistency).

## Relaxed heuristics (general framework)
Relax original problem \(P\) → \(P_{\text{rel}}\) s.t. \( \mathrm{Cost}_{\text{rel}}(s,a)\le \mathrm{Cost}(s,a)\).  
Define \( h(s)=\mathrm{FutureCost}_{\text{rel}}(s)\). Then \(h\) is **consistent**.  
Compute \(h\): closed-form (Manhattan), easier search, independent subproblems; combine via \( \max(h_1,h_2)\) (still consistent).

## DP vs UCS (review)
DP: acyclic; any costs; explores all \(N\) states.  
UCS: cycles; requires non-negative; explores only states cheaper than goal.
