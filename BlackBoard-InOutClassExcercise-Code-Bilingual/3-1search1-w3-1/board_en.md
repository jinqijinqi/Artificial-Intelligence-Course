# Board (EN) — Search 1 (Modeling · Tree Search · Dynamic Programming · UCS)

## Modeling a search problem
- **State** \(s\): summary of past sufficient for optimal future decisions.
- **Start** \(s_{\text{start}}\), **IsEnd**(s), **Actions**(s), **Succ**(s,a), **Cost**(s,a).
- **Objective**: minimum total cost along a start→end path.

## Tree search (branching b, depth upper bound D, solution depth d)
- **Backtracking** (any costs): time \(O(b^D)\), space \(O(D)\).
- **DFS** (assumption: all costs = 0): stop at first end; worst time \(O(b^D)\), space \(O(D)\).
- **BFS** (assumption: all costs equal \(c\ge 0\)): time/space \(O(b^d)\).
- **DFS-ID** (same assumption as BFS): time \(O(b^d)\), space \(O(d)\).

## Dynamic programming (acyclic state graph)
Recurrence for **future cost**:
\[
\mathrm{FutureCost}(s)=
\begin{cases}
0, & \text{if IsEnd}(s)\\[2mm]
\min_{a\in \mathrm{Actions}(s)}\ [\mathrm{Cost}(s,a)+\mathrm{FutureCost}(\mathrm{Succ}(s,a))],& \text{otherwise}
\end{cases}
\]
Memoize to avoid recomputation → exponential→polynomial saving when states are few.

## Uniform Cost Search (UCS) — non-negative costs
- Enumerate states in **increasing past cost** \( \mathrm{PastCost}(s)\).
- Maintain **Explored / Frontier / Unexplored**; pop min-priority from frontier.
- Correct when \( \mathrm{Cost}(s,a)\ge 0\); equivalent to Dijkstra (on implicit graphs).

## Complexity takeaway
- Time is exponential in depth in worst case; space can be reduced to linear with DFS-ID.
- Choose algorithm by cost assumptions + memory budget.

*(Aligned to lecture slides: applications, beyond reflex, tree search variants, DP, UCS.)*
