# Homework — SAME Problem Programmatically: Tree Search · DP · UCS

Implement the **TransportationProblem(n, unit_cost=False)** and the following algorithms:

1) **Backtracking search** (minimum-cost path; count node expansions).  
2) **DFS / BFS / DFID** under **unit_cost=True** (treat both actions as cost=1), compare nodes expanded vs solution depth \(d\).  
3) **Dynamic programming** (memoized) for acyclic graphs: compute FutureCost(1) and reconstruct an optimal path.  
4) **Uniform Cost Search (UCS)** for non-negative costs (default 1/2), return optimal path and cost.

**Report**: for \(n \in \{10, 50, 100, 500\}\)  
- optimal cost (DP and UCS must match), path length, nodes expanded (all methods), peak frontier size (BFS/DFID/UCS).  
- discuss when DFID beats BFS in space; when UCS outperforms BFS under non-equal costs.

**Grading (Base/Challenge)**: Correctness 60, Engineering 20, Analysis 20.  
**Challenge**: add **k-tram** \(s\to ks\) with cost \(c_k\); design an admissible heuristic \(h(s)\) and try **A\*** (optional).
