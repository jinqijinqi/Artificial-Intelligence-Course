# Board (EN) — Games I (Minimax · Alpha–Beta · Evaluation · Depth Limit · Quiescence · ID-DFS)

## Zero-sum game model
State s, player p(s)∈{MAX,MIN}, legal actions A(s), successor Succ(s,a), terminal utility U(s).

## Minimax recursion
\(V(s)=\begin{cases}
U(s), & s \text{ terminal}\\
\max_{a\in A(s)} V(\mathrm{Succ}(s,a)), & p(s)=\text{MAX}\\
\min_{a\in A(s)} V(\mathrm{Succ}(s,a)), & p(s)=\text{MIN}
\end{cases}\)

## Alpha–Beta pruning
Maintain bounds \([\alpha,\beta]\) along the path (best guaranteed score for MAX/MIN).
- MAX node: prune child if value ≥ β; update α ← max(α,child).
- MIN node: prune child if value ≤ α; update β ← min(β,child).
Invariant: true minimax value ∈ [α,β] ⇒ pruning preserves correctness.

## Complexity (branching b, depth d)
Minimax: \(O(b^d)\). With perfect move ordering, αβ: \(O(b^{d/2})\).

## Depth-limited search (cutoff at depth D)
Use evaluation \( \mathrm{Eval}(s)=w^\top \phi(s)\). Typical features: material/threats/mobility/center control.
**Quiescence**: extend search in tactically volatile nodes (e.g., forced wins/blocks) to reduce horizon effect.

## Iterative deepening + ordering
Run depth limits 1,2,…,D; use best line to order moves; transposition table caches (state→(depth,score,flag)).
