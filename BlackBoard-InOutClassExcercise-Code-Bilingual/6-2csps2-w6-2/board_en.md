# Board (EN) — CSPs 2 (Backtracking · MCV/MRV+LCV · Forward Checking · AC‑3 · Beam · ICM)

## Partial-assignment weight
For factor graph with factors \(f_j \ge 0\), assignment weight: \(\mathrm{Weight}(x)=\prod_j f_j(x)\).
For partial x, multiply **dependent factors** \(D(x,X_i)\): those touching \(X_i\) and only already-assigned vars.

## Backtracking search (exact)
Pick unassigned **variable** (MCV/MRV), **order values** (LCV), compute \(\delta=\prod_{f\in D(x,X_i)} f(x\cup\{X_i\!:\!v\})\).  
If \(\delta=0\) ⇒ prune. Recurse with updated domains via **lookahead**.

**MCV/MRV**: choose smallest current domain. **LCV**: prefer value that leaves most neighbor options.

## Forward checking (lookahead, 1‑step)
After setting \(X_j\!\!=\!v\), delete from each neighbor \(X_i\) any value \(u\) with some factor forbidding \((u,v)\).

## Arc consistency & AC‑3
**Arc consistent**: each value \(x_i\) of \(X_i\) has some supporting \(x_j\) in \(X_j\) on all \(f\) that involve \((X_i,X_j)\).  
**AC‑3**: repeatedly enforce arc consistency on neighbors whose domains changed; worst‑case \(O(ED^3)\).

## Approximate inference
**Beam search (K)**: keep top‑K partial assignments at each depth. Time ~ \(O(nKb)\).  
**Local search (ICM)**: start with full assignment, update one variable to maximize local product of touching factors; monotone ↑, may get stuck.
