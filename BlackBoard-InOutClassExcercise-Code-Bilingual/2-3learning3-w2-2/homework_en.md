# Homework — SAME Problems (Implementation)

**Part A — Two-layer network backprop (squared loss)**  
Implement `forward`/`backward` for \(h=\sigma(V\phi(x))\), \(s=w\cdot h\), \(L=(s-y)^2\); check with finite diff.

**Part B — K-means with restarts/k++**  
Implement `kmeans(X,K,init='random'|'k++',restarts=10,max_iter=100)`; compare losses across seeds.

**Mini Part C — Validation for L2 & early stopping**  
Split train/val; grid-search \(\lambda\in\{0,10^{-3},10^{-2},10^{-1}\}\); add early stopping; report best \(\lambda\) & MSE.

**Grading**: Correctness 60, Engineering 20, Analysis 20.
