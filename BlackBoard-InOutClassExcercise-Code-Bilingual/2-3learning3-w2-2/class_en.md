# Class — Same Problem, Two Tracks

## Track A — Backprop by hand (squared loss)
Given \(w=[3,1]\), \(\phi(x)=[1,2]\), \(y=2\).
1) Forward: score, residual, loss.  
2) Backward: compute \(\nabla_w L\) via the computation graph.  *(Answer: \([6,12]\)).*

## Track B — One K-means iteration
Points: (0,0),(0,3),(3,0),(3,3), plus jitter points. \(K=2\), \(\mu_1=(0,0)\), \(\mu_2=(3,3)\).
1) Assign by Euclidean distance. 2) Update means. 3) Compare objective before/after.

**Submission**: key equations + numeric results (2 decimals).
