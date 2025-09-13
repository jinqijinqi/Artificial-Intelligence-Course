# Homework — Implement the SAME problems programmatically

**Part R1 — Linear regression (squared loss + GD/SGD)**  
- Implement `fit_linear_gd(X, y, lr=0.1, epochs=200)` where $\phi(x)=[1,x]$ or general $\phi$.  
- Plot/print loss over epochs; report final $w$.  
**Base**: matches reference on `data/regression_toy.csv`.  
**Challenge**: add SGD (minibatch) and compare speed vs GD.

**Part C1 — Linear classification (hinge loss subgradient)**  
- Implement `fit_hinge_gd(X, y, lr=0.1, epochs=200)`; labels in $\{\pm1\}$.  
- Report train hinge loss and 0–1 accuracy.  
**Base**: matches reference on `data/classification_toy.csv`.  
**Challenge**: add L2 regularization and study margin distributions.

**Submission**: code + short report (≤1 page) with curves/tables.  
**Grading (Base/Challenge)**: Correctness 60, Engineering 20, Analysis 20.
