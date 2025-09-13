# Class — SAME Problem: SGD vs GD for Linear Regression (+ feature mapping micro-step)

**Dataset (hand steps)** `data/regression_hand.csv`: (1,1), (2,3), (4,3), with \( \phi(x)=[1,x]\), step \( \eta=0.1\).  
1) **GD one step** from \( w^{(0)}=[0,0]\): compute \( \nabla \mathrm{MSE}(w^{(0)}) \) and update \( w^{(1)}\).  
2) **SGD two updates** in order (1,1) → (2,3) from \( w^{(0)}\).  
3) **Quadratic feature mini-step**: \( \phi_2(x)=[1,x,x^2]\). On (1,1) from \( w=[0,0,0]\), do one SGD update with \( \eta=0.1\).

**Submit**: key equations + numbers (2 decimals).
