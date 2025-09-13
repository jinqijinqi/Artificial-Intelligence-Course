# Class — Same-problem hand steps (R1 & C1)

**R1 (Linear Regression, squared loss, GD 1 step)**  
Dataset $(x,y)\in\{(1,1),(2,3),(4,3)\}$, $\phi(x)=[1,x]$, start $w^{(0)}=[0,0]$, step $\eta=0.1$.  
1) Compute $\nabla \text{TrainLoss}(w^{(0)})$.  
2) Update $w^{(1)}=w^{(0)}-\eta\nabla \text{TrainLoss}(w^{(0)})$.  
3) (If time) Compute $\nabla\text{TrainLoss}(w^{(1)})$ and $w^{(2)}$.

**C1 (Linear Classification, hinge loss, one subgradient step)**  
Points $(x,y)\in\{([0,2],+1),([-2,0],+1),([1,-1],-1)\}$, $\phi(x)=[x_1,x_2]$, current $w=[0.5,1.0]$.  
1) For each example, compute hinge loss and subgradient.  
2) Average to get $\nabla$ (subgradient) of TrainLoss; discuss why some terms are 0.  
3) (Optional) One subgradient step with step $\eta=0.1$.
**Submission**: key equations + numeric results (2 decimals).
