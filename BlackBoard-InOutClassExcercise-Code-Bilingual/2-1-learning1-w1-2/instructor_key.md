# Instructor Key (worked answers)

## R1 — Regression GD steps (η=0.1), φ(x)=[1,x], D={(1,1),(2,3),(4,3)}
- At \(w^{(0)}=[0,0]\):
  \(\nabla \text{TrainLoss}(w^{(0)})=\frac{2}{3}\sum (w\cdot\phi(x)-y)\phi(x)=[-4.67,-12.67]\) (rounded).
  \(w^{(1)}=[0.47,\,1.27]\).
- At \(w^{(1)}\): \(\nabla \text{TrainLoss}(w^{(1)})=[2.18,\,7.24]\) → \(w^{(2)}=[0.25,\,0.54]\).
- Converges near \(w^*\approx[1.00,\,0.57]\).

## C1 — Hinge loss at w=[0.5,1.0]
- For ([0,2],+1): margin \(=2\Rightarrow\) loss 0, grad \([0,0]\).
- For ([-2,0],+1): margin \(=-1\Rightarrow\) loss 2, grad \([2,0]\).
- For ([1,-1],-1): margin \(=0.5\Rightarrow\) loss 0.5, grad \([1,-1]\).
- Average (sub)gradient \(\approx[1.00,\,-0.33]\). Average loss \(\approx0.83\).
