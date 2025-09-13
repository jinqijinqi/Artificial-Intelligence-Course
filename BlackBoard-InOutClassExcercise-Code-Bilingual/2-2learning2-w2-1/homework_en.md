# Homework — SAME Problem Programmatically: SGD & Feature Mappings

**Part A — GD / SGD / Minibatch**  
Implement `fit_linear(X,y, method, lr, epochs, batch_size, lr_schedule)` with:
- `method ∈ {gd, sgd, minibatch}`; `lr_schedule ∈ {constant, sqrt_decay}` ($\eta_t=\eta_0/\sqrt{t}$).
- Compare speed, epochs-to-target-loss, and final MSE on `data/regression_nonlinear.csv` under $\phi=[1,x]$ vs $\phi_2=[1,x,x^2]$.

**Part B — Non-linear features**  
Implement polynomial degree-2, 5-bin piecewise, and cosine features; compare MSE.

**Part C (optional) — Two-layer NN**  
Train a tiny 2-layer ReLU net on `data/classification_xor.csv` to 100% train accuracy.

**Grading (Base/Challenge)**: Correctness 60, Engineering 20, Analysis 20.
