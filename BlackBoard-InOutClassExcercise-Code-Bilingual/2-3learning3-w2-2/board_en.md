# Board (EN) — Machine Learning 3 (Backpropagation · K-means · Generalization · Best practices)

## Backpropagation via computation graphs
Loss (two-layer net, squared): \(L=(w\cdot\sigma(V\phi(x)) - y)^2\).
Nodes carry forward values \(f_i\); backward values \(g_i=\partial L/\partial f_i\).
Local derivatives: `+`→1, `-`→(1,−1), `·`→(b,a), `(\cdot)^2`→\(2a\), `max`→ indicators, `σ`→\(σ(1-σ)\).
Two-layer grads: \( \nabla_w L=2\,\text{res}\,h,\ \nabla_V L=2\,\text{res}\,(w\circ h\circ(1-h))\phi^\top\),
with \(h=\sigma(V\phi(x))\), \(\text{res}=w\cdot h - y\).

## K-means (unsupervised)
Objective: \( \sum_i \|\phi(x_i)-\mu_{z_i}\|^2 \).
Alternate: Assign \(z_i\gets\arg\min_k\|\phi(x_i)-\mu_k\|^2\); Update \(\mu_k\gets \text{mean}\{\phi(x_i):z_i=k\}\).
Local minima ⇒ restarts / kmeans++.

## Generalization & regularization
Train/Val/Test; pick hyperparams on Val only.
Approximation vs Estimation error; L2 (weight decay) \(J(w)+\frac{\lambda}{2}\|w\|^2\),
update \(w\gets w-\eta(\nabla J+\lambda w)\); early stopping reduces effective norm.

## Best practices
Start simple; can you overfit 5 examples? multiple seeds; error bars / subgroup metrics; log everything.
