# 板书（中文）— 机器学习 3（反向传播 · K-means · 泛化 · 实践要点）

## 反向传播（计算图）
两层网平方损失 \(L=(w\cdot\sigma(V\phi(x)) - y)^2\)。
前向值 \(f_i\)；反向值 \(g_i=\partial L/\partial f_i\)。
局部导数：`+`→1，`-`→(1,−1)，乘法→(b,a)，平方→\(2a\)，`max`→指示，`σ`→\(σ(1-σ)\)。
两层网梯度：\( \nabla_w L=2\,\text{res}\,h,\ \nabla_V L=2\,\text{res}\,(w\circ h\circ(1-h))\phi^\top\)。

## K-means（无监督）
目标：\( \sum_i \|\phi(x_i)-\mu_{z_i}\|^2 \)。
交替最小化：分配 \(z_i\)；更新 \(\mu_k\) 为簇均值。局部最优 ⇒ 多次重启 / kmeans++。

## 泛化与正则化
Train/Val/Test；仅在 Val 上调参。近似误差 vs 估计误差；
L2（权值衰减）\(J(w)+\frac{\lambda}{2}\|w\|^2\)，更新 \(w\gets w-\eta(\nabla J+\lambda w)\)；
早停降低有效范数。

## 实践要点
从小做起（能否过拟合 5 个样本）；多随机种子；误差条与子群体指标；完整日志。
