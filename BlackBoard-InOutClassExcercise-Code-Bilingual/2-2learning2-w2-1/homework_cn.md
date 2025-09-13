# 作业 — 同题的程序实现：SGD 与特征映射

**A 部分 — GD / SGD / 小批量**  
实现 `fit_linear(X,y, method, lr, epochs, batch_size, lr_schedule)`：
- `method ∈ {gd, sgd, minibatch}`；`lr_schedule ∈ {constant, sqrt_decay}`（$\eta_t=\eta_0/\sqrt{t}$）。
- 在 `data/regression_nonlinear.csv` 上，分别用 $\phi=[1,x]$ 与 $\phi_2=[1,x,x^2]$ 比较速度、达到目标损失的轮数、以及最终 MSE。

**B 部分 — 非线性特征**  
实现二次多项式、5 分箱、余弦特征；比较 MSE。

**C（可选）— 两层 ReLU 网络**  
在 `data/classification_xor.csv` 上训练小型两层网，达到 100% 训练精度。

**评分（基础/挑战）**：正确性 60，工程 20，分析 20。
