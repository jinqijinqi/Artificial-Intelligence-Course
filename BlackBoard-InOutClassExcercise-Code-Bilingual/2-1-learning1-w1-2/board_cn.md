# 板书（中文）— 机器学习 1（线性回归与线性分类）

## 1) 监督学习与“反射式”模型
- 经验风险：\(\hat R(\theta)=\frac{1}{N}\sum_i \ell(f_\theta(x_i), y_i)\)。
- 反射式预测器 \(f\)：从 \(x\) 到 \(y\) 的快速前馈。任务类型：二/多分类、回归、结构化预测。

## 2) 线性回归（平方损失）
- 特征映射 \(\phi(x)\)；线性打分 \(f_w(x)=w\cdot\phi(x)\)。
- 训练损失：\(\displaystyle \text{TrainLoss}(w)=\frac{1}{|D|}\sum_{(x,y)}(w\cdot\phi(x)-y)^2\)。
- 梯度：\(\displaystyle \nabla_w\text{TrainLoss}(w)=\frac{2}{|D|}\sum_{(x,y)}(w\cdot\phi(x)-y)\,\phi(x)\)。
- 梯度下降：\(w\leftarrow w-\eta\,\nabla\text{TrainLoss}(w)\)。

## 3) 线性分类（边际与损失）
- 分类器：\(f_w(x)=\mathrm{sign}(w\cdot\phi(x))\)；**分数** \(s=w\cdot\phi(x)\)，**边际** \(m=s\,y\)。
- 0–1 损失：\(\mathbb{1}[m\le 0]\)（几乎处处梯度为 0，难以用 GD 直接优化）。
- 合页损失：\(\ell_{\text{hinge}}(x,y,w)=\max\{1-m,0\}\)，次梯度
  \(\displaystyle \partial_w \ell_{\text{hinge}}=\begin{cases}-\phi(x)y & m<1\\ 0 & m>1\end{cases}\)。
- 逻辑损失（旁注）：\(\ell_{\text{log}}=\log(1+e^{-m})\)。

## 4) 对照要点
- 回归用**残差** \(s-y\)；分类用**边际** \(s\cdot y\)。
- 常见损失：平方/绝对 vs. 0–1/合页/逻辑；均可用（随机）梯度法训练。
