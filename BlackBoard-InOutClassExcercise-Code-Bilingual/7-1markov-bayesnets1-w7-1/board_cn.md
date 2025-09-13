# 板书（中文）— 马尔可夫网与贝叶斯网 I

## 因子图 → 权重
变量 \(X=(X_1,\dots,X_n)\)；非负因子 \(f_1,\dots,f_m\)。  
赋值权重：\(\mathrm{Weight}(x)=\prod_{j=1}^m f_j(x)\)。

## 马尔可夫网（MRF）
\[
P(X=x)=\frac{\mathrm{Weight}(x)}{Z},\quad Z=\sum_{x'} \mathrm{Weight}(x').
\]
**边缘概率** \(P(X_i=v)=\sum_{x:\,x_i=v} P(X=x)\)。  
最优权重赋值未必对应最高**边缘**（“数量优势”）。

## Gibbs 采样（估计边缘）
按变量轮替：按 \(P(X_i=\cdot\mid X_{-i})\propto \mathrm{Weight}(x\cup\{X_i:\cdot\})\) 采样更新 \(x_i\)。  
累计计数估计 \( \hat P(X_i=v)\)。

## 贝叶斯网（BN）
有向无环图（DAG）：  
\[
P(X_1=x_1,\dots,X_n=x_n)=\prod_{i=1}^n p(x_i\mid x_{\text{Parents}(i)}).
\]
BN 用**局部条件分布**表达生成过程。

## 解释性排斥（v 结构）
两因 \(B,E\) 正向影响效应 \(A\)：  
\[
P(B{=}1\mid A{=}1,E{=}1) < P(B{=}1\mid A{=}1).
\]
（即便先验 \(B\perp E\)。）

— 示例：目标跟踪（观测/转移因子）、Ising 模型、图像去噪。
