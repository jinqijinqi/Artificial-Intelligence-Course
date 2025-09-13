# 板书（中文）— 贝叶斯网络 III（学习：极大似然 · 平滑 · EM）

## 监督学习（完全观测）
对 BN 的局部条件 \(p(x_i \mid x_{\mathrm{pa}(i)})\)：给定完整样本 \(x\in D_{\text{train}}\)，  
**计数 + 归一化（MLE）**：
\[
\hat p(x_i \mid x_{\mathrm{pa}(i)}) = \frac{\mathrm{count}(x_{\mathrm{pa}(i)}, x_i)}{\sum_{x'_i}\mathrm{count}(x_{\mathrm{pa}(i)}, x'_i)}。
\]
**参数共享**：多节点可共用同一张表 \(p_d\)；学习阶段向**同一**计数表写入。

## 平滑（拉普拉斯/伪计数）
为避免 0 概率，对每个本地取值都加 \(\lambda>0\)：
\[
\hat p_\lambda(x_i \mid x_{\mathrm{pa}(i)}) = \frac{\lambda + \mathrm{count}(x_{\mathrm{pa}(i)}, x_i)}{\sum_{x'_i} (\lambda + \mathrm{count}(x_{\mathrm{pa}(i)}, x'_i))}.
\]
\(\lambda\) 越大越靠近均匀；数据量增大时平滑影响减弱。

## EM（最大边缘似然）
隐变量 \(H\)，可见 \(E\)。目标：
\[
\max_\theta \prod_{e} P(E=e;\theta) = \max_\theta \prod_e \sum_h P(H=h,E=e;\theta)。
\]

**E 步**：\(q(h)\leftarrow P(H=h\mid E=e;\theta)\)（枚举/前向–后向等）。  
**M 步**：把 \((h,e)\) 视为带权样本，做**计数+归一化**（可加 \(\lambda\)）。

—— 例：一/二/三节点 BN、朴素贝叶斯、HMM（起始/转移/发射的共享）。
