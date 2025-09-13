# 板书（中文）— 贝叶斯网络 II
**概率程序 ↔ 贝叶斯网；BN→MRF 做推断；Gibbs；前向–后向；粒子滤波**

## 概率程序（定义联合分布）
调用随机数生成器设定 \(X_1,\dots,X_n\)。程序生成赋值 \(x\) 的概率**即** \(P(X{=}x)\)。
- **告警**：\(B\sim\mathrm{Bern}(\varepsilon),\ E\sim\mathrm{Bern}(\varepsilon),\ A=B\lor E\)。
- **HMM**：\(H_i\!\sim p(H_i\!\mid H_{i-1}),\ E_i\!\sim p(E_i\!\mid H_i)\)。

## 贝叶斯网（BN）
DAG + 局部条件：\(P(x)=\prod_i p(x_i\mid x_{\mathrm{pa}(i)})\)。

## BN→MRF 做推断
把每个局部条件看作**因子**。  
无证据：\(P(x)=\prod_j f_j(x)\)，**\(Z{=}1\)**。  
有证据 \(E{=}e\)：将 \(e\) 代入，得到剩余变量上的因子；\(Z=P(E{=}e)\)。

### 推断前可做的图简化
- **去掉未观测叶子**（边缘化为 1）；  
- **丢弃与查询不连通的部分**；  
→ 然后用任意 MRF 推断（如 Gibbs）。

## Gibbs 采样（估计边缘）
对 \(i=1..n\) 轮替：按 \(P(X_i\mid X_{-i})\propto\) 与 \(X_i\) 相邻因子的乘积采样。

## HMM 的前向–后向
格点节点：\((i,h)\)。  
前向 \(F_i(h)=\sum_{h'}F_{i-1}(h')\,p(h\mid h')\,p(e_i\mid h)\)。  
后向 \(B_i(h)=\sum_{h'}B_{i+1}(h')\,p(h'\mid h)\,p(e_{i+1}\mid h')\)。  
平滑：\(P(H_i{=}h\mid E{=}e)=\frac{F_i(h)B_i(h)}{\sum_v F_i(v)B_i(v)}\)。  
时间 \(O(n|\mathcal{H}|^2)\)。

## 粒子滤波（大域上的过滤）
用 \(K\) 个**粒子**表示 \(P(H_i\mid e_{1:i})\)。每步：  
1) **提议** \(h_i\sim p(\cdot\mid h_{i-1})\)，2) **加权** \(w=p(e_i\mid h_i)\)，3) **重采样** \(\propto w\)。  
相较束搜索更保多样性；可扩展到大/连续状态。
