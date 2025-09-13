# 作业 — 同题程序实现：BN II（Gibbs · 前向–后向 · 粒子滤波）

**A 部分 — 概率程序**
- 实现 `sample_alarm(eps)` 与 HMM 采样器 `sample_hmm(T)`。

**B 部分 — BN→MRF + Gibbs**
- 在玩具医疗 BN（C,A,H,I）上结合证据 \(H{=}1,I{=}1\) 构建 MRF，写 Gibbs 估计 \(P(C{=}1\mid H{=}1,I{=}1)\)。

**C 部分 — HMM**
- 实现 **forward_backward(evidence)**（返回各时刻 \(H_i\) 的边缘）；  
- 实现 **particle_filter(evidence,K)**（提议–加权–重采样），仅保存最后 \(H_i\) 的计数；  
- 在 \(i=3\) 将过滤后验与精确平滑 \(P(H_3\mid E)\) 对比，测试 \(K\in\{50,200,1000\}\)，报告 \(\ell_1\) 误差与时间；  
- （选做）加入 **束搜索** 基线（K 同粒子），讨论多样性与精度。

**提交**：代码 + ≤2 页说明（后验/误差表与简评）。
