# Homework — SAME Problem Programmatically: BN II (Gibbs · F–B · Particle Filter)

**Part A — Prob. programming**
- Implement `sample_alarm(eps)` and an HMM sampler `sample_hmm(T)`.

**Part B — BN→MRF + Gibbs**
- Build the reduced MRF for the toy medical BN (C,A,H,I) under evidence \(H{=}1,I{=}1\); implement Gibbs to estimate \(P(C{=}1\mid H{=}1,I{=}1)\).

**Part C — HMM**
- Implement **forward_backward(evidence)** (return marginals for all \(H_i\));  
- Implement **particle_filter(evidence,K)** with propose–weight–resample; track counts only for the last \(H_i\).  
- Compare filtering posteriors at \(i=3\) for \(K\in\{50,200,1000\}\) vs exact smoothing \(P(H_3\mid E)\); report \(\ell_1\) error and runtime.  
- (Optional) Add **beam search** baseline (K same as particles) and discuss diversity vs accuracy.

**Deliverables**: code + ≤2-page note (tables: posterior & error; brief discussion).
