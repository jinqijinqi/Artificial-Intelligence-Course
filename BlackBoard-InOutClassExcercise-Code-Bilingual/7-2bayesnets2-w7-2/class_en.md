# Class — SAME Problem: HMM (3 steps) + Prob. Programs + BN→MRF
State domain \(\{0,1,2\}\). Prior \(p(H_1)=\mathrm{Unif}\). Transition \(p(h_i\mid h_{i-1})=\frac12[\!h_i{=}h_{i-1}\!]+\frac14[\!|h_i-h_{i-1}|=1\!]\).  
Emission \(p(e_i\mid h_i)=\frac12[\!e_i{=}h_i\!]+\frac14[\!|e_i-h_i|=1\!]\). Evidence \((e_1,e_2,e_3)=(0,2,2)\).

**Tasks**
1) **Prob. program**: write pseudocode that samples \(H_{1:3},E_{1:3}\); and the **alarm** program \(B,E\sim\mathrm{Bern}(\varepsilon), A=B\lor E\).  
2) **BN→MRF** with evidence: plug in \(E\!=\!e\), then **remove unobserved leaves** and **discard disconnected components** for the query \(P(H_2\mid E)\).  
3) **Forward–Backward**: compute \(F_1,F_2,F_3\) and \(B_3,B_2,B_1\), then \(P(H_2\mid E)\). Show your arithmetic (fractions are fine).  
4) **One Gibbs update** on the reduced MRF for \(H_2\) given neighbors.  
5) **Particle filtering (K=4)**: show one full step at \(i=3\): propose from each \(h_2\), weight by \(p(e_3\mid h_3)\), resample; report particle counts.
