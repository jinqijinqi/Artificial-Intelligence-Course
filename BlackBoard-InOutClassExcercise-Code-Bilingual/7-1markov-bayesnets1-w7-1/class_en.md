# Class — SAME Problem: 3-Step 1D Tracking (as MRF & as BN/HMM)

**Domains** \(X_1,X_2,X_3\in\{0,1,2\}\). Observations \(o=(0,2,2)\).

**MRF factors (undirected)**  
Observation \(o_i(x_i)=\max(0,2-|x_i-o_i|)\) → table values in \(\{0,1,2\}\).  
Transition \(t_i(x_i,x_{i+1})=\begin{cases}2&x_i=x_{i+1}\\1&|x_i-x_{i+1}|=1\\0&\text{else}\end{cases}\).

**Tasks (MRF)**
1) **Exact \(Z\) & marginals**: enumerate all assignments with non-zero weight to get  
   \(Z=\sum_x \prod o_i(x_i)\,t_1(x_1,x_2)\,t_2(x_2,x_3)\). Then compute \(P(X_2=1)\), \(P(X_2=2)\).  
   Compare with the **max-weight assignment**.  
2) **One Gibbs update**: with \(x_1{=}1,x_3{=}2\), compute unnormalized weights for \(x_2\in\{0,1,2\}\)  
   via touching factors \(o_2,t_1,t_2\), normalize to get \(P(X_2=\cdot\mid X_1{=}1,X_3{=}2)\).

**BN view (HMM)**  
Directed chain \(H_1\to H_2\to H_3\), emissions \(H_i\to E_i\) with \(E_i=o_i\).  
Take \(p(H_{i+1}\mid H_i)\propto t_i\); \(p(E_i\mid H_i)\propto o_i\) (row-normalized).

3) **Posterior on middle state**: compute/derive \(P(H_2\mid E_1{=}0,E_2{=}2,E_3{=}2)\) qualitatively (or by simple enumeration/forward pass).  
4) (**Optional, explaining away mini-case**) Using the alarm network \(B,E\to A\): compare \(P(B{=}1\mid A{=}1)\) vs \(P(B{=}1\mid A{=}1,E{=}1)\).
