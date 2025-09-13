# Instructor Key — Markov & Bayesian Networks I (3-step tracking + alarm)

**MRF exact (matches lecture table):**  
Enumerate non-zero assignments → \(Z=26\).  
Max-weight assignment: \((x_1,x_2,x_3)=(1,2,2)\) with weight 8.  
Marginals: \(P(X_2{=}1)=0.62,\ P(X_2{=}2)=0.38\).  
→ Shows “max-weight ≠ highest marginal.”  

**One Gibbs update (example):**  
With \(x_1{=}1,x_3{=}2\):  
\(\tilde w(x_2{=}0)=o_2(0)\,t(1,0)\,t(0,2)=0\)  
\(\tilde w(x_2{=}1)=o_2(1)\,t(1,1)\,t(1,2)=1\cdot 2\cdot 1=2\)  
\(\tilde w(x_2{=}2)=o_2(2)\,t(1,2)\,t(2,2)=2\cdot 1\cdot 2=4\)  
Normalize → \(P(X_2{=}1\mid\cdot)=1/3,\ P(X_2{=}2\mid\cdot)=2/3\).

**BN/HMM posterior:**  
Using row-normalized \(t,o\), a forward–backward pass yields a high posterior on \(H_2=1\) or \(2\) depending on evidence (see reference code output).  

**Explaining away (alarm BN):**  
\(P(B{=}1\mid A{=}1)=\frac{1}{2-\varepsilon},\quad P(B{=}1\mid A{=}1,E{=}1)=\varepsilon\) — decreases upon conditioning on \(E{=}1\).
