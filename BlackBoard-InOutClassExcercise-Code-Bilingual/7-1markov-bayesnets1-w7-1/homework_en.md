# Homework — SAME Problem Programmatically: 3-Step Tracking as MRF & BN

**Part A — MRF (exact + Gibbs)**
1) Build factors \(o_i,t_i\). Enumerate all assignments, compute \(Z\) and marginals \(P(X_i)\).  
2) Implement **Gibbs sampling** (systematic scan). Track counts of \(X_2\) after burn-in; compare to exact \(P(X_2)\).  
   Plot iterations vs \(\ell_\infty\) error of marginals.

**Part B — BN/HMM (exact)**
3) Construct a BN with chain \(H_1\to H_2\to H_3\), emissions \(H_i\to E_i\) (row-normalize from \(t_i,o_i\)).  
4) Implement exact inference for \(P(H_2\mid E_1{=}0,E_2{=}2,E_3{=}2)\) by enumeration or forward–backward. Compare to MRF \(P(X_2)\).

**(Optional) Explaining away**
5) Implement the **alarm BN** with \(p(b){=}\varepsilon,\ p(e){=}\varepsilon,\ p(a\mid b,e)=[a=b\lor e]\).  
   Show \(P(B{=}1\mid A{=}1)=\frac{1}{2-\varepsilon}\), \(P(B{=}1\mid A{=}1,E{=}1)=\varepsilon\).

**Deliverables**: code + 1–2 page note (tables for exact vs Gibbs; BN posterior; plots).
