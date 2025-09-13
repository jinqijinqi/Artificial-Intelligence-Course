# Instructor Key — BN II (HMM & Alarm)

**1) BN→MRF simplification**  
- Conditioning on \(E=(0,2,2)\) turns \(p(E_i\mid H_i)\) into unary factors on \(H_i\).  
- For queries about \(H_2\), no unobserved leaves remain after plugging evidence; graph stays a chain.

**2) Forward–Backward (domain {0,1,2})**  
Use prior uniform; transitions/emissions: same=\(1/2\), neighbor=\(1/4\).  
Compute \(F_1\) from \(p(H_1)\,p(E_1\mid H_1)\); recurse to \(F_2,F_3\).  
Backward from \(B_3\equiv 1\) to get \(B_2,B_1\).  
Normalize \(S_2(h)=F_2(h)B_2(h)\) to obtain \(P(H_2\mid E)\).

**3) One Gibbs step for \(H_2\)**  
Sample from \(P(H_2\mid H_1{=}h_1,H_3{=}h_3,E_2{=}2)\propto p(h_2\mid h_1)\,p(2\mid h_2)\,p(h_3\mid h_2)\).

**4) Particle filtering (K=4 @ i=3)**  
Propose each \(h_3\sim p(\cdot\mid h_2)\); weight by \(p(e_3{=}2\mid h_3)\in\{0,\tfrac14,\tfrac12\}\);  
resample → particles cluster near \(h_3=2\) but keep diversity.

**5) Alarm program**  
\(B,E\sim \mathrm{Bern}(\varepsilon),\ A=B\lor E\). As a BN: \(p(b)p(e)p(a\mid b,e)\); as a program: three lines of pseudocode.
