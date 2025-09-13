# Instructor Key — Movie Ratings BN (Learning)

**Step 1 — MLE (supervised, sharing \(p_R\))**  
- From the 5 labeled triples, \(p_G(d)=3/5,\ p_G(c)=2/5\).  
- For \(g=d\): counts on \(R\) from \((4,5),(4,4),(5,3)\) across \(R_1,R_2\) → \(r\in\{3:1,4:3,5:2\}\) ⇒ \(p_R(\cdot\mid d)=(1,0,1,3,2)/7\).  
- For \(g=c\): from \((1,2),(5,4)\) → \(r\in\{1:1,2:1,4:1,5:1\}\) ⇒ \(p_R(\cdot\mid c)=(1,1,0,1,1)/4\).

**Step 2 — Laplace smoothing (\(\lambda=1\))**  
- Add 1 to each rating bin per \(g\). Zeros become positive; e.g., \(p_R(3\mid c)=1/9>0\).

**Step 3 — One EM iter (unlabeled \((2,2)\),(1,2))**  
- **E**: \(q_g \propto p_G(g)\,p_R(r_1\mid g)\,p_R(r_2\mid g)\) using smoothed tables; normalize to get posteriors over \(G\).  
- **M**: add fractional counts to \(p_G\) and \(p_R\); probabilities typically sharpen towards the class explaining \((2,2)\).

**Notes**  
- Sharing \(p_R\) across \(R_1,R_2\) increases effective data and stabilizes estimates.  
- Larger \(\lambda\) pulls CPT rows toward uniform; as data grows, effect diminishes.
