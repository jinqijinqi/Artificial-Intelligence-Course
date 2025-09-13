# Class — SAME Problem: Movie Ratings BN (G → R1, R2) with Missing G

**Variables & domains**
- Genre \(G\in\{\mathrm{c},\mathrm{d}\}\) (comedy/drama)
- Raters \(R_1,R_2\in\{1,2,3,4,5\}\)

**Data**
- **Supervised set** (fully observed): \((G,R_1,R_2)\in\{(d,4,5),(d,4,4),(d,5,3),(c,1,2),(c,5,4)\}\).
- **Unsupervised set** (missing \(G\)): \((?,2,2),(?,1,2)\).

**Tasks**
1) **MLE (count & normalize)** under **parameter sharing** \(p_R(\cdot\mid g)\) for both \(R_1,R_2\). Compute \(p_G(g)\) and \(p_R(r\mid g)\) from the supervised set.
2) **Laplace smoothing** with \(\lambda=1\): recompute \(p_R(r\mid g)\). Which entries change from 0 to \(>0\)?
3) **One EM iteration** using the two unsupervised examples:  
   - **E-step**: for each \((r_1,r_2)\), compute \(q_g \propto p_G(g)\,p_R(r_1\mid g)\,p_R(r_2\mid g)\); normalize.  
   - **M-step**: add fractional counts to \(p_G, p_R\) (optionally with \(\lambda\)). Report updated \(p_G\) and any changed \(p_R\) rows.
4) (**Optional**) Discuss how increasing \(\lambda\) changes posteriors and updates.

**Deliverable**: your tables for steps 1–3 (show arithmetic).
