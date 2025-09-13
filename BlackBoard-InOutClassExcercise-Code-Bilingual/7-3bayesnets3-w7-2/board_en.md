# Board (EN) — Bayesian Networks III (Learning: MLE · Smoothing · EM)

## Supervised learning in BNs (fully observed)
BN with local CPTs \(p(x_i \mid x_{\mathrm{pa}(i)})\). With full assignments \(x\in D_{\text{train}}\):  
**Count & normalize (MLE)** for each local table and parent setting:
\[
\hat p(x_i \mid x_{\mathrm{pa}(i)}) = \frac{\mathrm{count}(x_{\mathrm{pa}(i)}, x_i)}{\sum_{x'_i}\mathrm{count}(x_{\mathrm{pa}(i)}, x'_i)}.
\]

**Parameter sharing**: multiple nodes can reuse one table \(p_d\). Learning writes to the **same** counts.

## Smoothing (Laplace / pseudocounts)
Avoid zeros by adding \(\lambda>0\) to **every** local outcome:
\[
\hat p_\lambda(x_i \mid x_{\mathrm{pa}(i)}) = \frac{\lambda + \mathrm{count}(x_{\mathrm{pa}(i)}, x_i)}{\sum_{x'_i} (\lambda + \mathrm{count}(x_{\mathrm{pa}(i)}, x'_i))}.
\]
Larger \(\lambda\) → closer to uniform; data dominates as \(|D_{\text{train}}|\to\infty\).

## EM for missing variables (maximum marginal likelihood)
Hidden \(H\), observed \(E\). Objective:
\[
\max_\theta \prod_{e\in D_{\text{train}}} P(E=e;\theta) = \max_\theta \prod_e \sum_h P(H=h,E=e;\theta).
\]

**E-step**: \(q(h) \leftarrow P(H=h\mid E=e;\theta)\) via inference (enumeration / F–B for HMM).  
**M-step**: treat \((h,e)\) with weight \(q(h)\) as fractional data → **count & normalize** (with optional \(\lambda\)).

— Examples: one/two/three-node BNs, Naive Bayes, HMM (start/trans/emit with sharing).
