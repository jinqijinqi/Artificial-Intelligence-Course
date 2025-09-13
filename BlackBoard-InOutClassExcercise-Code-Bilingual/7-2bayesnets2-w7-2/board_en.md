# Board (EN) — Bayesian Networks II
**Probabilistic programs ↔ Bayesian networks; BN→MRF for inference; Gibbs; Forward–Backward; Particle Filtering**

## Probabilistic program (defines a joint)
Randomized program that sets variables \(X_1,\dots,X_n\). The probability of generating assignment \(x\) **is** \(P(X{=}x)\).
- **Alarm**: \(B\sim\mathrm{Bern}(\varepsilon),\ E\sim\mathrm{Bern}(\varepsilon),\ A=B\lor E\).
- **HMM**: \(H_{i}\!\sim p(H_i\mid H_{i-1}),\ E_i\!\sim p(E_i\mid H_i)\).

## Bayesian network (BN)
DAG with local CPTs: \(P(x)=\prod_i p(x_i\mid x_{\mathrm{pa}(i)})\).

## Reduce BN→MRF for inference
Treat each local CPT as a **factor** (nonnegative).  
Without evidence: \(P(x)=\prod_j f_j(x)\) with **\(Z{=}1\)**.  
With evidence \(E{=}e\): plug in \(e\) to get factors on remaining vars; \(Z=P(E{=}e)\).

### Graph simplifications before inference
- **Remove unobserved leaves** (marginalizes to 1).  
- **Discard disconnected components** (w.r.t. query).  
→ Run any MRF inference (e.g., Gibbs).

## Gibbs sampling (estimate marginals)
Repeat for \(i=1..n\): sample \(X_i\sim P(\cdot\mid X_{-i})\propto\) product of **touching** factors.

## HMM smoothing via Forward–Backward
Lattice nodes: \((i,h)\).  
Forward \(F_i(h)=\sum_{h'}F_{i-1}(h')\cdot p(h\mid h')\,p(e_i\mid h)\).  
Backward \(B_i(h)=\sum_{h'}B_{i+1}(h')\cdot p(h'\mid h)\,p(e_{i+1}\mid h')\).  
Smoothing: \(P(H_i{=}h\mid E{=}e)=\frac{F_i(h)B_i(h)}{\sum_v F_i(v)B_i(v)}\).  
Time \(O(n|\mathcal{H}|^2)\).

## Particle filtering (filtering in large domains)
Represent \(P(H_i\mid e_{1:i})\) by \(K\) **particles**. For each time step:
1) **Propose** \(h_i\sim p(\cdot\mid h_{i-1})\) for each particle,  
2) **Weight** by \(w=p(e_i\mid h_i)\),  
3) **Resample** \(K\) particles \(\propto w\).  
Keeps diversity vs beam search; scales to large/continuous \(\mathcal{H}\).
