# Homework — SAME Problem Programmatically: Learning a Movie-Ratings BN

Implement a compact learner with **parameter sharing** for \(p_R(\cdot\mid g)\).

1) **MLE**: `fit_mle(supervised_data, share_R=True)` → CPTs `pG`, `pR`.  
2) **Laplace smoothing**: `fit_mle(..., lambda_=1.0)`; grid \(\lambda\in\{0,0.5,1,2\}\); report zero→positive flips.  
3) **EM**: `fit_em(mixed_data, init, lambda_=1.0, iters=1..10)` on mixed supervised+unsupervised examples; plot log-likelihood vs iters.  
4) **Posterior checks**: after EM, compute \(P(G\mid r_1,r_2)\) for the unsupervised pairs.  
5) **(Optional)** Naive Bayes extension: one-vs-rest word classification with parameter sharing for `p_word(·|y)` and Laplace smoothing.

Deliverables: code + ≤2-page note (tables: CPTs for MLE/EM, λ-sweep summary, likelihood curve).
