# Board (EN) — Markov & Bayesian Networks I

## Factor graphs → weights
Variables \(X=(X_1,\dots,X_n)\); nonnegative factors \(f_1,\dots,f_m\).  
Assignment weight: \(\mathrm{Weight}(x)=\prod_{j=1}^m f_j(x)\).

## Markov network (MRF)
\[
P(X=x)=\frac{\mathrm{Weight}(x)}{Z},\quad Z=\sum_{x'} \mathrm{Weight}(x').
\]
**Marginal** \(P(X_i=v)=\sum_{x:\,x_i=v} P(X=x)\).  
Max-weight assignment need not match highest **marginal** (strength-in-numbers).

## Gibbs sampling (estimate marginals)
Iterate variables \(i=1..n\): sample \(x_i\sim P(X_i=\cdot\mid X_{-i}=x_{-i})\propto \mathrm{Weight}(x\cup\{X_i:\cdot\})\).  
Keep counts to estimate \( \hat P(X_i=v)\).

## Bayesian network (BN)
DAG over \(X_1,\dots,X_n\).  
\[
P(X_1=x_1,\dots,X_n=x_n)=\prod_{i=1}^n p(x_i\mid x_{\text{Parents}(i)}).
\]
BNs encode **local conditionals** (generative process).

## Explaining away (v-structure)
If two causes \(B,E\) positively influence effect \(A\):  
\[
P(B{=}1\mid A{=}1,E{=}1) < P(B{=}1\mid A{=}1).
\]
(Even when \(B\perp E\) a priori.)

— Examples: object tracking (obs & transition factors), Ising model, image denoising.
