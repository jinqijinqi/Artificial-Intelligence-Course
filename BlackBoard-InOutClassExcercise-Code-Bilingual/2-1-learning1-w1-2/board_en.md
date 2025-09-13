# Board (EN) — Machine Learning 1 (Linear Regression & Linear Classification)

## 1) Supervised learning & reflex predictors
- Empirical risk: \(\hat R(\theta)=\frac{1}{N}\sum_i \ell(f_\theta(x_i), y_i)\).
- Reflex model \(f\): fast feed-forward from \(x\) to \(y\). Tasks: binary/multiclass classification, regression, structured prediction.

## 2) Linear regression (squared loss)
- Feature extractor \(\phi(x)\); linear score \(f_w(x)=w\cdot\phi(x)\).
- Train loss (MSE): \(\displaystyle \text{TrainLoss}(w)=\frac{1}{|D|}\sum_{(x,y)}(w\cdot\phi(x)-y)^2\).
- Gradient: \(\displaystyle \nabla_w\text{TrainLoss}(w)=\frac{2}{|D|}\sum_{(x,y)}(w\cdot\phi(x)-y)\,\phi(x)\).
- Gradient descent: \(w\leftarrow w-\eta\,\nabla\text{TrainLoss}(w)\).

## 3) Linear classification (margins & losses)
- Predictor: \(f_w(x)=\mathrm{sign}(w\cdot\phi(x))\); **score** \(s=w\cdot\phi(x)\), **margin** \(m=s\,y\).
- Zero-one loss: \(\mathbb{1}[m\le 0]\) (non-optimizable by GD: gradient \(=0\) a.e.).
- Hinge loss: \(\ell_{\text{hinge}}(x,y,w)=\max\{1-m,0\}\), subgradient
  \(\displaystyle \partial_w \ell_{\text{hinge}}=\begin{cases}-\phi(x)y & m<1\\ 0 & m>1\end{cases}\).
- Logistic loss (digression): \(\ell_{\text{log}}=\log(1+e^{-m})\).

## 4) Key contrasts
- Regression: relate score to target via **residual** \(s-y\) → squared/absolute loss.
- Classification: relate score to target via **margin** \(s\cdot y\) → zero-one/hinge/logistic.
- Both trained by (stochastic) gradient methods.
