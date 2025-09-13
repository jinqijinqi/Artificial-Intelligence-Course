# Board (EN) — Machine Learning 2 (SGD · Step size · Non-linear Features · Two-layer NNs · Feature Templates)

## 1) Stochastic Gradient Descent (SGD)
Train loss: \( \displaystyle \mathrm{TrainLoss}(w)=\tfrac{1}{|D|}\sum_{(x,y)\in D}\mathrm{Loss}(x,y;w)\).
- **GD**: \(w \leftarrow w - \eta \nabla_w \mathrm{TrainLoss}(w)\) (one full pass/step).
- **SGD**: loop over examples, \(w \leftarrow w - \eta \nabla_w \mathrm{Loss}(x,y;w)\).
- **Minibatch** (size \(B\)): average gradients over \(B\) examples.

**Step size**: constant (e.g., \( \eta=0.1\)) or decreasing \( \eta_t=\eta_0/\sqrt{t}\). Larger \(\eta\) → faster but unstable; smaller \(\eta\) → stable but slow.

## 2) Non-linear features (linear in weights)
Predictor \(f_w(x)=w\cdot\phi(x)\) is linear in \(w\) and \(\phi(x)\), but \(\phi\) can be non-linear in \(x\):
- Quadratic: \( \phi(x)=[1, x, x^2]\).
- Piecewise/binning features: \( \phi_b(x)=[\mathbf{1}_{x\in \text{bin}_j}]_j\).
- Periodic: e.g., include \( \cos(\omega x)\).

## 3) Two-layer neural networks
Hidden representation \( h(x)=\sigma(V\,\phi(x)) \) (apply nonlinearity elementwise).  
Score \( s(x)=w\cdot h(x) \) → regression/classification.  
**ReLU** \( \sigma(z)=\max(z,0) \): avoids vanishing on \(z>0\), cheap to compute.

## 4) Feature templates & sparse/dense implementations
- Templates define *families* of features (e.g., “endsWith ___”, pixel intensities at (row,col,channel)).
- **Sparse dicts** for NLP-like templates; **dense arrays** for CV-like pixels. Store only nonzeros in sparse settings.
