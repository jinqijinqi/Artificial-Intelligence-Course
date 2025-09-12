# UESTC Artificial Intelligence Course (2025-2026)
# Lecturer: Jin Qi
# You are welcome!
---
**Grading:**

Seminar (10%): Digits Recognition, Poetry GPT

In-Class Excercise (30%)

After-Class Homework (40%)

Projects (20%)



## **Report  Grading:**

* Mathematical formulation: clear, correct (20)
* Code: runs, documented, reproducible (20)
* Results & analysis: matches theory, well explained (20)
* Reflection: insightful, connects AI to real task (20)
* Formatting: neat, professional (20)

---

# Artificial Intelligence Assignment Report Example(Template) 

* **course：** Foundation of Artificial Intelligence
* **school：** Information and Communication Engineering(student's school)
* **Name:** Alice Smith
* **Student ID:** 2023123456
* **Assignment Title:** Week 8: Learning from Examples — kNN Fusion Method Selection
* **Submission Date:** September 27, 2024

---

## 1. Objective

The objective of this assignment is to understand and implement a simple supervised learning algorithm — k-Nearest Neighbors (kNN) — to select an image fusion method (average or max fusion) based on source image statistics. This demonstrates how basic AI learning techniques can guide decision-making in image fusion tasks.

---

## 2. Assignment Procedure / Steps

* Construct a small dataset where each data point consists of the mean values of an IR and VIS image patch, along with a label for the best fusion method (0: average, 1: max).
* Train a kNN classifier (k=1) using scikit-learn.
* Predict the fusion method for a new IR/VIS patch based on its statistics.

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 1. Construct training data: features are [IR_mean, VIS_mean], label is fusion method
X_train = np.array([[0.7, 0.4], [0.3, 0.8]])  # Example: two patches
y_train = np.array([0, 1])  # 0: average fusion, 1: max fusion

# 2. Train kNN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# 3. Predict for a new patch
X_test = np.array([[0.6, 0.5]])  # Example: new patch with IR mean 0.6, VIS mean 0.5
y_pred = knn.predict(X_test)
print(f"Predicted fusion method: {y_pred[0]} (0: average, 1: max)")
```

---

## 3. Algorithm and Mathematical Derivation

The kNN algorithm predicts the label for a new data point by finding the closest sample(s) in the feature space.

* **Distance formula** (Euclidean distance for two features):

  $$d = \sqrt{(IR_{mean}^{(test)} - IR_{mean}^{(i)})^2 + (VIS_{mean}^{(test)} - VIS_{mean}^{(i)})^2}$$
* **kNN prediction rule:**
  Let $X_{test}$ be the new patch feature.
  The predicted label is $y_{i^*}$, where

  $$i^* = \arg\min_i d(X_{test}, X_i)$$

  For $k=1$, we select the single closest training sample.

---

## 4. Experimental Results and Analysis

**Results:**

```
Predicted fusion method: 0 (0: average, 1: max)
```

* The classifier predicted “average fusion” for the test patch with IR mean 0.6 and VIS mean 0.5, since this patch is closer (in feature space) to the first training point (\[0.7, 0.4], label 0).

**Analysis:**

* The result is consistent with the mathematical expectation:

  * Distance to \[0.7, 0.4] = $sqrt((0.6-0.7)^2 + (0.5-0.4)^2) = sqrt(0.01 + 0.01) = sqrt(0.02) ≈ 0.14$
  * Distance to \[0.3, 0.8] = $sqrt((0.6-0.3)^2 + (0.5-0.8)^2) = sqrt(0.09 + 0.09) = sqrt(0.18) ≈ 0.42$
    So, the classifier correctly predicts 0 (average fusion).
* This demonstrates that even a simple kNN can automate fusion method selection when given appropriate features and labels.

---

## 5. Reflection and Insights

This assignment helped me connect supervised learning (as presented in AIMA Ch.18) with practical decision-making in image fusion. I learned how to structure a dataset, implement kNN using scikit-learn, and interpret the results both mathematically and programmatically.

The exercise also made me realize that, while kNN is very intuitive and easy to implement, its effectiveness depends heavily on good feature design and sufficient labeled examples. In real-world fusion, we would need a larger and more diverse dataset, possibly including more features (such as edge strength or contrast), and we might experiment with larger values of k or more sophisticated classifiers.

This approach could be extended to more complex fusion tasks, or even as a first step before using deep learning methods. It also showed me the importance of explainable, reproducible AI pipelines in scientific and engineering applications.

---
# Weekly homework(**Submission**: paper version of short report with curves/tables) (implemented by python) (due to the next class in next week)
## [Download empty report template here](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/homework1-week1-Pytorch%20Install.docx)<br/>

## Week 1. (2-1-learning1-w1-2)
**Part R1 — Linear regression (squared loss + GD/SGD)**  
- Implement `fit_linear_gd(X, y, lr=0.1, epochs=200)` where $\phi(x)=[1,x]$ or general $\phi$.  
- Plot/print loss over epochs; report final $w$.  
**Base**: matches reference on `data/regression_toy.csv`.  
**Challenge**: add SGD (minibatch) and compare speed vs GD.

**Reference codes:***
ref_regression.py
```python
import numpy as np

def add_bias(x):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x])

def fit_linear_gd(X, y, lr=0.1, epochs=200):
    X = np.asarray(X); y = np.asarray(y).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    hist = []
    for _ in range(epochs):
        pred = X @ w
        err = pred - y
        loss = (err**2).mean()
        grad = (2.0/n) * (X.T @ err)
        w -= lr * grad
        hist.append(loss)
    return w, np.array(hist)

if __name__ == "__main__":
    # Toy dataset from slides
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([1.0, 3.0, 3.0])
    X = add_bias(x)
    w, hist = fit_linear_gd(X, y, lr=0.1, epochs=200)
    print("w* =", w)
    print("final loss =", hist[-1])
```

**Part C1 — Linear classification (hinge loss subgradient)**  
- Implement `fit_hinge_gd(X, y, lr=0.1, epochs=200)`; labels in $\{\pm1\}$.  
- Report train hinge loss and 0–1 accuracy.  
**Base**: matches reference on `data/classification_toy.csv`.  
**Challenge**: add L2 regularization.

**Reference codes:***
ref_classification.py

```python
import numpy as np

def fit_hinge_gd(X, y, lr=0.1, epochs=200, l2=0.0):
    X = np.asarray(X); y = np.asarray(y).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    hist = []
    for _ in range(epochs):
        margins = (X @ w) * y
        # subgradient: average over samples
        mask = margins < 1.0
        grad = -(X[mask].T @ y[mask]) / n + l2 * w
        # hinge loss value
        loss = np.maximum(1 - margins, 0).mean() + 0.5*l2*np.dot(w,w)
        w -= lr * grad
        hist.append(loss)
    # 0-1 accuracy
    acc = (np.sign(X@w) == y).mean()
    return w, np.array(hist), acc

if __name__ == "__main__":
    # Toy points from slides
    X = np.array([[0.0, 2.0],
                  [-2.0, 0.0],
                  [1.0, -1.0]])
    y = np.array([+1, +1, -1])
    w, hist, acc = fit_hinge_gd(X, y, lr=0.1, epochs=50)
    print("w* =", w, "acc =", acc, "final hinge loss =", hist[-1])

```
## Week 2. (2-2learning2-w2-1)

**Part A — GD / SGD / Minibatch**  
Implement `fit_linear(X,y, method, lr, epochs, batch_size, lr_schedule)` with:
- `method ∈ {gd, sgd, minibatch}`; `lr_schedule ∈ {constant, sqrt_decay}` ($\eta_t=\eta_0/\sqrt{t}$).
- Compare speed, epochs-to-target-loss, and final MSE on [`regression_nonlinear.csv`](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/regression_nonlinear.csv) under $\phi=[1,x]$ vs $\phi_2=[1,x,x^2]$.

(Note: you need to load the data file `regression_nonlinear.csv`) 

```python
import time, math
import numpy as np

def add_bias_1d(x):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x])

def poly2_1d(x):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x, x**2])

def loss_mse(X, y, w):
    err = X @ w - y
    return float((err**2).mean())

def grad_mse_full(X, y, w):
    n = X.shape[0]
    return (2.0/n) * (X.T @ (X@w - y))

def fit_linear(X, y, method="sgd", lr=0.1, epochs=10, batch_size=32, lr_schedule="constant", seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X); y = np.asarray(y).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    t_updates = 0
    losses = []

    for ep in range(epochs):
        idx = np.arange(n); rng.shuffle(idx)
        if method == "gd":
            eta = lr / math.sqrt(max(1, t_updates)) if lr_schedule=="sqrt_decay" else lr
            g = grad_mse_full(X, y, w)
            w -= eta * g
            t_updates += 1
            losses.append(loss_mse(X, y, w))
        elif method == "sgd":
            for i in idx:
                xi, yi = X[i], y[i]
                eta = lr / math.sqrt(max(1, t_updates)) if lr_schedule=="sqrt_decay" else lr
                g = 2.0 * (xi @ w - yi) * xi
                w -= eta * g
                t_updates += 1
            losses.append(loss_mse(X, y, w))
        else:  # minibatch
            B = max(1, min(batch_size, n))
            for k in range(0, n, B):
                j = idx[k:k+B]
                Xb, yb = X[j], y[j]
                eta = lr / math.sqrt(max(1, t_updates)) if lr_schedule=="sqrt_decay" else lr
                g = (2.0/len(j)) * (Xb.T @ (Xb@w - yb))
                w -= eta * g
                t_updates += 1
            losses.append(loss_mse(X, y, w))

    return w, np.array(losses)
```

**Part B — Non-linear features**  
Implement polynomial degree-2, 5-bin piecewise, and cosine features; compare MSE.

(Note: you need to load the data file `regression_nonlinear.csv`) 

```python
import numpy as np

def phi_linear_1d(x):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x])

def phi_poly2_1d(x):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x, x**2])

def phi_bins_1d(x, B=5, lo=0.0, hi=5.0):
    x = np.asarray(x).reshape(-1)
    edges = np.linspace(lo, hi, B+1)
    X = np.zeros((len(x), B))
    for i, xi in enumerate(x):
        b = np.searchsorted(edges, xi, side="right") - 1
        b = min(max(b, 0), B-1)
        X[i, b] = 1.0
    return np.hstack([np.ones((len(x),1)), X])

def phi_periodic_1d(x, omega=3.0):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x, x**2, np.cos(omega*x)])
```
**Part C  — Two-layer NN**  
Train a tiny 2-layer ReLU net on [`classification_xor.csv`](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/classification_xor.csv) to 100% train accuracy.

(Note: you need to load the data file `classification_xor.csv`) 

```python
import numpy as np
def relu(z): return np.maximum(z, 0.0)
def d_relu(z): return (z > 0).astype(float)

class TinyTwoLayer:
    def __init__(self, d_in, d_h=4, lr=0.1, epochs=200, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(scale=0.5, size=(d_h, d_in)); self.b1 = np.zeros(d_h)
        self.W2 = rng.normal(scale=0.5, size=(1, d_h));    self.b2 = np.zeros(1)
        self.lr, self.epochs = lr, epochs
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).reshape(-1)  # {0,1}
        for _ in range(self.epochs):
            z1 = X @ self.W1.T + self.b1; h = relu(z1)
            z2 = h @ self.W2.T + self.b2
            yhat = 1/(1+np.exp(-z2)).reshape(-1)
            dz2 = (yhat - y)[:,None]
            gW2 = dz2.T @ h / len(y); gb2 = dz2.mean(0)
            dh  = dz2 @ self.W2; dz1 = dh * d_relu(z1)
            gW1 = dz1.T @ X / len(y); gb1 = dz1.mean(0)
            self.W2 -= self.lr*gW2; self.b2 -= self.lr*gb2
            self.W1 -= self.lr*gW1; self.b1 -= self.lr*gb1
        return self
    def predict(self, X):
        z1 = X @ self.W1.T + self.b1; h = relu(z1)
        z2 = h @ self.W2.T + self.b2
        yhat = 1/(1+np.exp(-z2)).reshape(-1)
        return (yhat >= 0.5).astype(int)
```



---

# All Projects (Project 1, 2, 3, 7 are necessarily required; Others are encouraged)

1. (Due Thursday of Week 2) Homework 1 -- PyTorch Installation (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/homework1-week1-Pytorch%20Install.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/get-started/locally/)<br/>

2. (Due Thursday of Week 3) Homework 2 -- Learning PyTorch with Examples (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/homework1-week1-Pytorch%20Install.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)<br/>

3. (Due Thursday of Week 4) Homework 3 -- What is torch.nn really? (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/homework1-week1-Pytorch%20Install.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/tutorials/beginner/nn_tutorial.html)<br/>

4. (Due Thursday of Week 6) Homework 4 -- Image Recognition (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/homework1-week1-Pytorch%20Install.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)<br/>

5. (Due Thursday of Week 6) Homework 5 -- Speech Recognition (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/homework1-week1-Pytorch%20Install.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html)<br/>

6. (Due Thursday of Week 6) Homework 6 -- Text Recognition (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/homework1-week1-Pytorch%20Install.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/tutorials/beginner/translation_transformer.html)<br/>

7. (Due Thursday of Week 7） Homework 7--Chinese Poem GPT (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx)<br/>
   [poemGPT](https://github.com/jinqijinqi/poemGPT/tree/main)<br/>

8. (Due Thursday of Week 8) Homework 7 -- Mario play agent (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/homework1-week1-Pytorch%20Install.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br/>
