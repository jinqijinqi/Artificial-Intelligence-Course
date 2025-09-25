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
## [Download empty report template here](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/homework-Template.docx)<br/>

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

## Week 2-1. (2-2learning2-w2-1)

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

## Week 2-2. (2-3learning3-w2-2)

# In Class — Same Problem, Two Tracks

## Track A — Backprop by hand (squared loss)
Given $w=[3,1]$, $\phi(x)=[1,2]$, $y=2$.
1) Forward: score, residual, loss.  
2) Backward: compute $\nabla_w L$ via the computation graph.  *(Answer: $[6,12]$).*

## Track B — One K-means iteration
Points: (0,0),(0,3),(3,0),(3,3), plus jitter points. $K=2$, $\mu_1=(0,0)$, $\mu_2=(3,3)$.
1) Assign by Euclidean distance. 2) Update means. 3) Compare objective before/after.

**Submission**: key equations + numeric results (2 decimals).

# Out Class Homework — SAME Problems (Implementation)

**Part A — Two-layer network backprop (squared loss)**  
Implement `forward`/`backward` for $h=\sigma(V\phi(x))$, $s=w\cdot h$, $L=(s-y)^2$; check with finite diff.

**Part B — K-means with restarts/k++**  
Implement `kmeans(X,K,init='random'|'k++',restarts=10,max_iter=100)`; compare losses across seeds.

**Mini Part C — Validation for L2 & early stopping**  
Split train/val; grid-search $\lambda\in$ { $0,10^{-3},10^{-2},10^{-1}$ }; add early stopping; report best $\lambda$ & MSE.

## reference codes
ref_backprop_two_layer.py

```python

import numpy as np
def sigmoid(z): return 1/(1+np.exp(-z))
def forward(phi, V, w, y):
    z1 = V @ phi; h = sigmoid(z1); s = float(w @ h)
    residual = s - float(y); loss = residual**2
    return (phi, V, w, y, z1, h, s, residual, loss)
def backward(cache):
    phi, V, w, y, z1, h, s, residual, loss = cache
    grad_w = 2.0 * residual * h
    g = 2.0 * residual * (w * h * (1-h))
    grad_V = np.outer(g, phi)
    return grad_w, grad_V, loss
```

ref_kmeans.py

```python

import numpy as np
def kmeans_pp_init(X, K, rng):
    n = X.shape[0]; centers = np.empty((K, X.shape[1]))
    i0 = rng.integers(n); centers[0] = X[i0]; d2 = np.full(n, np.inf)
    for k in range(1, K):
        d2 = np.minimum(d2, ((X - centers[k-1])**2).sum(1))
        probs = d2 / d2.sum(); i = rng.choice(n, p=probs); centers[k] = X[i]
    return centers
def kmeans(X, K, init='random', restarts=10, max_iter=100, seed=0):
    rng = np.random.default_rng(seed); best = None
    for _ in range(restarts):
        if init=='k++': centers = kmeans_pp_init(X, K, rng)
        else:
            lo, hi = X.min(0), X.max(0); centers = rng.uniform(lo, hi, size=(K, X.shape[1]))
        for _ in range(max_iter):
            d2 = ((X[:,None,:]-centers[None,:,:])**2).sum(-1)
            assign = d2.argmin(1)
            new = centers.copy()
            for k in range(K):
                idx = np.where(assign==k)[0]
                if len(idx)>0: new[k] = X[idx].mean(0)
            if np.allclose(new, centers): centers = new; break
            centers = new
        loss = ((X - centers[assign])**2).sum()
        if best is None or loss < best[-1]: best = (assign, centers, float(loss))
    return best
```
ref_validation_regularization.py

```python

import numpy as np
def add_bias_1d(x):
    x = np.asarray(x).reshape(-1,1); return np.hstack([np.ones_like(x), x])
def fit_ridge(X, y, lam=0.0, lr=0.1, epochs=300, early_stop=False, patience=20, X_val=None, y_val=None):
    X = np.asarray(X); y = np.asarray(y).reshape(-1)
    n, d = X.shape; w = np.zeros(d); best = (np.inf, w.copy()); wait = 0
    def mse(A,b,w): e=A@w-b; return float((e**2).mean())
    for ep in range(epochs):
        grad = (2.0/n) * (X.T @ (X@w - y)) + lam * w
        w -= lr * grad
        if X_val is not None:
            v = mse(X_val, y_val, w)
            if v < best[0]-1e-10: best = (v, w.copy()); wait=0
            else:
                wait += 1
                if early_stop and wait>=patience: break
    return best[1] if early_stop else w
```

## Week 3-1. (3-1search1-w3-1)

# In Class — SAME Problem: Transportation (1 → n)

**Problem (modeling)**  
Street blocks 1..n. From state $s$:
- **walk** to $s+1$ with cost 1
- **tram** to $2s$ with cost 2 (only if $2s\le n$)
Start $s=1$; **end** if $s=n$. States only increase ⇒ **acyclic**.

## Tasks
1) **Model**: write Start, IsEnd, Actions, Succ, Cost.  
2) **Tree search sketch**: list reachable states up to depth 3 from $s=1$. What are $b$, upper bound $D$, and a plausible solution depth $d$?  
3) **DP recurrence**: derive FutureCost(s). Fill a table for $n=10$ from $s=10\downarrow 1$.  
4) **UCS by hand (first few steps)** for $n=10$: show the frontier (state : pastCost) after each pop until first reaching $n$. Optimal path and cost?

**Submission**: formulas + your table + UCS snapshots (2–3 decimals).

# Out Class Homework — SAME Problem Programmatically: Tree Search · DP · UCS

Implement the **TransportationProblem(n, unit_cost=False)** and the following algorithms:

1) **Backtracking search** (minimum-cost path; count node expansions).  
2) **DFS / BFS / DFID** under **unit_cost=True** (treat both actions as cost=1), compare nodes expanded vs solution depth $d$.  
3) **Dynamic programming** (memoized) for acyclic graphs: compute FutureCost(1) and reconstruct an optimal path.  
4) **Uniform Cost Search (UCS)** for non-negative costs (default 1/2), return optimal path and cost.

**Report**: for $n \in$ { 10, 50, 100, 500 }
- optimal cost (DP and UCS must match), path length, nodes expanded (all methods), peak frontier size (BFS/DFID/UCS).  
- discuss when DFID beats BFS in space; when UCS outperforms BFS under non-equal costs.
  
**Challenge**: add **k-tram** $s\to ks$ with cost $c_k$; design an admissible heuristic $h(s)$ and try **A\***.

## reference codes
ref_search_algorithms.py

```python

from collections import deque
import heapq

# ---------- SearchProblem interface ----------
class SearchProblem:
    def start_state(self):
        raise NotImplementedError
    def is_end(self, s):
        raise NotImplementedError
    def succ_and_cost(self, s):
        '''Yield (action, s', cost).'''
        raise NotImplementedError

# ---------- TransportationProblem ----------
class TransportationProblem(SearchProblem):
    '''
    States: integers s in [1..n]
    Actions:
      - 'walk' to s+1 with cost 1
      - 'tram' to 2*s with cost 2 (only if 2*s <= n)
    If unit_cost=True, treat both actions as cost=1 (for BFS/DFID demonstrations).
    '''
    def __init__(self, n, unit_cost=False):
        assert n >= 1
        self.n = n
        self.unit_cost = unit_cost

    def start_state(self):
        return 1

    def is_end(self, s):
        return s == self.n

    def succ_and_cost(self, s):
        if s < self.n:
            # walk
            c = 1 if not self.unit_cost else 1
            yield ('walk', s+1, c)
            # tram
            if 2*s <= self.n:
                c = 2 if not self.unit_cost else 1
                yield ('tram', 2*s, c)

# ---------- Utilities ----------
def reconstruct_path(parent, end_state):
    path = []
    s = end_state
    while s in parent:
        s_prev, action, cost = parent[s]
        path.append((action, s, cost))
        s = s_prev
    path.reverse()
    return path

# ---------- Backtracking (exponential) ----------
def backtracking_min_cost(problem):
    best = {'cost': float('inf'), 'path': None}
    expansions = 0

    def dfs(s, cost_so_far, parent):
        nonlocal expansions
        if cost_so_far >= best['cost']:
            return
        expansions += 1
        if problem.is_end(s):
            best['cost'] = cost_so_far
            best['path'] = reconstruct_path(parent, s)
            return
        for action, sp, c in problem.succ_and_cost(s):
            parent[sp] = (s, action, c)
            dfs(sp, cost_so_far + c, parent)
            parent.pop(sp, None)

    dfs(problem.start_state(), 0, {})
    return best['path'], best['cost'], expansions

# ---------- DFS (unit-cost, stop at first goal) ----------
def dfs_first_solution(problem, max_depth=10**6):
    start = problem.start_state()
    stack = [(start, 0)]
    parent = {}
    seen = set([start])
    expansions = 0
    while stack:
        s, depth = stack.pop()
        expansions += 1
        if problem.is_end(s):
            return reconstruct_path(parent, s), depth, expansions
        if depth == max_depth:
            continue
        for action, sp, c in problem.succ_and_cost(s):
            if sp not in seen:
                seen.add(sp)
                parent[sp] = (s, action, c)
                stack.append((sp, depth+1))
    return None, None, expansions

# ---------- BFS (unit-cost optimal) ----------
def bfs_unit_cost(problem):
    start = problem.start_state()
    q = deque([start])
    parent = {}
    seen = set([start])
    expansions = 0
    depth = {start: 0}
    while q:
        s = q.popleft()
        expansions += 1
        if problem.is_end(s):
            return reconstruct_path(parent, s), depth[s], expansions
        for action, sp, c in problem.succ_and_cost(s):
            if sp not in seen:
                seen.add(sp)
                parent[sp] = (s, action, c)
                depth[sp] = depth[s] + 1
                q.append(sp)
    return None, None, expansions

# ---------- DFS with Iterative Deepening (unit-cost optimal) ----------
def dfid_unit_cost(problem, max_depth=10**6):
    start = problem.start_state()
    expansions_total = 0
    for limit in range(max_depth+1):
        stack = [(start, 0)]
        parent = {}
        seen = {start}
        while stack:
            s, depth = stack.pop()
            expansions_total += 1
            if problem.is_end(s):
                return reconstruct_path(parent, s), depth, expansions_total
            if depth == limit:
                continue
            for action, sp, c in problem.succ_and_cost(s):
                if sp not in seen:
                    seen.add(sp)
                    parent[sp] = (s, action, c)
                    stack.append((sp, depth+1))
    return None, None, expansions_total

# ---------- Dynamic Programming (acyclic) ----------
def dp_future_cost(problem):
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def F(s):
        if problem.is_end(s):
            return 0
        best = float('inf')
        for action, sp, c in problem.succ_and_cost(s):
            best = min(best, c + F(sp))
        return best
    cost = F(problem.start_state())
    # reconstruct greedily
    path = []
    s = problem.start_state()
    while not problem.is_end(s):
        best_act = None
        best_val = float('inf')
        for action, sp, c in problem.succ_and_cost(s):
            val = c + F(sp)
            if val < best_val:
                best_val = val; best_act = (action, sp, c)
        action, sp, c = best_act
        path.append((action, sp, c))
        s = sp
    return path, cost

# ---------- Uniform Cost Search (Dijkstra on implicit graph) ----------
def ucs(problem):
    start = problem.start_state()
    frontier = [(0, start)]
    parent = {}
    best_cost = {start: 0}
    explored = set()
    expansions = 0
    while frontier:
        cost, s = heapq.heappop(frontier)
        if s in explored:
            continue
        explored.add(s)
        expansions += 1
        if problem.is_end(s):
            return reconstruct_path(parent, s), cost, expansions
        for action, sp, c in problem.succ_and_cost(s):
            new_cost = cost + c
            if new_cost < best_cost.get(sp, float('inf')):
                best_cost[sp] = new_cost
                parent[sp] = (s, action, c)
                heapq.heappush(frontier, (new_cost, sp))
    return None, float('inf'), expansions

if __name__ == "__main__":
    # Sanity: n=10 should have optimal total cost 6
    prob = TransportationProblem(10, unit_cost=False)
    path_dp, cost_dp = dp_future_cost(prob)
    path_ucs, cost_ucs, exp_ucs = ucs(prob)
    print("DP:", cost_dp, path_dp)
    print("UCS:", cost_ucs, path_ucs, "expansions:", exp_ucs)
```

## Week 3-2. (3-2search2-w3-2)

# In Class — SAME Problem with Constraint → Relaxation → A*

**Original problem (constrained Transportation)**  
State $s=(\text{loc}, \Delta)$, $\Delta=\sharp\text{walk}-\sharp\text{tram}\ge 0$.  
Start $(1,0)$, End $(n, \Delta\ge 0)$. Actions:
- walk: $ (loc,\Delta)\to(loc+1,\Delta+1)$, cost 1  
- tram: $ (loc,\Delta)\to(2\cdot loc,\Delta-1)$ if $\Delta-1\ge 0$, cost 2

**Relaxed problem**: drop $\Delta\ge 0$. State is **location** only.  
Compute $ \mathrm{FutureCost}_{\text{rel}}(\text{loc})$ (DP or UCS on **reversed** relaxed graph).

**Heuristic**: $h((\text{loc},\Delta)) := \mathrm{FutureCost}_{\text{rel}}(\text{loc})$.

## Tasks
1) Prove $h$ is **consistent** by the relaxation theorem.  
2) For $n=30$, list first ~8 pops of A* (show $g+h$) and compare to UCS.  
3) Fill a table of $ \mathrm{FutureCost}_{\text{rel}}(\text{loc})$ for loc=1..16.  
4) Bonus: $h_0=0$, $h_{\text{walk}}=n-\text{loc}$, show $h_{\max}=\max(h,h_{\text{walk}})$ consistent.

# Out Class Homework  — A* & Relaxed Heuristics (Same Problem)

Implement:
1) **A\***(`astar(problem, h)`) as UCS on modified cost: `cost' = cost + h(s') - h(s)`; return optimal path/cost and node counts.
2) **Relaxed heuristic** $h_{\text{rel}}$: compute $\mathrm{FutureCost}_{\text{rel}}(\text{loc})$ via UCS on the **reversed relaxed** problem (don’t stop early).
3) Baselines: $h_0=0$; $h_{\text{walk}}(\text{loc})=n-\text{loc}$.
4) Combine: $h_{\max}=\max(h_{\text{rel}},h_{\text{walk}})$.

**Experiments** $n\in\{50,200,1000\}$:
- Optimal cost from A* must match UCS;  
- **Expanded states** & **peak frontier**: UCS vs A* with $h_0,h_{\text{walk}},h_{\text{rel}},h_{\max}$;  
- Verify programmatically that all expanded edges satisfy `cost' ≥ 0` (consistency).  
- Optional: plot expansions vs heuristic.

**Optional (Structured Perceptron)**: given target paths, learn walk/tram costs and re-run A*.

# reference code
ref_search2_astar.py

```python
from collections import deque
import heapq

# ---------- Base ----------
class SearchProblem:
    def start_state(self): raise NotImplementedError
    def is_end(self, s):   raise NotImplementedError
    def succ_and_cost(self, s):
        """yield (action, s', cost)"""
        raise NotImplementedError

# ---------- Constrained Transportation ----------
class ConstrainedTransportation(SearchProblem):
    """
    State: (loc, delta), delta = #walk - #tram >= 0
    Start: (1, 0); End: any (n, delta>=0)
    Actions:
      walk: (loc, d) -> (loc+1, d+1)  cost 1
      tram: (loc, d) -> (2*loc, d-1)  cost 2, only if d-1 >= 0 and 2*loc <= n
    """
    def __init__(self, n):
        assert n >= 1
        self.n = n
    def start_state(self): return (1, 0)
    def is_end(self, s): loc, d = s; return loc == self.n and d >= 0
    def succ_and_cost(self, s):
        loc, d = s
        if loc < self.n:
            yield ("walk", (loc+1, d+1), 1)
            if d-1 >= 0 and 2*loc <= self.n:
                yield ("tram", (2*loc, d-1), 2)

# ---------- Relaxed Transportation (drop delta constraint) ----------
class RelaxedTransportation:
    def __init__(self, n): self.n = n
    def neighbors(self, s):
        if s < self.n:
            yield (s+1, 1)
            if 2*s <= self.n:
                yield (2*s, 2)

# ---------- UCS (for reference) ----------
def ucs(problem):
    start = problem.start_state()
    pq = [(0, start)]
    best = {start: 0}
    parent = {}
    explored = set()
    expansions = 0
    while pq:
        cost, s = heapq.heappop(pq)
        if s in explored: continue
        explored.add(s); expansions += 1
        if problem.is_end(s):
            return reconstruct(parent, s), cost, expansions
        for a, sp, c in problem.succ_and_cost(s):
            nc = cost + c
            if nc < best.get(sp, float("inf")):
                best[sp] = nc
                parent[sp] = (s, a, c)
                heapq.heappush(pq, (nc, sp))
    return None, float("inf"), expansions

def reconstruct(parent, s):
    path = []
    while s in parent:
        ps, a, c = parent[s]
        path.append((a, s, c))
        s = ps
    path.reverse()
    return path

# ---------- A* (UCS with f=g+h) ----------
def astar(problem, h):
    start = problem.start_state()
    pq = [(h(start), 0, start)]  # (f, g, s)
    best_g = {start: 0}
    parent = {}
    explored = set()
    expansions = 0
    while pq:
        f, g, s = heapq.heappop(pq)
        if s in explored: continue
        explored.add(s); expansions += 1
        if problem.is_end(s):
            return reconstruct(parent, s), g, expansions
        for a, sp, c in problem.succ_and_cost(s):
            new_g = g + c
            new_f = new_g + h(sp)
            if new_g < best_g.get(sp, float("inf")):
                best_g[sp] = new_g
                parent[sp] = (s, a, c)
                heapq.heappush(pq, (new_f, new_g, sp))
    return None, float("inf"), expansions

# ---------- Heuristics ----------
def h_zero(s): return 0

def make_h_walk(n):
    def h(s):
        loc = s if isinstance(s, int) else s[0]
        return max(0, n - loc)
    return h

def make_h_relaxed(n):
    """
    Compute relaxed FutureCost via UCS on the REVERSED relaxed problem.
    Equivalent to Dijkstra from goal node n on edges:
      (s-1)->s cost 1
      (2*s)->s cost 2
    """
    INF = 10**18
    dist = [INF]*(n+1)
    dist[n] = 0
    pq = [(0, n)]
    while pq:
        d, s = heapq.heappop(pq)
        if d != dist[s]: continue
        # reversed edges
        if s - 1 >= 1:
            v = s - 1; nd = d + 1
            if nd < dist[v]:
                dist[v] = nd; heapq.heappush(pq, (nd, v))
        if 2*s <= n:
            v = 2*s; nd = d + 2
            if nd < dist[v]:
                dist[v] = nd; heapq.heappush(pq, (nd, v))
    def h(s):
        loc = s if isinstance(s, int) else s[0]
        return dist[loc]
    return h

def make_h_max(h1, h2):
    return lambda s: max(h1(s), h2(s))

# ---------- Consistency checker ----------
def check_consistency(problem, h, samples=2000, seed=0):
    import random
    rnd = random.Random(seed)
    for _ in range(samples):
        s = problem.start_state()
        for __ in range(100):
            for a, sp, c in problem.succ_and_cost(s):
                cprime = c + h(sp) - h(s)
                if cprime < -1e-9:
                    return False
            succs = list(problem.succ_and_cost(s))
            if not succs: break
            a, sp, c = rnd.choice(succs)
            s = sp
            if problem.is_end(s): break
    return True

if __name__ == "__main__":
    n = 60
    prob = ConstrainedTransportation(n)
    h0 = h_zero
    hw = make_h_walk(n)
    hr = make_h_relaxed(n)
    hm = make_h_max(hw, hr)
    _, cu, _ = ucs(prob)
    _, c0, _ = astar(prob, h0)
    _, cw, _ = astar(prob, hw)
    _, cr, _ = astar(prob, hr)
    _, cm, _ = astar(prob, hm)
    print("Costs:", cu, c0, cw, cr, cm)
    print("Consistent(walk)?", check_consistency(prob, hw))
    print("Consistent(relaxed)?", check_consistency(prob, hr))

```
## Week 4-1. (4-1mdp1-w4-1)

# In Class — SAME Problem: Dice Game (Policy Evaluation & Value Iteration)

**Setup**  
State space $S=\{\text{in}, \text{end}\}$. Actions at **in**: `stay` or `quit`; **end** has no actions.  
Transitions (γ is specified):  
- `quit`: $T(\text{in},\text{quit},\text{end})=1$, reward $R=10$.  
- `stay`: $T(\text{in},\text{stay},\text{in})=\tfrac{2}{3}$, $T(\text{in},\text{stay},\text{end})=\tfrac{1}{3}$, reward $R=4$ on both branches.

## Tasks
1) **Closed-form** (γ=1): For policy π(stay), solve $V^{\pi}(\text{in}) = \tfrac{1}{3}(4+0) + \tfrac{2}{3}(4+V^{\pi}(\text{in}))$.  
2) **Policy evaluation (iterative)**: with γ=1, initialize $V^{(0)}\equiv0$ and compute $V^{(t)}(\text{in})$ for t=1..5; report the max change $\Delta_t$.  
3) **Value iteration**: initialize $V^{(0)}\equiv0$; compute $V^{(t)}$ and the greedy action $\pi^{(t)}(\text{in})$ for t=1..5. When does policy switch from `quit` to `stay`?  
4) **Discounting**: set γ=0.5. Recompute (1)–(3). Which policy is optimal now?

**Submission**: your recurrence steps, the sequence $V^{(t)}(\text{in})$, and the first t where $\pi^{(t)}$ stabilizes.

# Out Class Homework — SAME Problem Programmatically: Policy Evaluation & Value Iteration on Dice MDP

Implement a tiny MDP toolkit and solve the dice game:

1) **MDP interface** with `states()`, `actions(s)`, `transitions(s,a)` → list of `(s', prob, reward)`, `is_end(s)`, `start_state`.  
2) **policy_evaluation(mdp, policy, gamma, eps)** returning $V^{\pi}$ with $\max_s|Δ|\le$ eps; report iterations and runtime counts.  
3) **value_iteration(mdp, gamma, eps)** returning $V^*, \pi^*$; also return an iteration log $(\max_s|Δ|, \pi^{(t)})$.  
4) **Experiments**:  
   - With γ=1: compare $V^{\pi=\text{stay}}$, $V^{\pi=\text{quit}}$, and $V^*$ (values at **in**).  
   - With γ∈{0.0, 0.5, 0.9}: repeat; analyze when the optimal policy flips.  
5) **(Optional)** Add a **3×4 Volcano GridWorld** MDP (slip probability p, step reward r_step, terminal rewards r_goal/r_lava) and run value iteration for 10, 20, 50 iterations to illustrate “value propagation”.

**Deliverables**: code + a short report (1–2 pages) with tables/plots of $V$, greedy policy vs iteration, and conclusions.

# reference code
ref_mdp1.py

```python
from typing import Dict, List, Tuple, Iterable

State = str
Action = str
Transition = Tuple[State, float, float]  # (next_state, prob, reward)

# --------- MDP base ---------
class MDP:
    def states(self) -> Iterable[State]: ...
    def actions(self, s: State) -> Iterable[Action]: ...
    def transitions(self, s: State, a: Action) -> Iterable[Transition]:
        """Yield (s', prob, reward). Probabilities over s' must sum to 1 for each (s,a)."""
        ...
    def is_end(self, s: State) -> bool: ...
    @property
    def start_state(self) -> State: ...

# --------- Dice Game MDP ---------
class DiceMDP(MDP):
    """
    States: 'in', 'end'
    Actions at 'in': 'stay' or 'quit'; 'end' has no actions.
    Rewards: stay gives +4 then stochastic termination; quit gives +10 then terminate.
    """
    def __init__(self): pass
    def states(self): return ['in', 'end']
    def actions(self, s): return ['stay','quit'] if s == 'in' else []
    def transitions(self, s, a):
        if s == 'end': return []
        if a == 'quit':
            yield ('end', 1.0, 10.0)
        elif a == 'stay':
            yield ('in', 2/3, 4.0)
            yield ('end', 1/3, 4.0)
        else:
            raise ValueError(a)
    def is_end(self, s): return s == 'end'
    @property
    def start_state(self): return 'in'

# --------- Policy evaluation ---------
def policy_evaluation(mdp: MDP, policy: Dict[State, Action], gamma: float=1.0, eps: float=1e-8,
                      max_iters: int=10_000) -> Dict[State, float]:
    V = {s: 0.0 for s in mdp.states()}
    for t in range(max_iters):
        delta = 0.0
        V_prev = V.copy()
        for s in mdp.states():
            if mdp.is_end(s):
                V[s] = 0.0
                continue
            a = policy[s]
            val = 0.0
            for sp, p, r in mdp.transitions(s, a):
                val += p * (r + gamma * V_prev[sp])
            delta = max(delta, abs(val - V_prev[s]))
            V[s] = val
        if delta <= eps:
            break
    return V

# --------- Value iteration ---------
def value_iteration(mdp: MDP, gamma: float=1.0, eps: float=1e-8, max_iters: int=10_000):
    V = {s: 0.0 for s in mdp.states()}
    for t in range(max_iters):
        delta = 0.0
        V_prev = V.copy()
        for s in mdp.states():
            if mdp.is_end(s):
                V[s] = 0.0
                continue
            best = float('-inf')
            for a in mdp.actions(s):
                q = 0.0
                for sp, p, r in mdp.transitions(s, a):
                    q += p * (r + gamma * V_prev[sp])
                if q > best:
                    best = q
            delta = max(delta, abs(best - V_prev[s]))
            V[s] = best
        if delta <= eps:
            break
    # greedy policy
    policy = {}
    Q = {}
    for s in mdp.states():
        if mdp.is_end(s): continue
        best_a, best_q = None, float('-inf')
        for a in mdp.actions(s):
            q = 0.0
            for sp, p, r in mdp.transitions(s, a):
                q += p * (r + gamma * V[sp])
            Q[(s,a)] = q
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a
    return V, policy, Q

# --------- Demo ---------
if __name__ == "__main__":
    mdp = DiceMDP()
    # Policy: always stay
    pi_stay = {'in':'stay'}
    V_stay = policy_evaluation(mdp, pi_stay, gamma=1.0, eps=1e-10)
    print("V^pi(stay) at 'in':", V_stay['in'])  # 12 (γ=1)

    # Policy: always quit
    pi_quit = {'in':'quit'}
    V_quit = policy_evaluation(mdp, pi_quit, gamma=1.0)
    print("V^pi(quit) at 'in':", V_quit['in'])  # 10

    # Value iteration
    Vstar, pistar, Q = value_iteration(mdp, gamma=1.0, eps=1e-10)
    print("V* at 'in':", Vstar['in'], "pi*:", pistar['in'])

```

## Week 4-2. (4-2mdp2-w4-2)
# In Class — SAME Problem: 3×4 Volcano GridWorld (PI vs VI vs Q-Iteration)

World: 3×4 grid, wall at (2,2); Goal G=(1,4) reward +1 (absorbing); Lava L=(2,4) reward −1 (absorbing);
step reward $r_{step}=−0.04$; slip p=0.2; γ=0.99. Actions: U/D/L/R with perpendicular slip.

Tasks:
1) One VI sweep from $V^{(0)}=0$ → compute $V^{(1)}$ (show one full cell calculation).
2) Policy evaluation (uniform-right policy) for 3 iterations; report $\max_s |V^{(t)}−V^{(t−1)}|$.
3) Policy improvement using current V; draw greedy arrows.
4) Compare two PI improvement rounds (PE 5 iters each) vs two VI sweeps (values/policies).
5) With residual ε=0.01, give $||V−V^*||_\infty$ bound for γ=0.99.

# Out Class Homework — SAME Problem Programmatically: PI · VI · Q-Iteration on GridWorld

Implement:
1) value_iteration(mdp, gamma, eps, async=False) with residual logs.
2) policy_iteration(mdp, gamma, eval_eps, max_pe_iters) returning V*, π* and logs.
3) q_value_iteration(mdp, gamma, eps).
4) Experiments over slip∈{0.0,0.1,0.2}, r_step∈{−0.04,−0.02,0.0}.
5) (Optional) prioritized sweeping VI.

Deliverables: code + one-page summary (convergence curves + greedy policies).

# reference codes
ref_mdp2.py

```python
from typing import Dict, Tuple, Iterable
State = Tuple[int,int]  # (row, col)
Action = str            # 'U','D','L','R'

class GridWorldMDP:
    def __init__(self, rows=3, cols=4, walls={(2,2)}, goals={(1,4):1.0}, lava={(2,4):-1.0},
                 step_reward=-0.04, slip=0.2):
        self.R = rows; self.C = cols
        self.walls = set(walls)
        self.terminal = dict(goals); self.terminal.update(lava)
        self.step_reward = step_reward; self.slip = slip
        self.actions_list = ['U','D','L','R']
    def states(self):
        for r in range(1,self.R+1):
            for c in range(1,self.C+1):
                if (r,c) not in self.walls: yield (r,c)
    def is_end(self,s): return s in self.terminal
    def actions(self,s): return [] if self.is_end(s) else self.actions_list
    def _move(self,s,a):
        r,c=s; drc={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}[a]
        rr,cc=r+drc[0],c+drc[1]
        if not (1<=rr<=self.R and 1<=cc<=self.C) or (rr,cc) in self.walls: return s
        return (rr,cc)
    def transitions(self,s,a):
        if self.is_end(s): return
        perp={'U':['L','R'],'D':['L','R'],'L':['U','D'],'R':['U','D']}[a]
        outcomes=[(self._move(s,a),1-self.slip),
                  (self._move(s,perp[0]),self.slip/2.0),
                  (self._move(s,perp[1]),self.slip/2.0)]
        probs={}
        for sp,p in outcomes: probs[sp]=probs.get(sp,0.0)+p
        for sp,p in probs.items():
            r=self.terminal.get(sp,self.step_reward)
            yield (sp,p,r)

def value_iteration(mdp, gamma=0.99, eps=1e-6):
    V={s:0.0 for s in mdp.states()}
    iters=0
    while True:
        iters+=1; delta=0.0
        for s in list(mdp.states()):
            if mdp.is_end(s): V[s]=mdp.terminal[s]; continue
            best=float('-inf')
            for a in mdp.actions(s):
                q=0.0
                for sp,p,r in mdp.transitions(s,a):
                    q+=p*(r+gamma*V[sp])
                if q>best: best=q
            delta=max(delta,abs(best-V[s])); V[s]=best
        if delta<=eps: break
    pi={}
    for s in mdp.states():
        if mdp.is_end(s): continue
        best_a,best_q=None,float('-inf')
        for a in mdp.actions(s):
            q=sum(p*(r+gamma*V[sp]) for sp,p,r in mdp.transitions(s,a))
            if q>best_q: best_q,best_a=q,a
        pi[s]=best_a
    return V,pi,iters

def policy_evaluation(mdp, pi, gamma=0.99, eps=1e-8, max_iters=10000):
    V={s:0.0 for s in mdp.states()}
    for _ in range(max_iters):
        delta=0.0; Vprev=V.copy()
        for s in mdp.states():
            if mdp.is_end(s): V[s]=mdp.terminal[s]; continue
            a=pi[s]
            val=sum(p*(r+gamma*Vprev[sp]) for sp,p,r in mdp.transitions(s,a))
            delta=max(delta,abs(val-Vprev[s])); V[s]=val
        if delta<=eps: break
    return V

def policy_improvement(mdp, V, gamma=0.99):
    pi={}
    for s in mdp.states():
        if mdp.is_end(s): continue
        best_a,best_q=None,float('-inf')
        for a in mdp.actions(s):
            q=sum(p*(r+gamma*V[sp]) for sp,p,r in mdp.transitions(s,a))
            if q>best_q: best_q,best_a=q,a
        pi[s]=best_a
    return pi

def policy_iteration(mdp, gamma=0.99, eval_eps=1e-8, max_pe_iters=1000):
    pi={s:'R' for s in mdp.states() if not mdp.is_end(s)}
    iters=0
    while True:
        iters+=1
        V=policy_evaluation(mdp, pi, gamma=gamma, eps=eval_eps, max_iters=max_pe_iters)
        new_pi=policy_improvement(mdp, V, gamma=gamma)
        if new_pi==pi: break
        pi=new_pi
    return V,pi,iters

def q_value_iteration(mdp, gamma=0.99, eps=1e-6):
    Q={(s,a):0.0 for s in mdp.states() for a in mdp.actions(s)}
    def best_next(sp):
        return 0.0 if mdp.is_end(sp) else max(Q[(sp,a)] for a in mdp.actions(sp))
    iters=0
    while True:
        iters+=1; delta=0.0
        for s in mdp.states():
            if mdp.is_end(s):
                for a in ['U','D','L','R']:
                    if (s,a) in Q: Q[(s,a)]=mdp.terminal[s]
                continue
            for a in mdp.actions(s):
                old=Q[(s,a)]
                new=sum(p*(r+gamma*best_next(sp)) for sp,p,r in mdp.transitions(s,a))
                Q[(s,a)]=new
                delta=max(delta,abs(new-old))
        if delta<=eps: break
    pi={s:max(mdp.actions(s), key=lambda a: Q[(s,a)]) for s in mdp.states() if not mdp.is_end(s)}
    return Q,pi,iters

if __name__=='__main__':
    mdp=GridWorldMDP()
    V_vi,pi_vi,it_vi=value_iteration(mdp,eps=1e-5); print('VI sweeps:',it_vi)
    V_pi,pi_pi,it_pi=policy_iteration(mdp); print('PI iters:',it_pi)
    Q,pi_q,it_q=q_value_iteration(mdp,eps=1e-5); print('Q-Iter iters:',it_q)

```

## Week 5-1. (5-1games1-w5-1)
# In Class — SAME Problem: Tic-Tac-Toe (Minimax/Alpha–Beta/Eval)

**Representation** Board is 3×3 (row-major string of 9 chars), MAX='X', MIN='O', '.' empty.
**Given position (MAX to move)**: `X.O..O...`

## Tasks
1) **Depth-2 minimax** by hand: enumerate MAX moves, then optimal MIN replies; leaf utilities $U\in$ {+1,0,-1}.  
2) **Alpha–Beta trace** with move ordering (center > corner > edge): write α/β updates and pruned branches.  
3) **Depth-limited evaluation (D=3)**: propose $\mathrm{Eval}(s)=w^\top\phi(s)$ with features:
   - open X two-in-a-row, open O two-in-a-row, center control, #X corners.  
   Give compact formula and evaluate the given position.  
4) **Quiescence**: find a position where static depth-1 eval fails (horizon effect). Propose a tactical extension rule for TTT.

# Out Class Homework — SAME Problem Programmatically: Minimax, Alpha–Beta & Eval on Tic-Tac-Toe

Implement:
1) **Game API**: `legal_moves`, `next_state`, `is_terminal`, `winner`, pretty-print.  
2) **minimax(s, depth)** with node counts; **alphabeta(s, depth)** with ordering (center>corner>edge) + transposition table.  
3) **Evaluation** $w^\top\phi$: (open-X-2s, open-O-2s, centerX, cornerX). Do grid search for w to maximize win rate vs depth-2 minimax.  
4) **Experiments**: on 50 random mid-game states, compare expanded nodes and αβ speedup across depths; plot depth vs nodes.  
5) **(Optional)** iterative deepening with time budget; killer-move ordering.

```python
from typing import List, Tuple, Optional, Dict

MAX, MIN = 'X', 'O'

def pretty(s: str) -> str:
    g = [s[i:i+3] for i in range(0,9,3)]
    return "\n".join(" ".join(c if c != '.' else '_' for c in row) for row in g)

def player_to_move(s: str) -> str:
    return MAX if s.count(MAX) == s.count(MIN) else MIN

def legal_moves(s: str) -> List[int]:
    return [i for i,c in enumerate(s) if c == '.']

def next_state(s: str, a: int) -> str:
    p = player_to_move(s)
    return s[:a] + p + s[a+1:]

def lines() -> List[Tuple[int,int,int]]:
    return [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def winner(s: str) -> Optional[str]:
    for a,b,c in lines():
        if s[a] != '.' and s[a] == s[b] == s[c]:
            return s[a]
    return None

def is_terminal(s: str) -> bool:
    return winner(s) is not None or '.' not in s

def utility(s: str) -> int:
    w = winner(s)
    if w == MAX: return +1
    if w == MIN: return -1
    return 0

# ---- Evaluation ----
def eval_features(s: str) -> Tuple[int,int,int,int]:
    # (open-X-2s, open-O-2s, centerX, cornerX)
    openX = openO = 0
    for a,b,c in lines():
        line = s[a]+s[b]+s[c]
        if line.count(MIN)==0 and line.count(MAX)==2: openX += 1
        if line.count(MAX)==0 and line.count(MIN)==2: openO += 1
    centerX = 1 if s[4]==MAX else 0
    corners = [0,2,6,8]
    cornerX = sum(1 for i in corners if s[i]==MAX)
    return (openX, openO, centerX, cornerX)

def eval_linear(s: str, w=(3, -3, 1, 1)) -> int:
    f = eval_features(s)
    return sum(wi*fi for wi,fi in zip(w,f))

# ---- Minimax / Alpha-Beta ----
def minimax(s: str, depth: int) -> Tuple[int, Optional[int], int]:
    """Return (value, best_move, nodes) from perspective of player_to_move(s)."""
    nodes = 0
    def mm(state, d) -> int:
        nonlocal nodes
        nodes += 1
        if is_terminal(state) or d == 0:
            return utility(state) if is_terminal(state) else eval_linear(state)
        p = player_to_move(state)
        moves = legal_moves(state)
        if p == MAX:
            best = -10**9
            for a in moves:
                best = max(best, mm(next_state(state,a), d-1))
            return best
        else:
            best = 10**9
            for a in moves:
                best = min(best, mm(next_state(state,a), d-1))
            return best
    p = player_to_move(s)
    best_move = None
    best_val = -10**9 if p==MAX else 10**9
    for a in legal_moves(s):
        v = mm(next_state(s,a), depth-1)
        if (p==MAX and v>best_val) or (p==MIN and v<best_val):
            best_val, best_move = v, a
    return best_val, best_move, nodes

def move_order_heuristic(s: str, moves: List[int]) -> List[int]:
    # center > corners > edges
    center = [4]; corners = [0,2,6,8]; edges = [1,3,5,7]
    order = center + corners + edges
    return sorted(moves, key=lambda a: order.index(a) if a in order else 99)

def alphabeta(s: str, depth: int, w=(3,-3,1,1)) -> Tuple[int, Optional[int], int, int]:
    """Return (value, best_move, nodes, prunes)."""
    nodes = prunes = 0
    TT: Dict[Tuple[str,int], int] = {}  # simple transposition: (state,depth)->value

    def ab(state, d, alpha, beta) -> int:
        nonlocal nodes, prunes
        nodes += 1
        key = (state, d)
        if key in TT:
            return TT[key]
        if is_terminal(state) or d == 0:
            val = utility(state) if is_terminal(state) else eval_linear(state, w)
            TT[key] = val
            return val
        p = player_to_move(state)
        moves = move_order_heuristic(state, legal_moves(state))
        if p == MAX:
            val = -10**9
            for a in moves:
                val = max(val, ab(next_state(state,a), d-1, alpha, beta))
                alpha = max(alpha, val)
                if alpha >= beta:
                    prunes += 1
                    break
            TT[key]=val; return val
        else:
            val = 10**9
            for a in moves:
                val = min(val, ab(next_state(state,a), d-1, alpha, beta))
                beta = min(beta, val)
                if alpha >= beta:
                    prunes += 1
                    break
            TT[key]=val; return val

if __name__ == "__main__":
    s = "X.O..O..."
    print(pretty(s))
    print("Player to move:", player_to_move(s))
    v1, a1, n1 = minimax(s, depth=4)
    print("Minimax depth=4:", v1, "move", a1, "nodes", n1)
    v2, a2, n2, p2 = alphabeta(s, depth=6)
    print("AlphaBeta depth=6:", v2, "move", a2, "nodes", n2, "prunes", p2)
```
## Week 5-2. (5-2games2-w5-2)
# InClass — SAME Problem: Dice-to-21 (Expectimax with Chance Nodes)

**Rules**  
Score $s\in\{0,\dots\}$. On each turn, MAX chooses **roll** or **stop**.  
- **stop**: terminal payoff $U = s$.  
- **roll**: chance node; add $X\sim\mathrm{Unif}\{1,\dots,6\}$ to score. If new score $>21$ (**bust**), terminal payoff $U=-10$.

**Tasks**
1) **Depth-2 expectimax** from $s=18$: compute value of `roll` vs `stop` under **risk-neutral** $U(x)=x$. Which action is better?  
2) **Risk-averse** utility $U(x)=\sqrt{\max(x,0)} - 2\cdot \max(-x,0)$. Recompute (1). Does the decision change?  
3) **Cutoff $D=4$ with Eval**: propose $\phi(s)=(s,\ \mathbb{1}[s\ge 20],\ \mathbb{1}[s\le 15])$ and a linear $w$. Evaluate a partial tree at $s=12$.  
4) **Sampling**: at a roll node, estimate expectation with $k=3$ samples; discuss variance vs bias.

**Deliverables**: your tree sketches and numeric values (show arithmetic).

# OutClass Homework — SAME Problem Programmatically: Expectimax on Dice-to-21

Implement:
1) **Game API** with chance nodes: `succ_max(s)` (actions), `succ_chance(s, a)` → outcomes with probs, `is_terminal(s)`, `utility(s)`.  
2) **expectimax(state, depth, utility_fn, eval_fn)** supporting MAX and CHANCE; depth-limited with cutoff + Eval.  
3) **Risk studies**: compare policies under (a) risk-neutral $U(x)=x$, (b) risk-averse $U(x)=\sqrt{x_+}-\lambda x_-$ for $\lambda\in\{1,2,4\}$.  
4) **Sampling expectimax**: at chance nodes, use $k\in\{2,4,8\}$ samples; report value error vs exact and node counts.  
5) **(Optional)** MCTS (UCT) baseline with 10k rollouts; compare move choice at $s=18$.

Deliverables: code + short report (tables of values, node counts, plots of k vs error).

```python
from typing import Dict, Tuple, Callable
import random, math

# --------- Dice-to-21 ---------
TARGET = 21

def utility_risk_neutral(x: int) -> float:
    return float(x)

def utility_risk_averse(x: int, lam: float=2.0) -> float:
    xp = max(x, 0); xn = max(-x, 0)
    return math.sqrt(xp) - lam * xn

def succ_max(s: int):
    """Return list of actions at MAX state s."""
    return ["stop", "roll"]

def succ_chance(s: int, a: str):
    """Return list of outcomes (prob, next_state, immediate_reward_flag)."""
    if a == "stop":
        # terminal handled by is_terminal/utility; no chance children
        return []
    # roll: outcomes 1..6, uniform
    outcomes = []
    for x in range(1,7):
        sp = s + x
        outcomes.append((1/6.0, sp))
    return outcomes

def is_terminal(s: int, last_action: str=None) -> bool:
    if last_action == "stop": return True
    return s > TARGET

def terminal_payoff(s: int, last_action: str=None) -> int:
    if last_action == "stop":
        return s
    # bust
    return -10

# --------- Expectimax ---------
def expectimax(state: int, depth: int,
               utility_fn: Callable[[int], float]=utility_risk_neutral,
               eval_fn: Callable[[int], float]=lambda s: s,
               last_action: str=None):
    """
    Returns (value, best_action, nodes) from MAX perspective under chance.
    Depth-limited: at depth==0, use eval_fn(state).
    """
    nodes = 0
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def max_node(s: int, d: int):
        nonlocal nodes
        nodes += 1
        if is_terminal(s, None):  # bust state (s>TARGET)
            return utility_fn(terminal_payoff(s, None)), None
        if d == 0:
            return eval_fn(s), None
        best_val = -1e18
        best_act = None
        for a in succ_max(s):
            if a == "stop":
                val = utility_fn(terminal_payoff(s, "stop"))
            else:
                val = chance_node(s, a, d-1)
            if val > best_val:
                best_val, best_act = val, a
        return best_val, best_act

    @lru_cache(maxsize=None)
    def chance_node(s: int, a: str, d: int):
        nonlocal nodes
        nodes += 1
        # expectation over outcomes
        ev = 0.0
        for p, sp in succ_chance(s, a):
            if is_terminal(sp, None):
                ev += p * utility_fn(terminal_payoff(sp, None))
            elif d == 0:
                ev += p * eval_fn(sp)
            else:
                val, _ = max_node(sp, d)
                ev += p * val
        return ev

    val, act = max_node(state, depth)
    return val, act, nodes

# --------- Sampling Expectimax ---------
def expectimax_sample(state: int, depth: int, k: int=4,
                      utility_fn: Callable[[int], float]=utility_risk_neutral,
                      eval_fn: Callable[[int], float]=lambda s: s):
    """
    Monte Carlo at chance nodes: sample k outcomes (with replacement).
    """
    nodes = 0
    def max_node(s: int, d: int):
        nonlocal nodes
        nodes += 1
        if is_terminal(s, None):
            return utility_fn(terminal_payoff(s, None)), None
        if d == 0:
            return eval_fn(s), None
        best_val, best_act = -1e18, None
        for a in succ_max(s):
            if a == "stop":
                val = utility_fn(terminal_payoff(s, "stop"))
            else:
                val = chance_node(s, a, d-1)
            if val > best_val:
                best_val, best_act = val, a
        return best_val, best_act

    def chance_node(s: int, a: str, d: int):
        nonlocal nodes
        nodes += 1
        ev = 0.0
        for _ in range(k):
            x = random.randint(1,6)
            sp = s + x
            if is_terminal(sp, None):
                ev += utility_fn(terminal_payoff(sp, None))
            elif d == 0:
                ev += eval_fn(sp)
            else:
                v, _ = max_node(sp, d)
                ev += v
        return ev / k

    return max_node(state, depth) + (nodes,)

# --------- Simple evals ---------
def eval_linear(s: int):
    # features: (score, near_target, far)
    return 1.0*s + 5.0*(1 if s>=20 else 0) - 3.0*(1 if s<=15 else 0)

if __name__ == "__main__":
    for depth in [2,3,4]:
        v, a, n = expectimax(18, depth, utility_fn=utility_risk_neutral, eval_fn=eval_linear)
        print(f"depth={depth} -> value={v:.3f}, act={a}, nodes={n}")
    v2, a2, n2 = expectimax(18, 3, utility_fn=lambda x: (x if x>=0 else -2*abs(x)), eval_fn=eval_linear)
    print("risk-averse depth=3 ->", v2, a2, n2)
    vs, as_, ns = expectimax_sample(18, 4, k=4, eval_fn=eval_linear)
    print("sampling depth=4 k=4 ->", vs, as_, ns)
```

# Week 6-1:(6-1csps1-w6-1)
# InClass — SAME Problem: Australia Map Coloring

**Variables & domains**  
$X=\{\mathrm{WA},\mathrm{NT},\mathrm{SA},\mathrm{Q},\mathrm{NSW},\mathrm{V},\mathrm{T}\}$.  
$\mathrm{Dom}=\{\text{R},\text{G},\text{B}\}$.

**Binary constraints** (neighbors must differ)  
Edges: (WA,NT), (WA,SA), (NT,SA), (NT,Q), (SA,Q), (SA,NSW), (SA,V), (Q,NSW), (NSW,V).  
Tasmania (T) is isolated ⇒ independent.

## Tasks
1) **Factor-graph sketch**: circles for provinces, squares for $[u\neq v]$.  
2) **Backtracking by hand (first few steps)**  
   - Use **MRV** (minimum remaining values) + **degree** tie-break: start with SA (highest degree).  
   - Try SA=R. Forward-check neighbors’ domains; continue 3–4 assignments; document any dead ends & backtracks.  
3) **Decomposition**: after mainland is colored, color T arbitrarily (any of R/G/B).  
4) **Deliverable**: your chosen consistent coloring and the order you assigned variables.

_Note_: Any consistent coloring earns full credit; show your reasoning and pruning.

# OutClass Homework — SAME Problem Programmatically: Australia Map Coloring (CSP)

Implement a tiny CSP toolkit and solve the Australia map:

1) **CSP core**  
   - Structures for `variables`, `domains`, and `constraints` (binary predicates).  
   - A function `neighbors(var)`.

2) **Backtracking search**  
   - Variable ordering: **MRV + degree tie-break**.  
   - Value ordering: try least-constraining value (LCV).  
   - **Forward checking** (propagate domain wipe-outs) with undo stack.

3) **Logs & metrics**  
   - Count nodes, backtracks; print the assignment order and domains after forward checking.

4) **Experiments**  
   - Compare plain backtracking vs +MRV vs +MRV+LCV vs +MRV+LCV+FC (forward checking).  
   - Report node counts on Australia; optionally add **N-Queens (N=8)** as a second CSP.

5) **(Optional)**  
   - Implement **AC-3** and compare with forward checking.  
   - Add **event scheduling** (Formulation 1) and solve a toy instance.

**Deliverables**: code + ≤2-page note (tables with node counts and brief analysis).

```python
from typing import Dict, List, Callable, Set, Tuple, Optional, Iterable

Assignment = Dict[str, str]
Domain = Dict[str, List[str]]
Constraint = Callable[[str, str, str, str], bool]  # (xi, vi, xj, vj) -> ok?

class CSP:
    def __init__(self, variables: List[str], domains: Domain):
        self.variables = variables
        self.domains = {v: list(domains[v]) for v in variables}
        self.neigh: Dict[str, Set[str]] = {v: set() for v in variables}
        self.binary_constraints: List[Tuple[str, str, Constraint]] = []

    def add_binary_constraint(self, xi: str, xj: str, pred: Constraint):
        self.neigh[xi].add(xj)
        self.neigh[xj].add(xi)
        self.binary_constraints.append((xi, xj, pred))
        self.binary_constraints.append((xj, xi, lambda a,va,b,vb,pred=pred: pred(b,vb,a,va)))

    def neighbors(self, x: str) -> Set[str]:
        return self.neigh[x]

    def consistent_pair(self, xi: str, vi: str, xj: str, vj: str) -> bool:
        # check constraints involving (xi,xj)
        for a,b,p in self.binary_constraints:
            if a==xi and b==xj:
                if not p(xi,vi,xj,vj): return False
        return True

# ---------- Heuristics ----------
def mrv(assignment: Assignment, csp: CSP) -> str:
    unassigned = [x for x in csp.variables if x not in assignment]
    # min remaining values
    lens = {x: sum(all(csp.consistent_pair(x,v, y, assignment[y]) for y in csp.neighbors(x) if y in assignment)
                   for v in csp.domains[x]) for x in unassigned}
    m = min(lens.values())
    candidates = [x for x in unassigned if lens[x]==m]
    if len(candidates)==1:
        return candidates[0]
    # degree tie-break: choose variable with most constraints on unassigned vars
    def degree(x): 
        return sum(1 for y in csp.neighbors(x) if y not in assignment)
    candidates.sort(key=lambda x: -degree(x))
    return candidates[0]

def lcv(x: str, assignment: Assignment, csp: CSP) -> List[str]:
    # least-constraining value ordering
    def score(v):
        cnt = 0
        for y in csp.neighbors(x):
            if y in assignment: 
                continue
            for w in csp.domains[y]:
                if not csp.consistent_pair(x,v,y,w):
                    cnt += 1
        return cnt
    return sorted(csp.domains[x], key=score)

# ---------- Forward Checking with undo ----------
def forward_check(x: str, v: str, assignment: Assignment, csp: CSP):
    # prune domains of neighbors; return list of (var, removed_values) to undo
    removed = []
    for y in csp.neighbors(x):
        if y in assignment: 
            continue
        to_remove = [w for w in csp.domains[y] if not csp.consistent_pair(x,v,y,w)]
        if to_remove:
            csp.domains[y] = [w for w in csp.domains[y] if w not in to_remove]
            removed.append((y, to_remove))
            if not csp.domains[y]:
                return False, removed
    return True, removed

def undo(removed, csp: CSP):
    for y, vals in removed:
        # restore in any order; keep unique
        cur = set(csp.domains[y])
        for w in vals:
            if w not in cur:
                csp.domains[y].append(w)

# ---------- Backtracking ----------
def backtracking_search(csp: CSP, use_lcv=True, use_fc=True):
    assignment: Assignment = {}
    nodes = 0; backtracks = 0
    order_log = []

    def backtrack():
        nonlocal nodes, backtracks
        if len(assignment)==len(csp.variables):
            return True
        x = mrv(assignment, csp)
        values = lcv(x, assignment, csp) if use_lcv else list(csp.domains[x])
        for v in values:
            nodes += 1
            # check consistency with assigned neighbors
            ok = all(csp.consistent_pair(x,v,y,assignment[y]) for y in csp.neighbors(x) if y in assignment)
            if not ok: 
                continue
            assignment[x]=v; order_log.append((x,v))
            removed = []
            if use_fc:
                ok, removed = forward_check(x,v,assignment,csp)
            if ok:
                if backtrack():
                    return True
            # undo
            if use_fc:
                undo(removed, csp)
            order_log.pop(); assignment.pop(x, None)
        backtracks += 1
        return False

    success = backtrack()
    return success, assignment, nodes, backtracks, order_log

# ---------- Australia instance ----------
def australia_csp():
    vars = ["WA","NT","SA","Q","NSW","V","T"]
    dom = {v:["R","G","B"] for v in vars}
    csp = CSP(vars, dom)
    edges = [("WA","NT"),("WA","SA"),("NT","SA"),("NT","Q"),
             ("SA","Q"),("SA","NSW"),("SA","V"),("Q","NSW"),("NSW","V")]
    ne = lambda xi,vi,xj,vj: vi != vj
    for a,b in edges:
        csp.add_binary_constraint(a,b,ne)
    return csp

if __name__ == "__main__":
    csp = australia_csp()
    ok, sol, nodes, backs, log = backtracking_search(csp, use_lcv=True, use_fc=True)
    print("Solved:", ok, "nodes:", nodes, "backtracks:", backs)
    print("Solution:", sol)
    print("Order:", log)
```

# Week 6-2: (6-2csps2-w6-2)
# InClass — SAME Problem: 3‑Step Object Tracking (Backtracking · FC/AC‑3 · Beam · ICM)

**Setup** Three variables $X_1,X_2,X_3\in\{0,1,2\}$. Observations $o=(0,2,2)$.
Observation factors $O_i(x_i)=\max(0, 2-|x_i-o_i|)$ giving $[2,1,0]$ around $o_i$.  
Transition factors $T_i(x_i,x_{i+1}) = 2$ if equal, $1$ if $|x_i-x_{i+1}|=1$, else $0$.

## Tasks
1) **Partial‑weight demo**: with assignment $x=\{X_1=0\}$, compute dependent factors when extending $X_2$ by v∈{0,1,2}.  
2) **Forward checking**: after setting $X_2=2$, cross out from $X_1,X_3$ any values with $T_1,T_2=0$. Show remaining domains.  
3) **AC‑3 trace**: starting from empty domains $\{0,1,2\}$, run AC‑3 once **using zeros in $T_i$**; write any values removed.  
4) **Beam (K=2)**: expand level by level; list top‑2 partials and their weights at depths 1,2,3.  
5) **ICM (one pass)**: from initial (0,0,0), update $X_2$ then $X_3$ then $X_1$ using local products; show new assignment and weight.

# OutClass Homework — SAME Problem Programmatically: Tracking with Backtracking/AC‑3 · Beam · ICM

1) **Data model** Build factors: domains {0,1,2}; $O_i(x)=\max(0,2-|x-o_i|)$ for $o=(0,2,2)$;
$T_i(x,y)=2\,[x=y]+1\,[|x-y|=1]$.

2) **Backtracking** with MCV/MRV + LCV + **forward checking**, computing partial weights via dependent factors.

3) **AC‑3** (binary, zero‑support pruning) and plug into backtracking (run AC‑3 on domain updates).

4) **Beam search** (K∈{1,2,3}) on partial weights; report best full assignment and weight; compare node counts vs backtracking.

5) **ICM** starting from 5 random initializations; report best weight reached and frequency of local optima.

6) **(Optional)** Add soft constraints on “acceleration” $A_i=|x_{i+1}-2x_i+x_{i-1}|$ with factor $\exp(-\lambda A_i)$; redo (2)-(5).

**Deliverables**: code + ≤2‑page report (tables: nodes/weights; short discussion).

```python
from typing import Dict, List, Tuple, Callable, Optional
import itertools, random

Var = str
Val = int
Assignment = Dict[Var, Val]

class WeightedCSP:
    def __init__(self, variables: List[Var], domains: Dict[Var, List[Val]]):
        self.variables = variables
        self.domains = {v:list(domains[v]) for v in variables}
        # factors: unary[var] -> dict[val]->w ; binary[(u,v)] -> dict[(a,b)]->w
        self.unary = {v:{} for v in variables}
        self.binary = {}  # key is ordered pair (u,v)
        self.neigh = {v:set() for v in variables}

    def add_unary(self, v: Var, table: Dict[Val, float]):
        self.unary[v] = dict(table)

    def add_binary(self, u: Var, v: Var, table: Dict[Tuple[Val,Val], float]):
        self.binary[(u,v)] = dict(table)
        self.binary[(v,u)] = {(b,a):w for (a,b),w in table.items()}
        self.neigh[u].add(v); self.neigh[v].add(u)

    # ----- factor evaluation -----
    def dep_weight(self, x: Assignment, var: Var, val: Val) -> float:
        """Product of factors touching var whose other vars are already assigned in x."""
        w = 1.0
        # unary
        if self.unary[var]:
            w *= self.unary[var].get(val, 0.0)
        # binary with assigned neighbors
        for nb in self.neigh[var]:
            if nb in x:
                w *= self.binary[(var,nb)].get((val, x[nb]), 0.0)
        return w

    def full_weight(self, x: Assignment) -> float:
        # assumes all variables assigned
        w = 1.0
        for v in self.variables:
            if self.unary[v]: w *= self.unary[v].get(x[v], 0.0)
        for (u,v), tab in self.binary.items():
            if (u < v):  # count each undirected pair once
                w *= tab.get((x[u], x[v]), 0.0)
        return w

# ---------- Lookahead: forward checking ----------
def forward_check(csp: WeightedCSP, x: Assignment, var: Var, val: Val):
    """Return (ok, removed) where removed is list of (y, values) pruned; prune only 0-supported values."""
    removed = []
    for y in csp.neigh[var]:
        if y in x: continue
        to_rm = []
        for b in list(csp.domains[y]):
            # check if any factor forbids (var=val, y=b)
            w = csp.binary[(var,y)].get((val,b), 0.0)
            if w == 0.0:
                to_rm.append(b)
        if to_rm:
            removed.append((y, to_rm))
            csp.domains[y] = [b for b in csp.domains[y] if b not in to_rm]
            if not csp.domains[y]:
                return False, removed
    return True, removed

def undo_fc(csp: WeightedCSP, removed):
    for y, vals in removed:
        for b in vals:
            if b not in csp.domains[y]:
                csp.domains[y].append(b)

# ---------- AC-3 ----------
from collections import deque
def enforce_arc_consistency(csp: WeightedCSP):
    """AC-3 using zero-support pruning on binary factors."""
    q = deque()
    for (u,v) in csp.binary.keys():
        q.append((u,v))
    changed = False
    while q:
        u,v = q.popleft()
        dom_u = list(csp.domains[u])
        removed = False
        for a in dom_u:
            # check if a has any supporting b in v's domain with nonzero factor
            ok = any(csp.binary[(u,v)].get((a,b),0.0) > 0.0 for b in csp.domains[v])
            if not ok:
                csp.domains[u].remove(a)
                removed = True
                changed = True
        if removed:
            for w in csp.neigh[u]:
                if w != v:
                    q.append((w,u))
    return changed

# ---------- Heuristics ----------
def mrv(csp: WeightedCSP, x: Assignment) -> Var:
    unassigned = [v for v in csp.variables if v not in x]
    # MRV: smallest domain size
    k = min(len(csp.domains[v]) for v in unassigned)
    cands = [v for v in unassigned if len(csp.domains[v]) == k]
    # tie-break by degree
    cands.sort(key=lambda v: -len([nb for nb in csp.neigh[v] if nb not in x]))
    return cands[0]

def lcv_values(csp: WeightedCSP, x: Assignment, var: Var) -> List[Val]:
    def score(val):
        # count how many neighbor values remain nonzero-compatible
        s = 0
        for nb in csp.neigh[var]:
            if nb in x: continue
            s += sum(1 for b in csp.domains[nb] if csp.binary[(var,nb)].get((val,b),0.0) > 0.0)
        return -s  # smaller is worse
    return sorted(list(csp.domains[var]), key=score)

# ---------- Backtracking ----------
def backtracking(csp: WeightedCSP):
    x: Assignment = {}
    best = (0.0, None)  # (weight, assignment)
    nodes = 0; backs = 0

    # optional AC-3 before search
    enforce_arc_consistency(csp)

    def dfs():
        nonlocal nodes, backs, best
        if len(x) == len(csp.variables):
            w = csp.full_weight(x)
            if w > best[0]: best = (w, dict(x))
            return True
        var = mrv(csp, x)
        for val in lcv_values(csp, x, var):
            nodes += 1
            delta = csp.dep_weight(x, var, val)
            if delta == 0.0: 
                continue
            x[var] = val
            # forward check + AC-3
            ok, removed = forward_check(csp, x, var, val)
            if ok:
                enforce_arc_consistency(csp)
                dfs()
            undo_fc(csp, removed)
            x.pop(var, None)
        backs += 1
        return False

    dfs()
    return best, nodes, backs

# ---------- Beam search ----------
def beam_search(csp: WeightedCSP, K: int):
    # candidates are (assignment, weight)
    cand = [({}, 1.0)]
    for var in csp.variables:
        # extend all
        ext = []
        for x, w in cand:
            for val in csp.domains[var]:
                delta = csp.dep_weight(x, var, val)
                if delta == 0.0: 
                    continue
                x2 = dict(x); x2[var] = val
                ext.append((x2, w*delta))
        # keep top-K by weight
        ext.sort(key=lambda t: t[1], reverse=True)
        cand = ext[:K] if ext else []
        if not cand: break
    # pick best full if exists
    best = max(cand, key=lambda t: t[1]) if cand else ({}, 0.0)
    return best

# ---------- Local search (ICM) ----------
def icm(csp: WeightedCSP, iters: int=10, seed: int=0):
    random.seed(seed)
    # random full assignment (not guaranteed positive weight)
    x = {v: random.choice(csp.domains[v]) for v in csp.variables}
    def local_weight(var, val):
        # local product: unary(var) * binaries with neighbors
        w = csp.unary[var].get(val, 1.0) if csp.unary[var] else 1.0
        for nb in csp.neigh[var]:
            b = x[nb]
            w *= csp.binary[(var,nb)].get((val, b), 0.0)
        return w
    improved = True
    steps = 0
    while improved and steps < iters:
        improved = False; steps += 1
        for v in csp.variables:
            best = max(csp.domains[v], key=lambda a: local_weight(v,a))
            if local_weight(v, best) > local_weight(v, x[v]):
                x[v] = best; improved = True
    # compute full weight at end (includes unary of all vars and binaries once)
    return x

# ---------- Instance: 3-step tracking ----------
def build_tracking_instance():
    vars = ["X1","X2","X3"]
    doms = {v:[0,1,2] for v in vars}
    csp = WeightedCSP(vars, doms)
    obs = { "X1":0, "X2":2, "X3":2 }
    # unary obs factors: 2,1,0 by distance
    for v in vars:
        table = {a: max(0, 2-abs(a-obs[v])) for a in doms[v]}
        csp.add_unary(v, table)
    # binary transitions
    def trans(a,b):
        if a==b: return 2
        if abs(a-b)==1: return 1
        return 0
    for (u,v) in [("X1","X2"),("X2","X3")]:
        tab = {}
        for a in doms[u]:
            for b in doms[v]:
                tab[(a,b)] = trans(a,b)
        csp.add_binary(u,v, tab)
    return csp

if __name__ == "__main__":
    csp = build_tracking_instance()
    best, nodes, backs = backtracking(csp)
    print("Backtracking best:", best, "nodes:", nodes, "backs:", backs)
    csp2 = build_tracking_instance()
    print("Beam K=2:", beam_search(csp2, K=2))
    csp3 = build_tracking_instance()
    print("ICM:", icm(csp3, iters=10, seed=0))
```

# Week 7-1: (7-1markov-bayesnets1-w7-1)
# InClass — SAME Problem: 3-Step 1D Tracking (as MRF & as BN/HMM)

**Domains** $X_1,X_2,X_3\in\{0,1,2\}$. Observations $o=(0,2,2)$.

**MRF factors (undirected)**  
Observation $o_i(x_i)=\max(0,2-|x_i-o_i|)$ → table values in $\{0,1,2\}$.  
Transition $t_i(x_i,x_{i+1})=\begin{cases}2&x_i=x_{i+1}\\1&|x_i-x_{i+1}|=1\\0&\text{else}\end{cases}$.

**Tasks (MRF)**
1) **Exact $Z$ & marginals**: enumerate all assignments with non-zero weight to get  
   $Z=\sum_x \prod o_i(x_i)\,t_1(x_1,x_2)\,t_2(x_2,x_3)$. Then compute $P(X_2=1)$, $P(X_2=2)$.  
   Compare with the **max-weight assignment**.  
2) **One Gibbs update**: with $x_1{=}1,x_3{=}2$, compute unnormalized weights for $x_2\in\{0,1,2\}$  
   via touching factors $o_2,t_1,t_2$, normalize to get $P(X_2=\cdot\mid X_1{=}1,X_3{=}2)$.

**BN view (HMM)**  
Directed chain $H_1\to H_2\to H_3$, emissions $H_i\to E_i$ with $E_i=o_i$.  
Take $p(H_{i+1}\mid H_i)\propto t_i$; $p(E_i\mid H_i)\propto o_i$ (row-normalized).

3) **Posterior on middle state**: compute/derive $P(H_2\mid E_1{=}0,E_2{=}2,E_3{=}2)$ qualitatively (or by simple enumeration/forward pass).  
4) (**Optional, explaining away mini-case**) Using the alarm network $B,E\to A$: compare $P(B{=}1\mid A{=}1)$ vs $P(B{=}1\mid A{=}1,E{=}1)$.

# OutClass Homework — SAME Problem Programmatically: 3-Step Tracking as MRF & BN

**Part A — MRF (exact + Gibbs)**
1) Build factors $o_i,t_i$. Enumerate all assignments, compute $Z$ and marginals $P(X_i)$.  
2) Implement **Gibbs sampling** (systematic scan). Track counts of $X_2$ after burn-in; compare to exact $P(X_2)$.  
   Plot iterations vs $\ell_\infty$ error of marginals.

**Part B — BN/HMM (exact)**
3) Construct a BN with chain $H_1\to H_2\to H_3$, emissions $H_i\to E_i$ (row-normalize from $t_i,o_i$).  
4) Implement exact inference for $P(H_2\mid E_1{=}0,E_2{=}2,E_3{=}2)$ by enumeration or forward–backward. Compare to MRF $P(X_2)$.

**(Optional) Explaining away**
5) Implement the **alarm BN** with $p(b){=}\varepsilon,\ p(e){=}\varepsilon,\ p(a\mid b,e)=[a=b\lor e]$.  
   Show $P(B{=}1\mid A{=}1)=\frac{1}{2-\varepsilon}$, $P(B{=}1\mid A{=}1,E{=}1)=\varepsilon$.

**Deliverables**: code + 1–2 page note (tables for exact vs Gibbs; BN posterior; plots).

```python
from typing import Dict, Tuple, List
import itertools, math, random
random.seed(0)

# ----- MRF: object tracking (3 steps) -----
dom = [0,1,2]
obs = {1:0, 2:2, 3:2}  # o=(0,2,2)

def o(i, x):
    return max(0, 2-abs(x-obs[i]))

def t(x, y):
    if x==y: return 2
    if abs(x-y)==1: return 1
    return 0

def weight(assign):  # assign: (x1,x2,x3)
    x1,x2,x3 = assign
    return o(1,x1)*o(2,x2)*o(3,x3)*t(x1,x2)*t(x2,x3)

def enumerate_exact():
    table = []
    Z = 0.0
    for a in itertools.product(dom, repeat=3):
        w = weight(a)
        if w>0:
            table.append((a, w))
            Z += w
    # marginals for X2
    p2 = {v:0.0 for v in dom}
    for (x1,x2,x3), w in table:
        p2[x2] += w/Z
    # max-weight assignment
    max_a, max_w = max(table, key=lambda t: t[1])
    return Z, p2, max_a, max_w

def gibbs(n_iters=5000, burn_in=500):
    # initialize randomly among support
    x = [random.choice(dom) for _ in range(3)]
    # If zero weight, force to support
    def cond_prob(i, x):
        # return distribution over dom for Xi given others
        probs = []
        for v in dom:
            y = x.copy()
            y[i]=v
            # local factors touching i: o_i, t with neighbors
            if i==0:
                w = o(1,v)*t(v,y[1])
            elif i==1:
                w = o(2,v)*t(y[0],v)*t(v,y[2])
            else:
                w = o(3,v)*t(y[1],v)
            probs.append(max(0.0,w))
        s = sum(probs)
        probs = [p/s if s>0 else 1.0/len(dom) for p in probs]
        return probs
    counts = {v:0 for v in dom}
    for it in range(n_iters):
        for i in range(3):
            probs = cond_prob(i, x)
            r = random.random(); c=0.0
            pick = 0
            for idx,p in enumerate(probs):
                c += p
                if r<=c:
                    pick=idx; break
            x[i]=dom[pick]
        if it>=burn_in:
            counts[x[1]] += 1
    total = sum(counts.values())
    p2_hat = {v: counts[v]/total for v in dom}
    return p2_hat

# ----- BN/HMM: H1->H2->H3, emissions E1..E3 -----
def row_norm_row(vs):
    s = sum(vs)
    return [vi/s if s>0 else 1.0/len(vs) for vi in vs]

# transition CPT p(h_{i+1}|h_i) from t
trans = {x: row_norm_row([t(x,y) for y in dom]) for x in dom}
# emission CPT p(e|h) from o
emit = {h: row_norm_row([o(1,h), o(1,h), o(1,h)]) for h in dom}  # same shape for each i; we will index by obs

def forward_backward(evidence):
    # evidence: dict {i: observed value at Ei} for i=1..3
    # prior over H1: uniform
    prior = [1/3]*3
    # forward
    alpha = [{} for _ in range(4)]  # 1..3
    alpha[1] = {h: prior[h]*emit[h][evidence[1]] for h in dom}
    def norm(d):
        s = sum(d.values()); 
        return {k: v/s for k,v in d.items()}
    alpha[1] = norm(alpha[1])
    alpha[2] = {h2: emit[h2][evidence[2]] * sum(alpha[1][h1]*trans[h1][h2] for h1 in dom) for h2 in dom}
    alpha[2] = norm(alpha[2])
    alpha[3] = {h3: emit[h3][evidence[3]] * sum(alpha[2][h2]*trans[h2][h3] for h2 in dom) for h3 in dom}
    alpha[3] = norm(alpha[3])
    # posterior of H2 via one-step smoothing: proportional to alpha2 * backward2
    # backward from the end:
    beta3 = {h:1.0 for h in dom}
    beta2 = {h2: sum(trans[h2][h3]*emit[h3][evidence[3]]*beta3[h3] for h3 in dom) for h2 in dom}
    # combine:
    post2 = {h: alpha[2][h]*beta2[h] for h in dom}
    s = sum(post2.values()); post2 = {k:v/s for k,v in post2.items()}
    return post2

# ----- Alarm BN illustrating explaining away -----
def alarm_probs(eps=0.05):
    # P(B=1|A=1) and P(B=1|A=1,E=1)
    # Using formulas from lecture
    p1 = 1.0/(2.0 - eps)
    p2 = eps
    return p1, p2

if __name__ == "__main__":
    Z, p2, argmax, w = enumerate_exact()
    print("Exact Z:", Z, "P(X2):", p2, "argmax:", argmax, "w:", w)
    print("Gibbs P(X2) ~", gibbs())
    print("BN posterior H2 | E=(0,2,2):", forward_backward({1:0,2:2,3:2}))
    print("Alarm explaining-away:", alarm_probs())
```

# Week 7-2 (7-2bayesnets2-w7-2)
# InClass — SAME Problem: HMM (3 steps) + Prob. Programs + BN→MRF
State domain $\{0,1,2\}$. Prior $p(H_1)=\mathrm{Unif}$. Transition $p(h_i\mid h_{i-1})=\frac12[\!h_i{=}h_{i-1}\!]+\frac14[\!|h_i-h_{i-1}|=1\!]$.  
Emission $p(e_i\mid h_i)=\frac12[\!e_i{=}h_i\!]+\frac14[\!|e_i-h_i|=1\!]$. Evidence $(e_1,e_2,e_3)=(0,2,2)$.

**Tasks**
1) **Prob. program**: write pseudocode that samples $H_{1:3},E_{1:3}$; and the **alarm** program $B,E\sim\mathrm{Bern}(\varepsilon), A=B\lor E$.  
2) **BN→MRF** with evidence: plug in $E\!=\!e$, then **remove unobserved leaves** and **discard disconnected components** for the query $P(H_2\mid E)$.  
3) **Forward–Backward**: compute $F_1,F_2,F_3$ and $B_3,B_2,B_1$, then $P(H_2\mid E)$. Show your arithmetic (fractions are fine).  
4) **One Gibbs update** on the reduced MRF for $H_2$ given neighbors.  
5) **Particle filtering (K=4)**: show one full step at $i=3$: propose from each $h_2$, weight by $p(e_3\mid h_3)$, resample; report particle counts.

# OutClass Homework — SAME Problem Programmatically: BN II (Gibbs · F–B · Particle Filter)

**Part A — Prob. programming**
- Implement `sample_alarm(eps)` and an HMM sampler `sample_hmm(T)`.

**Part B — BN→MRF + Gibbs**
- Build the reduced MRF for the toy medical BN (C,A,H,I) under evidence $H{=}1,I{=}1$; implement Gibbs to estimate $P(C{=}1\mid H{=}1,I{=}1)$.

**Part C — HMM**
- Implement **forward_backward(evidence)** (return marginals for all $H_i$);  
- Implement **particle_filter(evidence,K)** with propose–weight–resample; track counts only for the last $H_i$.  
- Compare filtering posteriors at $i=3$ for $K\in\{50,200,1000\}$ vs exact smoothing $P(H_3\mid E)$; report $\ell_1$ error and runtime.  
- (Optional) Add **beam search** baseline (K same as particles) and discuss diversity vs accuracy.

**Deliverables**: code + ≤2-page note (tables: posterior & error; brief discussion).

```python
from typing import Dict, List, Tuple
import random, math
random.seed(0)

# ----- Probabilistic programs -----
def bernoulli(eps: float) -> int:
    return 1 if random.random() < eps else 0

def sample_alarm(eps=0.05):
    B = bernoulli(eps)
    E = bernoulli(eps)
    A = 1 if (B or E) else 0
    return {"B":B,"E":E,"A":A}

def sample_hmm(T=3, domain=(0,1,2)):
    def trans(h_prev, h):
        if h==h_prev: return 0.5
        if abs(h-h_prev)==1: return 0.25
        return 0.0
    def emit(h, e):
        if e==h: return 0.5
        if abs(e-h)==1: return 0.25
        return 0.0
    H = [random.choice(domain)]
    E = [random.choice(domain)]
    # redraw E[0] conditioned on H[0]
    E[0] = random.choices(domain, [emit(H[0],e) for e in domain])[0]
    for i in range(1,T):
        H.append(random.choices(domain, [trans(H[i-1],h) for h in domain])[0])
        E.append(random.choices(domain, [emit(H[i],e) for e in domain])[0])
    return H, E

# ----- HMM Forward–Backward -----
def forward_backward(evidence: List[int], domain=(0,1,2)):
    n = len(evidence)
    def trans(hp, h):
        if h==hp: return 0.5
        if abs(h-hp)==1: return 0.25
        return 0.0
    def emit(h, e):
        if e==h: return 0.5
        if abs(e-h)==1: return 0.25
        return 0.0
    # prior uniform
    prior = {h:1/len(domain) for h in domain}
    F = [ {h:0.0 for h in domain} for _ in range(n) ]
    B = [ {h:1.0 for h in domain} for _ in range(n) ]
    # forward
    for h in domain:
        F[0][h] = prior[h] * emit(h, evidence[0])
    # normalize
    s = sum(F[0].values());  F[0] = {h: F[0][h]/s for h in domain}
    for i in range(1,n):
        for h in domain:
            F[i][h] = emit(h, evidence[i]) * sum(F[i-1][hp]*trans(hp,h) for hp in domain)
        s = sum(F[i].values());  F[i] = {h: F[i][h]/s for h in domain}
    # backward
    for i in reversed(range(n-1)):
        for h in domain:
            B[i][h] = sum(B[i+1][hn]*trans(h,hn)*emit(hn, evidence[i+1]) for hn in domain)
        s = sum(B[i].values());  B[i] = {h: B[i][h]/s for h in domain}
    # smoothing
    post = []
    for i in range(n):
        S = {h: F[i][h]*B[i][h] for h in domain}
        s = sum(S.values()); S = {h: S[h]/s for h in domain}
        post.append(S)
    return F, B, post

# ----- Particle Filter (filtering) -----
def particle_filter(evidence: List[int], K=200, domain=(0,1,2), seed=0):
    random.seed(seed)
    def trans(hp, h):
        if h==hp: return 0.5
        if abs(h-hp)==1: return 0.25
        return 0.0
    def emit(h, e):
        if e==h: return 0.5
        if abs(e-h)==1: return 0.25
        return 0.0
    # initialize H1 ~ prior uniform but weight by emission
    particles = random.choices(domain, k=K)
    weights = [emit(h, evidence[0]) for h in particles]
    # resample
    def resample(parts, ws):
        s = sum(ws)
        if s==0: ws = [1.0/len(ws)]*len(ws)
        else: ws = [w/s for w in ws]
        # multinomial resampling
        cs = []
        c=0.0
        for w in ws:
            c+=w; cs.append(c)
        new = []
        for _ in parts:
            r = random.random()
            j=0
            while r>cs[j]: j+=1
            new.append(parts[j])
        return new
    particles = resample(particles, weights)
    # iterate
    for i in range(1, len(evidence)):
        # propose
        proposed = []
        for hprev in particles:
            proposed.append(random.choices(domain, [trans(hprev,h) for h in domain])[0])
        # weight by emission
        weights = [emit(h, evidence[i]) for h in proposed]
        particles = resample(proposed, weights)
    # counts for last Hi
    counts = {h:0 for h in domain}
    for h in particles: counts[h]+=1
    total = sum(counts.values())
    approx = {h: counts[h]/total for h in domain}
    return approx, counts

# ----- Gibbs on tiny medical BN: C,A cause H,I; evidence H=1,I=1 -----
def gibbs_CA(num_iters=5000, burn=500, seed=0):
    random.seed(seed)
    # priors p(C=1)=0.1, p(A=1)=0.3; conditionals:
    pC = 0.1; pA=0.3
    # p(H=1|C,A): OR-like
    def pH(c,a): return 0.9 if (c or a) else 0.1
    # p(I=1|A): itchy if allergies
    def pI(a): return 0.8 if a==1 else 0.2
    # evidence H=1, I=1
    c,a = 0,1
    cntC1=0
    for it in range(num_iters):
        # sample C | A,H=1,I=1 ∝ p(C)p(H=1|C,A)
        w0 = (1-pC)*pH(0,a)
        w1 = pC*pH(1,a)
        s = w0+w1
        c = 1 if random.random() < (w1/s) else 0
        # sample A | C,H=1,I=1 ∝ p(A)p(H=1|C,A)p(I=1|A)
        w0 = (1-pA)*pH(c,0)*pI(0)
        w1 = pA*pH(c,1)*pI(1)
        s = w0+w1
        a = 1 if random.random() < (w1/s) else 0
        if it>=burn:
            cntC1 += c
    return cntC1/(num_iters-burn)
```
# Week 7-3 (7-3bayesnets3-w7-2)

# InClass — SAME Problem: Movie Ratings BN (G → R1, R2) with Missing G

**Variables & domains**
- Genre $G\in\{\mathrm{c},\mathrm{d}\}$ (comedy/drama)
- Raters $R_1,R_2\in\{1,2,3,4,5\}$

**Data**
- **Supervised set** (fully observed): $(G,R_1,R_2)\in\{(d,4,5),(d,4,4),(d,5,3),(c,1,2),(c,5,4)\}$.
- **Unsupervised set** (missing $G$): $(?,2,2),(?,1,2)$.

**Tasks**
1) **MLE (count & normalize)** under **parameter sharing** $p_R(\cdot\mid g)$ for both $R_1,R_2$. Compute $p_G(g)$ and $p_R(r\mid g)$ from the supervised set.
2) **Laplace smoothing** with $\lambda=1$: recompute $p_R(r\mid g)$. Which entries change from 0 to $>0$?
3) **One EM iteration** using the two unsupervised examples:  
   - **E-step**: for each $(r_1,r_2)$, compute $q_g \propto p_G(g)\,p_R(r_1\mid g)\,p_R(r_2\mid g)$; normalize.  
   - **M-step**: add fractional counts to $p_G, p_R$ (optionally with $\lambda$). Report updated $p_G$ and any changed $p_R$ rows.
4) (**Optional**) Discuss how increasing $\lambda$ changes posteriors and updates.

**Deliverable**: your tables for steps 1–3 (show arithmetic).

# OutClass Homework — SAME Problem Programmatically: Learning a Movie-Ratings BN

Implement a compact learner with **parameter sharing** for $p_R(\cdot\mid g)$.

1) **MLE**: `fit_mle(supervised_data, share_R=True)` → CPTs `pG`, `pR`.  
2) **Laplace smoothing**: `fit_mle(..., lambda_=1.0)`; grid $\lambda\in\{0,0.5,1,2\}$; report zero→positive flips.  
3) **EM**: `fit_em(mixed_data, init, lambda_=1.0, iters=1..10)` on mixed supervised+unsupervised examples; plot log-likelihood vs iters.  
4) **Posterior checks**: after EM, compute $P(G\mid r_1,r_2)$ for the unsupervised pairs.  
5) **(Optional)** Naive Bayes extension: one-vs-rest word classification with parameter sharing for `p_word(·|y)` and Laplace smoothing.

Deliverables: code + ≤2-page note (tables: CPTs for MLE/EM, λ-sweep summary, likelihood curve).

```python
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import math, random

# Domains
G_vals = ["c","d"]
R_vals = [1,2,3,4,5]

Example = Tuple[Optional[str], int, int]  # (G or None, R1, R2)

def normalize(d: Dict):
    s = sum(d.values())
    if s == 0:
        # uniform fallback
        n = len(d)
        for k in d:
            d[k] = 1.0/n
    else:
        for k in d:
            d[k] /= s
    return d

def fit_mle(supervised: List[Example], lambda_: float=0.0, share_R: bool=True):
    """
    Fully observed MLE with optional Laplace smoothing and parameter sharing for p_R.
    Returns: pG (dict), pR (dict g-> {r: prob})
    """
    # Counts
    countG = defaultdict(float, {g: 0.0 for g in G_vals})
    if share_R:
        countR = {g: defaultdict(float, {r: 0.0 for r in R_vals}) for g in G_vals}
    else:
        countR1 = {g: defaultdict(float, {r: 0.0 for r in R_vals}) for g in G_vals}
        countR2 = {g: defaultdict(float, {r: 0.0 for r in R_vals}) for g in G_vals}
    # Laplace preload
    for g in G_vals:
        countG[g] += lambda_
        if share_R:
            for r in R_vals:
                countR[g][r] += lambda_
        else:
            for r in R_vals:
                countR1[g][r] += lambda_
                countR2[g][r] += lambda_
    # Tally data
    for g, r1, r2 in supervised:
        assert g is not None, "fit_mle expects fully observed data"
        countG[g] += 1
        if share_R:
            countR[g][r1] += 1
            countR[g][r2] += 1
        else:
            countR1[g][r1] += 1
            countR2[g][r2] += 1
    # Normalize
    pG = normalize(dict(countG))
    if share_R:
        pR = {g: normalize(dict(countR[g])) for g in G_vals}
    else:
        pR = {"R1": {g: normalize(dict(countR1[g])) for g in G_vals},
              "R2": {g: normalize(dict(countR2[g])) for g in G_vals}}
    return pG, pR

def log_likelihood(mixed: List[Example], pG, pR, share_R: bool=True):
    ll = 0.0
    for g_obs, r1, r2 in mixed:
        if g_obs is None:
            # sum over g
            s = 0.0
            for g in G_vals:
                if share_R:
                    s += pG[g]*pR[g][r1]*pR[g][r2]
                else:
                    s += pG[g]*pR["R1"][g][r1]*pR["R2"][g][r2]
            ll += math.log(max(s, 1e-12))
        else:
            g = g_obs
            if share_R:
                prob = pG[g]*pR[g][r1]*pR[g][r2]
            else:
                prob = pG[g]*pR["R1"][g][r1]*pR["R2"][g][r2]
            ll += math.log(max(prob, 1e-12))
    return ll

def e_step_posteriors(mixed: List[Example], pG, pR, share_R=True):
    """Return list of posteriors q for each example (dict over g), using current params."""
    qs = []
    for g_obs, r1, r2 in mixed:
        if g_obs is not None:
            q = {g: 1.0 if g==g_obs else 0.0 for g in G_vals}
        else:
            un = {}
            for g in G_vals:
                if share_R:
                    un[g] = pG[g]*pR[g][r1]*pR[g][r2]
                else:
                    un[g] = pG[g]*pR["R1"][g][r1]*pR["R2"][g][r2]
            s = sum(un.values())
            q = {g: (un[g]/s if s>0 else 1.0/len(G_vals)) for g in G_vals}
        qs.append(q)
    return qs

def m_step(mixed: List[Example], qs, lambda_: float=0.0, share_R: bool=True):
    # fractional counts with Laplace preload
    countG = defaultdict(float, {g: lambda_ for g in G_vals})
    if share_R:
        countR = {g: defaultdict(float, {r: lambda_ for r in R_vals}) for g in G_vals}
    else:
        countR1 = {g: defaultdict(float, {r: lambda_ for r in R_vals}) for g in G_vals}
        countR2 = {g: defaultdict(float, {r: lambda_ for r in R_vals}) for g in G_vals}
    for (g_obs, r1, r2), q in zip(mixed, qs):
        for g in G_vals:
            w = q[g]
            countG[g] += w
            if share_R:
                countR[g][r1] += w
                countR[g][r2] += w
            else:
                countR1[g][r1] += w
                countR2[g][r2] += w
    pG = normalize(dict(countG))
    if share_R:
        pR = {g: normalize(dict(countR[g])) for g in G_vals}
    else:
        pR = {"R1": {g: normalize(dict(countR1[g])) for g in G_vals},
              "R2": {g: normalize(dict(countR2[g])) for g in G_vals}}
    return pG, pR

def fit_em(mixed: List[Example], init=None, lambda_: float=0.0, iters: int=5, share_R=True):
    if init is None:
        # uniform init
        if share_R:
            pR = {g: {r: 1.0/len(R_vals) for r in R_vals} for g in G_vals}
        else:
            pR = {"R1": {g: {r: 1.0/len(R_vals) for r in R_vals} for g in G_vals},
                  "R2": {g: {r: 1.0/len(R_vals) for r in R_vals} for g in G_vals}}
        pG = {g: 1.0/len(G_vals) for g in G_vals}
    else:
        pG, pR = init
    history = [log_likelihood(mixed, pG, pR, share_R=share_R)]
    for _ in range(iters):
        qs = e_step_posteriors(mixed, pG, pR, share_R=share_R)
        pG, pR = m_step(mixed, qs, lambda_=lambda_, share_R=share_R)
        history.append(log_likelihood(mixed, pG, pR, share_R=share_R))
    return (pG, pR), history

if __name__ == "__main__":
    supervised = [("d",4,5),("d",4,4),("d",5,3),("c",1,2),("c",5,4)]
    pG, pR = fit_mle(supervised, lambda_=0.0, share_R=True)
    print("MLE pG:", pG); print("MLE pR:", pR)
    pG1, pR1 = fit_mle(supervised, lambda_=1.0, share_R=True)
    print("Laplace(1) pR for d:", pR1["d"])
    mixed = supervised + [(None,2,2),(None,1,2)]
    (pG_em, pR_em), hist = fit_em(mixed, init=(pG1,pR1), lambda_=1.0, iters=3, share_R=True)
    print("EM pG:", pG_em); print("LL hist:", hist)
```

# Week 8-1 (8-1logic1-w8-1)
# InClass — SAME Problem: Rain–Wet–Slippery KB

**KB** = { Rain, Rain → Wet, Wet → Slippery } over atoms {Rain, Wet, Slippery}.

Tasks
1) **Derivations (modus ponens only)**: add all formulas you can derive. Which ones appear?
2) **Ask/Tell via SAT reasoning** (by hand logic, not code):
   a) Is **Wet** entailed?  Check KB ∪ {¬Wet} satisfiable?
   b) Is **Rain → Slippery** entailed?
   c) Is **¬Rain** contradictory with KB?
3) **Contingency**: Is **Snow** contingent w.r.t. KB? Explain with models intuition.
4) **Shrink M(KB)**: If we Tell[¬Wet], what happens (entailed/contradict/contingent)?
Deliverable: one-page sheet with (i) derived set, (ii) entail/contradict/contingent judgments and 1–2 line justifications.

# OutClass Homework — SAME Problem Programmatically: A Tiny Propositional Logic Engine

Implement a minimal engine to support **Ask/Tell via SAT** and **forward chaining (modus ponens)**.

1) **AST & evaluator**: atoms, Not/And/Or/Imp/Iff; `eval(formula, model)` with model as dict.
2) **Truth-table SAT & entailment**: 
   - `satisfiable(KB)`, `entails(KB, f)` using: `KB ⊨ f` iff `KB ∪ {¬f}` is UNSAT.
   - Return a countermodel if satisfiable (to explain *not* entailed / *not* contradicted).
3) **Forward chaining** with just **modus ponens**, returning all derived formulas; show soundness (derived ⊆ entailed). 
   Discuss incompleteness by example (e.g., KB={Rain, Rain∨Snow → Wet}, cannot derive Wet).
4) **Experiments on the class KB**: verify answers to 2a–2c; list models that witness contingency.
5) **(Optional)** Implement DPLL and/or WalkSAT; compare node counts vs truth-table.

Deliverables: code + ≤2 pages (answers + short discussion of soundness/completeness).

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Iterable, Tuple, List, Optional
import itertools

# ---- AST ----
@dataclass(frozen=True)
class Var:
    name: str

@dataclass(frozen=True)
class Not:
    f: object

@dataclass(frozen=True)
class And:
    a: object; b: object

@dataclass(frozen=True)
class Or:
    a: object; b: object

@dataclass(frozen=True)
class Imp:
    a: object; b: object  # a -> b

@dataclass(frozen=True)
class Iff:
    a: object; b: object  # a <-> b

def atoms_in(f) -> Set[str]:
    if isinstance(f, Var): return {f.name}
    if isinstance(f, Not): return atoms_in(f.f)
    if isinstance(f, (And, Or, Imp, Iff)): return atoms_in(f.a) | atoms_in(f.b)
    raise TypeError(f"Unknown node: {f}")

def eval_formula(f, w: Dict[str, int]) -> int:
    if isinstance(f, Var): return 1 if w.get(f.name, 0) else 0
    if isinstance(f, Not): return 1 - eval_formula(f.f, w)
    if isinstance(f, And): return eval_formula(f.a, w) & eval_formula(f.b, w)
    if isinstance(f, Or):  return max(eval_formula(f.a, w), eval_formula(f.b, w))
    if isinstance(f, Imp): return 1 if (eval_formula(f.a, w)==0 or eval_formula(f.b, w)==1) else 0
    if isinstance(f, Iff): 
        ea, eb = eval_formula(f.a, w), eval_formula(f.b, w)
        return 1 if ea==eb else 0
    raise TypeError(f"Unknown node: {f}")

def models_of_KB(KB: Iterable[object]) -> List[Dict[str,int]]:
    atoms = sorted(set().union(*[atoms_in(f) for f in KB])) if KB else []
    sols = []
    for vals in itertools.product([0,1], repeat=len(atoms)):
        w = dict(zip(atoms, vals))
        if all(eval_formula(f, w)==1 for f in KB):
            sols.append(w)
    return sols

def satisfiable(KB: Iterable[object]) -> Tuple[bool, Optional[Dict[str,int]]]:
    sols = models_of_KB(KB)
    if sols: return True, sols[0]
    return False, None

def entails(KB: Iterable[object], f) -> bool:
    # KB |= f  iff  KB ∪ {¬f} is UNSAT
    sat, _ = satisfiable(list(KB) + [Not(f)])
    return not sat

# ---- Forward chaining with Modus Ponens only ----
def forward_chain_modus_ponens(KB: Iterable[object]) -> Set[object]:
    KB = set(KB)
    changed = True
    while changed:
        changed = False
        # collect (p, (p->q)) pairs
        facts = {f for f in KB if isinstance(f, Var) or (isinstance(f, Not) and isinstance(f.f, Var))}
        imps  = {f for f in KB if isinstance(f, Imp)}
        for imp in list(imps):
            p, q = imp.a, imp.b
            if p in KB and q not in KB:
                KB.add(q); changed = True
    return KB

# ---- Examples used in class ----
Rain, Wet, Slippery, Snow = map(Var, ["Rain","Wet","Slippery","Snow"])

if __name__ == "__main__":
    KB = {Rain, Imp(Rain,Wet), Imp(Wet,Slippery)}
    # Entailment checks
    print("KB entails Wet?", entails(KB, Wet))
    print("KB entails Rain->Slippery?", entails(KB, Imp(Rain,Slippery)))
    print("KB entails not Rain?", entails(KB, Not(Rain)))
    # Forward chaining (MP)
    FC = forward_chain_modus_ponens(KB)
    print("Forward-derived:", FC)
    # Contingency witness for Snow
    print("KB ∪ {Snow} satisfiable?", satisfiable(KB | {Snow}))
    print("KB ∪ {¬Snow} satisfiable?", satisfiable(KB | {Not(Snow)}))
```

# Week 8-2 (8-2logic2-w8-2)
# In Class — SAME Problem: Students–Courses–Knows

**Domain**  
Constants: `alice, bob, cs221, mdp`.  Predicates: `Takes(x,y)`, `Course(y)`, `Covers(y,z)`, `Knows(x,z)`.

**KB (Horn)**  
1) ∀x∀y∀z  (Takes(x,y) ∧ Covers(y,z)) → Knows(x,z)  
2) Takes(alice, cs221)  
3) Covers(cs221, mdp)  
4) Course(cs221)

### Tasks
A) **Propositional (Horn) via MP completeness**  
- Ground the KB (replace variables with constants) to propositional atoms like `Takes_alice_cs221`.  
- Using **forward chaining (MP only)**, derive `Knows(alice, mdp)`. Draw a small derivation DAG.

B) **CNF conversion (practice)**  
Convert $(A∧B)→(C∨D)$ to CNF step by step, then show how a Horn rule becomes a single clause.

C) **Propositional resolution (non-Horn add-on)**  
Augment with `¬Knows(alice, mdp)`. Convert everything to CNF and resolve to the **empty clause** $\Box$.

D) **FOL MP with unification**  
Without propositionalizing, show the **unifier** θ for premises `{Takes(alice,cs221), Covers(cs221,mdp)}` and rule (1), and derive `Knows(alice,mdp)`.

E) (**Optional, FO-resolution**)  
From clauses `[¬Takes(x,y) ∨ ¬Covers(y,z) ∨ Knows(x,z)]` and `[Takes(alice,cs221)]`, `[Covers(cs221,mdp)]`, perform one **FO-resolution** step to obtain the ground fact.

---

# OutClass Homework — SAME Problem Programmatically: Logic II Toolkit

Implement a compact toolkit:

1) **Propositional CNF + resolution**
   - AST → CNF (↔/→ elimination, push ¬, distribute).
   - Resolution refutation `entails_via_resolution(KB, f)` that returns a proof trace (pairs of parent clauses → resolvent), or a counterexample if not derived.

2) **Horn forward chaining**
   - Represent rules `(premises -> head)` and facts; derive all entailed atoms and a DAG of justifications.

3) **FOL unification + FO-MP**
   - Implement terms (Const/Var/Fun) and atoms `Pred(name,args)`.
   - `unify(a,b)` with occurs-check (basic); `subst(theta, obj)`.
   - `fo_modus_ponens(facts, rule)` returns new facts via most-general unifier.

4) **Experiments on the class KB**
   - Show: (i) FC derives `Knows(alice,mdp)`; (ii) CNF+resolution refutes KB∪{¬Knows(alice,mdp)}`; (iii) FO-MP derives `Knows(alice,mdp)` without grounding.

(**Optional**) FO-CNF (Skolemization) and one FO-resolution step.

**Deliverables**: code + ≤2-page note (CNF steps, resolution trace, FC graph, FO-MP unifiers).

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Optional, Iterable, Union
import itertools

# ===== Propositional AST =====
@dataclass(frozen=True)  # atoms
class PVar: name: str
@dataclass(frozen=True)  # unary
class PNot: f: object
@dataclass(frozen=True)  # binary
class PAnd: a: object; b: object
@dataclass(frozen=True)
class POr: a: object; b: object
@dataclass(frozen=True)
class PImp: a: object; b: object
@dataclass(frozen=True)
class PIff: a: object; b: object

def eliminate_iff_imp(f):
    if isinstance(f, PIff):
        # (a<->b) == (a->b)&&(b->a)
        return PAnd(eliminate_iff_imp(PImp(f.a,f.b)), eliminate_iff_imp(PImp(f.b,f.a)))
    if isinstance(f, PImp):
        # (a->b) == (!a || b)
        return POr(PNot(eliminate_iff_imp(f.a)), eliminate_iff_imp(f.b))
    if isinstance(f, PNot): return PNot(eliminate_iff_imp(f.f))
    if isinstance(f, PAnd): return PAnd(eliminate_iff_imp(f.a), eliminate_iff_imp(f.b))
    if isinstance(f, POr):  return POr(eliminate_iff_imp(f.a), eliminate_iff_imp(f.b))
    return f

def push_not(f):
    if isinstance(f, PNot):
        g = f.f
        if isinstance(g, PNot): return push_not(g.f)
        if isinstance(g, PAnd): return POr(push_not(PNot(g.a)), push_not(PNot(g.b)))
        if isinstance(g, POr):  return PAnd(push_not(PNot(g.a)), push_not(PNot(g.b)))
        return f
    if isinstance(f, PAnd): return PAnd(push_not(f.a), push_not(f.b))
    if isinstance(f, POr):  return POr(push_not(f.a), push_not(f.b))
    return f

def distribute_or_over_and(f):
    if isinstance(f, POr):
        A, B = distribute_or_over_and(f.a), distribute_or_over_and(f.b)
        if isinstance(A, PAnd):
            return PAnd(distribute_or_over_and(POr(A.a, B)), distribute_or_over_and(POr(A.b, B)))
        if isinstance(B, PAnd):
            return PAnd(distribute_or_over_and(POr(A, B.a)), distribute_or_over_and(POr(A, B.b)))
        return POr(A,B)
    if isinstance(f, PAnd): return PAnd(distribute_or_over_and(f.a), distribute_or_over_and(f.b))
    return f

def to_cnf(f):
    f1 = eliminate_iff_imp(f)
    f2 = push_not(f1)
    f3 = distribute_or_over_and(f2)
    # extract clauses as sets of literals (name, sign)
    clauses = []
    def gather(g):
        if isinstance(g, PAnd):
            gather(g.a); gather(g.b)
        else:
            # a clause
            lits = set()
            def collect(h):
                if isinstance(h, POr):
                    collect(h.a); collect(h.b)
                elif isinstance(h, PNot) and isinstance(h.f, PVar):
                    lits.add((h.f.name, False))
                elif isinstance(h, PVar):
                    lits.add((h.name, True))
                else:
                    # wrap non-literal as a fresh symbol (rare in our use)
                    lits.add((str(h), True))
            collect(g)
            clauses.append(frozenset(lits))
    gather(f3)
    return set(clauses)

def resolution_entails(kb_clauses: Set[frozenset], query_clauses: Set[frozenset]):
    # Refutation: add negation of query as CNF (here query_clauses already CNF)
    clauses = set(kb_clauses) | set(query_clauses)
    new = set()
    parents = {}  # child -> (c1,c2)
    def resolvents(c1, c2):
        res = set()
        for (p, s1) in c1:
            key = (p, not s1)
            if key in c2:
                # resolvent = (c1\{p^s1}) ∪ (c2\{p^¬s1})
                r = (c1 - {(p,s1)}) | (c2 - {key})
                res.add(frozenset(r))
        return res
    while True:
        pairs = [(c1,c2) for i,c1 in enumerate(clauses) for j,c2 in enumerate(clauses) if i<j]
        for (c1,c2) in pairs:
            for r in resolvents(c1,c2):
                if not r:  # empty clause
                    parents[r] = (c1,c2)
                    return True, parents
                if r not in clauses:
                    new.add(r)
                    parents[r] = (c1,c2)
        if new.issubset(clauses):  # no progress
            return False, parents
        clauses |= new
        new.clear()

# ===== Horn Forward Chaining =====
def forward_chain(facts: Set[str], rules: List[Tuple[Set[str], str]]):
    derived = set(facts)
    just = {}  # head -> premises
    changed = True
    while changed:
        changed = False
        for premises, head in rules:
            if premises.issubset(derived) and head not in derived:
                derived.add(head); just[head] = set(premises); changed=True
    return derived, just

# ===== First-Order: terms, atoms, substitution, unification, FO-MP =====
@dataclass(frozen=True)
class Const: name: str
@dataclass(frozen=True)
class Var: name: str
@dataclass(frozen=True)
class Fun:
    name: str
    args: Tuple[object, ...]
@dataclass(frozen=True)
class Pred:
    name: str
    args: Tuple[object, ...]

Term = Union[Const, Var, Fun]

def occurs(v: Var, t: Term) -> bool:
    if isinstance(t, Var): return t==v
    if isinstance(t, Fun): return any(occurs(v,a) for a in t.args)
    return False

def subst(theta: Dict[Var, Term], obj):
    if isinstance(obj, Var): return theta.get(obj, obj)
    if isinstance(obj, Const): return obj
    if isinstance(obj, Fun):  return Fun(obj.name, tuple(subst(theta,a) for a in obj.args))
    if isinstance(obj, Pred): return Pred(obj.name, tuple(subst(theta,a) for a in obj.args))
    if isinstance(obj, (list,tuple)): return type(obj)(subst(theta,x) for x in obj)
    return obj

def unify(a, b, theta=None):
    if theta is None: theta = {}
    a = subst(theta, a); b = subst(theta, b)
    if a==b: return theta
    if isinstance(a, Var):
        if occurs(a,b): raise ValueError("occurs check fails")
        theta = dict(theta); theta[a]=b; return theta
    if isinstance(b, Var):
        if occurs(b,a): raise ValueError("occurs check fails")
        theta = dict(theta); theta[b]=a; return theta
    if isinstance(a, Fun) and isinstance(b, Fun) and a.name==b.name and len(a.args)==len(b.args):
        for x,y in zip(a.args, b.args):
            theta = unify(x,y,theta)
        return theta
    if isinstance(a, Pred) and isinstance(b, Pred) and a.name==b.name and len(a.args)==len(b.args):
        for x,y in zip(a.args, b.args):
            theta = unify(x,y,theta)
        return theta
    raise ValueError("cannot unify")

def fo_modus_ponens(facts: List[Pred], rule_premises: List[Pred], rule_head: Pred):
    # try to unify conjunction of rule_premises with some subset of facts
    # naive: try all matchings of rule premises to facts
    results = []
    for combo in itertools.permutations(facts, r=len(rule_premises)):
        try:
            theta = {}
            ok=True
            for a,b in zip(combo, rule_premises):
                theta = unify(a, b, theta)
            head_inst = subst(theta, rule_head)
            results.append(head_inst)
        except Exception:
            ok=False
        if ok: break
    return results

# ===== Example wiring for class KB =====
def class_kb_demo():
    # Propositional Horn version
    facts = {"Takes_alice_cs221", "Covers_cs221_mdp"}
    rules = [
        ({"Takes_alice_cs221", "Covers_cs221_mdp"}, "Knows_alice_mdp")
    ]
    derived, just = forward_chain(facts, rules)

    # Propositional resolution refutation for KB ∧ ¬Knows
    A = PVar("Takes_alice_cs221"); B = PVar("Covers_cs221_mdp"); C = PVar("Knows_alice_mdp")
    rule = PImp(PAnd(A,B), C)
    kb_cnf = to_cnf(rule) | to_cnf(A) | to_cnf(B)
    neg_query = to_cnf(PNot(C))
    entails, proof = resolution_entails(kb_cnf, neg_query)

    # FO-MP
    alice, cs221, mdp = Const("alice"), Const("cs221"), Const("mdp")
    x,y,z = Var("x"), Var("y"), Var("z")
    facts_fo = [Pred("Takes",(alice,cs221)), Pred("Covers",(cs221,mdp))]
    rule_prems = [Pred("Takes",(x,y)), Pred("Covers",(y,z))]
    rule_head = Pred("Knows",(x,z))
    fo_results = fo_modus_ponens(facts_fo, rule_prems, rule_head)
    return derived, just, entails, fo_results

if __name__ == "__main__":
    print(class_kb_demo())
```

# Week 8-3 (9-conclusion-w8-2)
# In Class — SAME Problem: CampusBot (One scenario, many tools)

**Scenario**: Design an on-campus delivery robot *CampusBot* that must (i) detect crosswalks, (ii) plan routes, 
(iii) track pedestrians, (iv) comply with campus rules (no-go zones, time windows), and (v) meet ethical requirements.

Tasks
1) **Tool choice per subtask (justify briefly)**  
   - Crosswalk detection → (reflex-based model + inference + learning)  
   - Route planning under static map → (state-based, search)  
   - Stochastic travel times → (state-based, MDP)  
   - Pedestrian tracking from noisy positions → (variable-based)  
   - Restricted zones/time windows → (logic-based)
2) **Map to algorithms** (pick one each): {linear/CNN/kNN}, {UCS/A\*}, {value iteration}, {forward–backward/Gibbs/particle filter}, {model checking/MP/resolution}.
3) **Ethics checklist** (data/objective/inequality/harmful use/IA): list 1–2 concrete risks & mitigations each.
4) **Course roadmap**: propose a Methods–Applications–Foundations triad of next courses preparing you to ship CampusBot.
Deliverable: a one-pager table with columns **subtask → paradigm → algorithm → why** (+ ethics & roadmap sections).

# OutClass Homework — SAME Problem Programmatically: A Tiny Tool-Recommender + Demos

Implement a small toolkit for *CampusBot*:

1) **Tool recommender**: `recommend_tools(spec)` mapping problem features to a paradigm & algorithms.  
2) **Grid A\*** demo: `astar(grid, s, t)` (4-neighbor). Show found path length.  
3) **HMM tracker**: `forward_backward(evidence)` on 1D pedestrian positions (domain {0,1,2}).  
4) **Ethics checklist scorer**: `audit(spec)` that flags data risks, surrogate objectives, inequality, dual-use, and suggests IA-style mitigations.  
5) **Course triad**: `next_courses(goal)` returning **Methods–Applications–Foundations** suggestions.

Deliverables: code + ≤2-page memo (your CampusBot spec, recommender outputs, A\*/HMM screenshots, ethics flags, course triad).

```python
from typing import List, Tuple, Dict, Any
import math, heapq, random

# ---------- 1) Tool recommender ----------
def recommend_tools(spec: Dict[str, bool]) -> Dict[str, Any]:
    """
    spec flags: perception, path_planning, stochastic, hidden_state, logic_rules
    """
    rec = {"paradigms": [], "algorithms": []}
    if spec.get("perception"):
        rec["paradigms"].append("reflex")
        rec["algorithms"].append({"inference":"feedforward", "learning":"SGD", "models":["linear","CNN","kNN"]})
    if spec.get("path_planning"):
        rec["paradigms"].append("state")
        rec["algorithms"].append({"inference":"A* / UCS", "learning":"-", "models":["search"]})
    if spec.get("stochastic"):
        rec["paradigms"].append("state")
        rec["algorithms"].append({"inference":"value iteration", "learning":"TD / Q-learning", "models":["MDP"]})
    if spec.get("hidden_state"):
        rec["paradigms"].append("variable")
        rec["algorithms"].append({"inference":"forward-backward / particle / Gibbs", "learning":"MLE / EM", "models":["HMM/BN/MN"]})
    if spec.get("logic_rules"):
        rec["paradigms"].append("logic")
        rec["algorithms"].append({"inference":"model checking / MP / resolution", "learning":"-", "models":["prop/FOL"]})
    return rec

# ---------- 2) Grid A* demo ----------
def astar(grid: List[str], s: Tuple[int,int], t: Tuple[int,int]) -> Tuple[int, List[Tuple[int,int]]]:
    H, W = len(grid), len(grid[0])
    def h(p): return abs(p[0]-t[0]) + abs(p[1]-t[1])
    def nbrs(p):
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            x,y = p[0]+dx, p[1]+dy
            if 0<=x<H and 0<=y<W and grid[x][y] != '#':
                yield (x,y)
    g = {s:0}; came = {}; openq = [(h(s), 0, s)]
    seen=set()
    while openq:
        _, gc, u = heapq.heappop(openq)
        if u in seen: continue
        seen.add(u)
        if u==t: break
        for v in nbrs(u):
            ng = gc+1
            if ng < g.get(v, 1e9):
                g[v]=ng; came[v]=u
                heapq.heappush(openq, (ng+h(v), ng, v))
    if t not in came and s!=t: return (math.inf, [])
    path = [t]
    while path[-1]!=s:
        path.append(came[path[-1]])
    path.reverse()
    return (len(path)-1, path)

# ---------- 3) HMM forward-backward (domain {0,1,2}) ----------
def fb_1d(evidence: List[int]) -> List[Dict[int,float]]:
    dom = [0,1,2]
    def trans(hp,h): 
        if h==hp: return 0.5
        if abs(h-hp)==1: return 0.25
        return 0.0
    def emit(h,e):
        if e==h: return 0.5
        if abs(e-h)==1: return 0.25
        return 0.0
    n=len(evidence)
    prior = {h:1/3 for h in dom}
    F=[{h:0.0 for h in dom} for _ in range(n)]
    B=[{h:1.0 for h in dom} for _ in range(n)]
    for h in dom: F[0][h]=prior[h]*emit(h,evidence[0])
    s=sum(F[0].values()); F[0]={h:F[0][h]/s for h in dom}
    for i in range(1,n):
        for h in dom:
            F[i][h]=emit(h,evidence[i])*sum(F[i-1][hp]*trans(hp,h) for hp in dom)
        s=sum(F[i].values()); F[i]={h:F[i][h]/s for h in dom}
    for i in reversed(range(n-1)):
        for h in dom:
            B[i][h]=sum(B[i+1][hn]*trans(h,hn)*emit(hn,evidence[i+1]) for hn in dom)
        s=sum(B[i].values()); B[i]={h:B[i][h]/s for h in dom}
    post=[]
    for i in range(n):
        S={h:F[i][h]*B[i][h] for h in dom}; s=sum(S.values()); post.append({h:S[h]/s for h in dom})
    return post

# ---------- 4) Ethics checklist ----------
def audit(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    out = {"data":[], "objective":[], "inequality":[], "harm":[], "ia":[], "actions":[]}
    ds = spec.get("data_sources","")
    if "web" in ds.lower():
        out["data"].append("Web-scraped data can contain offensive content and historical bias; curate & filter.")
        out["actions"].append("Add data filters; human-in-the-loop review; document datasheets.")
    if spec.get("objective","").lower() in {"clicks","views"}:
        out["objective"].append("Surrogate objective may misalign with user welfare.")
        out["actions"].append("Use multi-objective optimization; long-term user value metrics.")
    if spec.get("users") and "underrepresented" in spec["users"]:
        out["inequality"].append("Potential disparity on under-represented groups.")
        out["actions"].append("Audit by group; collect balanced data; min-max (worst-group) loss.")
    if spec.get("potential_misuse"):
        out["harm"].append("Dual-use risks present.")
        out["actions"].append("Red-team; restrict API; watermarking/traceability.")
    out["ia"].append("Prefer IA: keep humans-in-the-loop; design interpretable controls.")
    return out

# ---------- 5) Course triad ----------
def next_courses(goal: str="robotics") -> Dict[str, List[str]]:
    M = {
        "robotics": {
            "Methods": ["CS229", "CS230", "CS234", "CS238"],
            "Applications": ["CS237AB", "CS223A"],
            "Foundations": ["EE364/CS334", "STATS200"]
        },
        "nlp": {
            "Methods": ["CS229", "CS230", "CS228", "CS236"],
            "Applications": ["CS224N", "CS224U", "CS224V", "CS224C", "CS324"],
            "Foundations": ["EE364/CS334", "STATS214/CS229M"]
        },
        "vision": {
            "Methods": ["CS229", "CS230", "CS228"],
            "Applications": ["CS231N", "CS231A", "CS348I"],
            "Foundations": ["EE364/CS334", "STATS200"]
        }
    }
    return M.get(goal.lower(), M["robotics"])
```

# All Projects (Project 1, 2, 3, 4，7 are necessarily required; Others are encouraged)

1. (Due Thursday of Week 2) Project 1 -- PyTorch Installation (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/Project1-pytorchInstallation.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/get-started/locally/)<br/>

2. (Due Thursday of Week 3)  Project 2 -- Learning PyTorch with Examples (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/Project2-Learning%20PyTorch%20with%20Examples.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)<br/>

3. (Due Thursday of Week 4)  Project 3 -- What is torch.nn really? (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/Project3-Torch-nn-Usage.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/tutorials/beginner/nn_tutorial.html)<br/>

4. (Due Thursday of Week 6)  Project 4 -- Image Recognition (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/Project4-DigitsRecognition.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)<br/>

5. (Due Thursday of Week 6)  Project 5 -- Speech Recognition (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/Project5-SpeechRecognition.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html)<br/>

6. (Due Thursday of Week 6)  Project 6 -- Text Recognition (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/Project6-TextRecognition.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/tutorials/beginner/translation_transformer.html)<br/>

7. (Due Thursday of Week 7）  Project 7--Chinese Poem GPT (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/Project7-Tang%20Poetry%20and%20Song%20Lyrics%20GPT.docx)<br/>
   [poemGPT](https://github.com/jinqijinqi/poemGPT/tree/main)<br/>

8. (Due Thursday of Week 8)  Project 8 -- Mario play agent (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/Project8-Mario%20Player.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br/>
