# UESTC Artificial Intelligence Course (2025-2026)
# Lecturer: Jin Qi
# You are welcome!

---

# **教师评分标准：**

* **数学公式**：推导清晰、准确（20分）
* **代码**：规范，运行无误，有适当注释（20分）
* **结果与分析**：结果完整，分析透彻（20分）
* **心得与思考**：有独立思考和AI联系实际应用（20分）
* **格式**：排版规范，结构清晰美观（20分）

---

# 《人工智能入门》“kNN图像融合方法选择” 实验报告范例

* **课程：** 人工智能基础
* **学院：** 信息与通信工程（学生所在学院）
* **姓名：** 张三
* **学号：** 2023123456
* **作业题目：** 第8周：实例学习——kNN图像融合方法选择
* **提交日期：** 2024年9月27日

---

## 一、实验目的

本实验旨在理解并实现一种基础的监督学习算法——k近邻（kNN），用于根据源图像统计量自动选择图像融合方法（加权平均或最大值融合）。该任务展示了基本的AI学习方法如何在图像融合决策中应用。

---

## 二、实验步骤

* 构建一个小型数据集，每条数据包括IR和VIS图像块的均值，以及最佳融合方法的标签（0：平均融合，1：最大值融合）。
* 使用scikit-learn训练一个kNN分类器（k=1）。
* 对新的IR/VIS图像块进行融合方法预测。

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 1. 构建训练数据：特征为[IR均值, VIS均值]，标签为融合方法
X_train = np.array([[0.7, 0.4], [0.3, 0.8]])
y_train = np.array([0, 1])  # 0：平均融合，1：最大值融合

# 2. 训练kNN分类器
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# 3. 预测新图像块的融合方法
X_test = np.array([[0.6, 0.5]])
y_pred = knn.predict(X_test)
print(f"预测的融合方法: {y_pred[0]}（0：平均，1：最大）")
```

---

## 三、算法与数学推导

kNN算法对新样本，通过在特征空间找到最近邻样本来确定其类别。

* **距离公式**（欧氏距离，两个特征）：

  $$d = \sqrt{(IR_{mean}^{(test)} - IR_{mean}^{(i)})^2 + (VIS_{mean}^{(test)} - VIS_{mean}^{(i)})^2}$$
* **kNN判决规则：**
  设 $X_{test}$ 为新样本特征，则预测标签为 $y_{i^*}$：

  $$i^* = \arg\min_i d(X_{test}, X_i)$$

  对于 $k=1$，选与新样本距离最近的训练样本的类别作为预测。

---

## 四、实验结果与分析

**运行结果：**

```
预测的融合方法: 0（0：平均，1：最大）
```

* 对于均值为0.6（IR）和0.5（VIS）的测试块，分类器预测“平均融合”（标签0），因为此点在特征空间距离第一个训练样本（\[0.7, 0.4]，标签0）更近。

**分析：**

* 计算距离如下：

  * 到\[0.7, 0.4]的距离 = $sqrt((0.6-0.7)^2 + (0.5-0.4)^2) ≈ 0.14$
  * 到\[0.3, 0.8]的距离 = $sqrt((0.6-0.3)^2 + (0.5-0.8)^2) ≈ 0.42$
    
    分类器正确预测为标签0（平均融合）。
* 该实验说明，即使是简单的kNN方法，只要特征设计合理，也能自动完成融合方法的决策。

---

## 五、心得与思考

本次作业让我将AIMA第18章“实例学习”中的监督学习与图像融合实际任务结合了起来。通过实践，体会到了如何构建数据集、调用scikit-learn实现kNN、并用数学和代码理解和解释预测结果。

本实验也让我认识到，kNN虽然直观、易实现，但其有效性取决于特征的选取和足够多样化的标签样本。实际应用中如果特征或样本不足，准确性会受影响。未来在更复杂的融合任务中，可以考虑用更多特征或更高级的分类器，甚至作为深度学习方法的前置参考。

这种方法同样适用于其它决策环节，并突出了可解释、可复现的AI流程对科研和工程的重要性。

---------------------------
下面是**严格按照《Artificial Intelligence: A Modern Approach》经典教材章节**编排的**周项目/作业清单与模板**。每周作业包含**姓名、学号、实验题目、实验内容（代码）、数学推导（公式/理论）、实验结果、分析与总结**等模块，适合标准本科/研究生人工智能入门课程教学。
-----
## 📙[请下载作业空模板.docx (请恰当修改程序所用的原始数据，保证与参考代码所用数据不一致！！！）](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx) <br/>

## 课堂及课后作业

## **Week 1：机器学习1（回归与分类）**

## 课堂练习

**R1（线性回归，平方损失，GD 1 步）**  
数据集 $(x,y)\in\{(1,1),(2,3),(4,3)\}$, $\phi(x)=[1, x]$, 初始 $w^{(0)}=[0, 0]$，步长 $\eta=0.1$。  
1) 计算 $\nabla \text{TrainLoss}(w^{(0)})$。  
2) 更新 $w^{(1)}=w^{(0)}-\eta\nabla \text{TrainLoss}(w^{(0)})$。
3) 算 $\nabla\text{TrainLoss}(w^{(1)})$ 与 $w^{(2)}$。

**C1（线性分类，合页损失，次梯度 1 步）**  
样本 $(x,y)\in\{([0,2],+1),([-2,0],+1),([1,-1],-1)\}$, $\phi(x)=[x_1,x_2]$, 当前 $w=[0.5,1.0]$。  
1) 分别计算每个样本的合页损失与次梯度。  
2) 求平均得到训练损失的（次）梯度；解释为何有的为 0。
3) 用 $\eta=0.1$ 做一次更新。  

## 课后练习

**R1（线性回归：平方损失 + GD/SGD）**  
- 实现 `fit_linear_gd(X, y, lr=0.1, epochs=200)` ($\phi(x)=[1,x]$ 或通用 $\phi$)。  
- 打印/绘制训练损失曲线；报告最终 $w$。  
**基础**：在 [`data/regression_toy.csv`](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/BlackBoard-InOutClassExcercise-Code-Bilingual/2-1-learning1-w1-2/data/regression_toy.csv) 上与参考一致。  
**挑战**：加入 SGD（或小批）并比较与 GD 的速度。

**C1（线性分类：合页损失的次梯度）**  
- 实现 `fit_hinge_gd(X, y, lr=0.1, epochs=200)`；标签取 $\{\pm1\}$。  
- 报告训练合页损失与 0–1 准确率。  
**基础**：在 [`data/classification_toy.csv`](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/BlackBoard-InOutClassExcercise-Code-Bilingual/2-1-learning1-w1-2/data/classification_toy.csv) 上与参考一致。  
**挑战**：加入 L2 正则。
---

## 参考代码— 同题的程序实现
**C1:** ref_classification.py

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

**R1:** ref_regression.py

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

## **Week 2-1：机器学习2**

## 课堂练习

1. ## 线性回归的 SGD vs GD（含二次特征微步）
**手算数据** [`data/regression_hand.csv`](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/BlackBoard-InOutClassExcercise-Code-Bilingual/2-2learning2-w2-1/data/regression_hand.csv)： $(1,1),(2,3),(4,3)$ , $\phi(x)=[1,x]$, 步长 $\eta=0.1$.  
1) **GD 一步**：从 $w^{(0)}=[0,0]$  计算梯度并更新 $w^{(1)}$.  
2) **SGD 两步**：按 (1,1) → (2,3) 顺序，从 $w^{(0)}$ 依次更新。  
3) **二次特征微步**： $\phi_2(x)=[1,x,x^2]$. 在样本 (1,1) 上、从  $w=[0,0,0]$,  $\eta=0.1$ 进行一次 SGD 更新。

## 课后练习 — 同题的程序实现

**A 部分 — GD / SGD / 小批量**  
实现 `fit_linear(X,y, method, lr, epochs, batch_size, lr_schedule)`：
- `method ∈ {gd, sgd, minibatch}`；`lr_schedule ∈ {constant, sqrt_decay}` ($\eta_t=\eta_0/\sqrt{t}$)。
- 在 [`data/regression_nonlinear.csv`](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/BlackBoard-InOutClassExcercise-Code-Bilingual/2-2learning2-w2-1/data/regression_nonlinear.csv) 上，分别用 $\phi=[1,x]$  与  $\phi_2=[1,x,x^2]$  比较速度、达到目标损失的轮数、以及最终 MSE。

**B 部分 — 非线性特征**  
实现二次多项式、5 分箱、余弦特征；比较 MSE。

**C— 两层 ReLU 网络**  
在 [`data/classification_xor.csv`](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/BlackBoard-InOutClassExcercise-Code-Bilingual/2-2learning2-w2-1/data/classification_xor.csv) 上训练小型两层网，达到 100% 训练精度。

## 参考代码— 同题的程序实现

**A:** ref_sgd_regression.py

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

**B:** ref_features.py

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

**C:** ref_nn_two_layer.py

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

## **Week 2-2：机器学习3**

## 课堂练习 — 同题双用（两条主线）

## A 线 — 手算反向传播（平方损失）
给  $w=[3,1]$ 、 $\phi=[1,2]$ 、 $V=[0.1, -0.2; 0.3, 0.05]$, $y=2$, 中间非线性函数 $\sigma(x)=\frac{1}{1+e^{-x}}$ ，前向→反向，求   $\nabla_{w,V}L$ 。

## B 线 — K-means 一次迭代
点集: (0,0),(0,3),(3,0),(3,3). $K=2$, $\mu_1=(0,0)$, $\mu_2=(3,3)$. 1) 分配；2) 更新；3) 比较目标值。

## 课后练习 — 同题程序实现

**A 部分 — 两层网反向传播（平方损失）**：[使用数据backprop_toy.json](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/BlackBoard-InOutClassExcercise-Code-Bilingual/2-3learning3-w2-2/data/backprop_toy.json)实现 `forward`/`backward` 并用差分校验。  
**B 部分 — 带重启/k++ 的 K-means**：实现并比较不同初始化与种子下的损失。  
**C 小练习 — L2 + 早停的验证集选择**：[修改并添加数据backprop_toy.json](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/BlackBoard-InOutClassExcercise-Code-Bilingual/2-3learning3-w2-2/data/backprop_toy.json)划分 train/val，网格搜索 $\lambda$，加早停并报告结果。

## 参考代码— 同题的程序实现

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

## **Week 3-1：3-1search1-w3-1**
## 课堂练习 — 同题：交通问题（1 → n）

**问题（建模）**  
街区 1..n。处于 $s$ 时：
- **walk** 到 $s+1$，代价 1
- **tram** 到 $2s$，代价 2（若 $2s\le n$）
起点 $s=1$；当 $s=n$ 为终点。状态单调增 ⇒ **无环**。

## 任务
1) **建模**：写出 Start、IsEnd、Actions、Succ、Cost。  
2) **树搜索草图**：从 $s=1$ 列出 3 层内可达状态。给出分支因子 $b$、上界深度 $D$ 与一个合理的解深度 $d$。  
3) **DP 递推**：推导 FutureCost(s)。对 $n=10$ 自 $s=10\downarrow 1$ 填表。  
4) **UCS 手算（前若干步）**：对 $n=10$ 列出每次弹出后的 frontier（state: pastCost），直到首次到达 $n$。最优路径与代价？

**提交**：关键公式 + 表格 + UCS 截图（2–3 位小数）。

## 课后练习  — 同题的程序实现：树搜索 · 动态规划 · UCS

实现 **TransportationProblem(n, unit_cost=False)** 与以下算法：

1) **回溯搜索**（返回最小代价路径；统计展开节点数）。  
2) 在 **unit_cost=True**（两种动作都视作代价 1）下实现 **DFS / BFS / DFID**，比较节点展开量与解深度 $d$。  
3) **动态规划**（记忆化，适用于无环）：计算 FutureCost(1) 并重构一条最优路径。  
4) **一致代价搜索（UCS）**（非负代价，默认为 1/2）：返回最优路径与代价。

**报告**：对 $n \in \{10, 50, 100, 500\}$  
- 给出最优代价（DP 与 UCS 应一致）、路径长度、各法的展开节点数与 frontier 峰值；  
- 讨论 DFID 何时在空间上优于 BFS；非等代价下 UCS 为何优于 BFS。
- 
**进阶**：加入 **k-tram** 动作 $s\to ks$ 费用 $c_k$；设计可采纳启发 $h(s)$ 尝试 **A\***（可选加分）。

## 参考代码— 同题的程序实现

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

## **Week 3-2：3-2search2-w3-2**

# 课堂练习 — 同题：带约束 → 松弛 → A*

**原问题（带约束的交通）**  
状态  $s=(\text{loc}, \Delta)$, $\Delta=$ `#` walk- `#` tram $\ge 0$.  
起点 $(1,0)$, 终点  $(n,\Delta\ge 0)$. 动作：
- walk: $(loc,\Delta)\to(loc+1,\Delta+1)$, 代价 1  
- tram: $(loc, \Delta)\to (2\cdot loc,\Delta-1)$ （需  $\Delta-1\ge 0$)，代价 2

**松弛问题**：移除 $\Delta\ge0$ 约束；状态仅为 **location**。  
计算 $\mathrm{FutureCost}_{\text{rel}}(\text{loc})$ （在**反向**松弛图上用 UCS）。

**启发**： $h((\text{loc},\Delta)):=\mathrm{FutureCost}_{\text{rel}}(\text{loc})$.

## 任务
1) 用“松弛 ⇒ 一致”定理证明 $h$ 一致。  
2) $n=30$ 时手算 A* 前 8 次弹出（写 $g+h$）并与 UCS 对比。  
3) 列 loc=1..16 的  $\mathrm{FutureCost}_{\text{rel}}(\text{loc})$ 表。  
4) 进阶: $h_0=0$, $h_{\text{walk}}=n-\text{loc}$, 验证 $h_{\max}=\max(h,h_{\text{walk}})$  一致。

# 课后练习作业 — A* 与松弛启发（同题）

实现：
1) **A\***（以 `cost' = cost + h(s') - h(s)` 的 UCS 实现），输出最优路径/代价与节点统计；  
2) **松弛启发**  $h_{\text{rel}}$ ：在**反向松弛问题**上用 UCS 求所有点的过去代价 ⇒ 即松弛的未来代价；  
3) 基线: $h_0=0$, $h_{\text{walk}}(\text{loc})=n-\text{loc}$;  
4) 组合: $h_{\max}=\max(h_{\text{rel}},h_{\text{walk}})$.

**实验** $n\in\{50,200,1000\}$:
- A* 与 UCS 的最优代价一致；  
- **展开节点数**与**frontier 峰值**：UCS 与 A* ($h_0,h_{\text{walk}},h_{\text{rel}}, h_{\max}$) 对比;  
- 程序验证所有扩展边满足 `cost' ≥ 0`（一致性）；  
- 选做：绘图比较启发强弱与展开量。

**选学（结构化感知机）**：给定目标路径，学习 walk/tram 代价并用 A* 复现。


## 参考代码— 同题的程序实现
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

## **Week 4-1：4-1mdp1-w4-1**

# 课堂练习 — 同题：掷骰子游戏（策略评估与价值迭代）

**设定**  
状态 $S=\{\text{in}, \text{end}\}$. **in** 上动作为 `stay` 或 `quit`；**end** 无动作。  
转移（给定 γ）：  
- `quit`: $T(\text{in}, \text{quit}, \text{end})=1$, 奖励 $R=10$.  
- `stay`: $T(\text{in},\text{stay},\text{in})=\tfrac{2}{3}$, $T(\text{in},\text{stay},\text{end})=\tfrac{1}{3}$, 两条分支奖励均 $R=4$.

## 任务
1) **闭式**（γ=1）：对策略 π(stay)，解 $V^{\pi}(\text{in})=\tfrac{1}{3}(4+0)+\tfrac{2}{3}(4+V^{\pi}(\text{in}))$.  
2) **迭代评估**：γ=1, $V^{(0)}\equiv0$, 计算 $t=1..5$ 的 $V^{(t)}(\text{in})$, 给出每步最大变化 $\Delta_t$.  
3) **价值迭代**: $V^{(0)}\equiv0$, 计算 $V^{(t)}$ 与贪心动作 $\pi^{(t)}(\text{in})$ (t=1..5). 策略何时由 `quit` 变为 `stay`？  
4) **折扣**：设 γ=0.5，重做 (1)–(3)。此时最优策略是谁？

**提交**：推导过程, $V^{(t)}(\text{in})$ 序列, $\pi^{(t)}$ 稳定的首个步数 t。

# 课后作业 — 同题程序实现：骰子 MDP 的策略评估与价值迭代

实现一个小型 MDP 工具并求解骰子游戏：

1) **MDP 接口**：`states()`、`actions(s)`、`transitions(s,a)` → `(s', prob, reward)` 列表、`is_end(s)`、`start_state`。  
2) **policy_evaluation(mdp, policy, gamma, eps)**：返回 $V^{\pi}$, 并满足 $\max_s|Δ|\le$ eps; 报告迭代次数与计数。  
3) **value_iteration(mdp, gamma, eps)**: 返回  $V^\*,\pi^\*$; 同时返回 $(\max_s|Δ|,\pi^{(t)})$ 迭代日志.  
4) **实验**：  
   - γ=1：比较 $V^{\pi=\text{stay}}$, $V^{\pi=\text{quit}}$, 以及 $V^*$ (在 **in** 的取值）。  
   - γ∈{0.0, 0.5, 0.9}：重复，分析最优策略何时翻转。  
5) **（选做）** 实现 **3×4 火山网格世界**（滑移概率 p、步长惩罚 r_step、终止奖励 r_goal/r_lava），做 10/20/50 次价值迭代展示“价值传播”。

**提交**：代码 + 1–2 页小报告（含 $V$, 贪心策略随迭代的表/图与结论）。

## 参考代码— 同题的程序实现
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

## **Week 4-2：4-2mdp2-w4-2**

# 课堂练习 — 同题：3×4 火山网格世界（PI vs VI vs Q 迭代）

世界：3×4，墙 (2,2)；目标 G=(1,4) 奖励 +1 吸收；岩浆 L=(2,4) 奖励 −1 吸收；
步长奖励 r_step=−0.04；滑移 p=0.2；γ=0.99；动作 U/D/L/R（含垂直滑移）。

任务：
1) 从 $V^{(0)}=0$ 做一次 VI 扫描得 $V^{(1)}$（演示一个格子完整计算）。
2) “全向右”策略做 3 次策略评估；给出 $max_s |V^{(t)}−V^{(t−1)}|$.
3) 用当前 V 做策略改进并画出贪心箭头。
4) 做两轮 PI（每轮评估 5 步）与两步 VI，比较值与策略。
5) 残差 ε=0.01、γ=0.99 时给出 $||V−V^*||_∞$ 上界。

# 课后作业 — 同题程序实现：PI · VI · Q 迭代（GridWorld）

实现：
1) value_iteration（含残差日志与停止）。
2) policy_iteration（迭代评估+贪心改进，返回 V*、π*）。
3) q_value_iteration。
4) 在不同滑移/步长奖励下比较收敛步数与策略。
5) （选做）优先级扫描 VI。

提交：代码 + 一页总结（收敛曲线与贪心策略）。

## 参考代码— 同题的程序实现
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

## **Week 5-1：5-1games1-w5-1**

# 课堂练习 — 同题：井字棋（Minimax/Alpha–Beta/估值）

**表示** 棋盘 3×3（行优先 9 字符串），MAX='X'，MIN='O'，`.` 为空。
**给定局面（MAX 走）**：`X.O..O...`

## 任务
1) **手算 2 层极大极小**：枚举 MAX 着法，再考虑 MIN 最优应手；叶子效用 $U\in\{+1,0,-1\}$。  
2) **Alpha–Beta 跟踪**（次序：中心>角>边）：写出 α/β 更新与被剪枝分支。  
3) **深度截断估值（D=3）**: 设 $\mathrm{Eval}(s)=w^\top\phi(s)$, 特征:
   - X 的开放二连、O 的开放二连、中心占位、X 的角格数。  
   写出紧凑公式，并在给定局面上求值。  
4) **静止搜索**：给出一个深度 1 评估失真（地平线效应）的局面，并提出在 TTT 中的战术延伸规则。

# 课后作业 — 同题程序实现：井字棋的极大极小、Alpha–Beta 与估值

实现：
1) **博弈接口**：`legal_moves`、`next_state`、`is_terminal`、`winner`、打印函数。  
2) **minimax(s, depth)**（记录节点数）；**alphabeta(s, depth)**（中心>角>边排序 + 置换表）。  
3) **估值** $w^\top\phi$：(open-X-2s, open-O-2s, centerX, cornerX)。做网格搜索以提升对 depth-2 minimax 的胜率。  
4) **实验**：在 50 个随机中盘局面对比节点展开数与 αβ 加速比，绘制“深度-节点数”曲线。  
5) **（选做）** 迭代加深（含时限）；killer move 排序。

## 参考代码— 同题的程序实现
ref_games1.py

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

## **Week 5-2：5-2games2-w5-2**

# 课堂练习 — 同题：掷骰到 21（含机会结点的 Expectimax）

**规则**  
分数 $s\in$ { $0,\dots$ }. 每回合 MAX 选 **roll** 或 **stop**：  
- **stop**：终止收益 $U=s$；  
- **roll**: 进入机会结点; 加上 $X\sim\mathrm{Unif}$ { $1,\dots,6$ }. 若新分 $>21$（爆），终止收益 $U=-10$。

**任务**
1) 从 $s=18$ 做 **2 层 expectimax**：在**风险中性** $U(x)=x$ 下比较 `roll` 和 `stop` 的值。  
2) **风险厌恶** $U(x)=\sqrt{\max(x,0)}-2\cdot\max(-x,0)$ 下重算 (1)。决策是否变化？  
3) **深度截断 $D=4$ + 估值**：给出 $\phi(s)=(s,\ \mathbb{1}[s\ge 20],\ \mathbb{1}[s\le 15])$ 与线性 $w$，在 $s=12$ 处评一个子树。  
4) **采样**：在掷骰机会结点用 $k=3$ 次样本估计期望；讨论方差与偏差。

**提交**：手绘树与数值（给出计算步骤）。

# 课后作业 — 同题程序实现：掷骰到 21 的期望极大

实现：
1) **带机会结点的接口**：`succ_max(s)`、`succ_chance(s, a)`（结果与概率）、`is_terminal(s)`、`utility(s)`。  
2) **expectimax(state, depth, utility_fn, eval_fn)**：支持 MAX/CHANCE，含深度截断与估值。  
3) **风险分析**：比较以下效用下的策略：(a) 风险中性 $U(x)=x$；(b) 风险厌恶 $U(x)=\sqrt{x_+}-\lambda x_-$, $\lambda\in\$ {1,2,4}。  
4) **采样期望极大**：在机会结点用 $k\in\{2,4,8\}$ 样本；报告与精确值的误差及节点数。  
5) **（选做）** MCTS（UCT）10k 次 rollout；比较在 $s=18$ 的选招。

提交：代码 + 简短报告（值/节点计数表与“k—误差”图）。

## 参考代码— 同题的程序实现
ref_games2.py

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
## **Week 6-1：6-1csps1-w6-1**

# 课堂练习 — 同题：澳大利亚地图着色

**变量与域**  
$X=$ { $\mathrm{WA},\mathrm{NT},\mathrm{SA},\mathrm{Q},\mathrm{NSW},\mathrm{V},\mathrm{T}$ }.  
$\mathrm{Dom}=$ { $\text{R},\text{G},\text{B}$ }.

**二元约束**（相邻不同色）  
边： (WA,NT)、(WA,SA)、(NT,SA)、(NT,Q)、(SA,Q)、(SA,NSW)、(SA,V)、(Q,NSW)、(NSW,V)。  
塔斯马尼亚 T 与大陆无邻接 ⇒ 可独立处理。

## 任务
1) **因子图草图**：省份画成圆, $[u\neq v]$ 因子画方框.  
2) **手算回溯（前几步）**  
   - 用 **MRV** + **度数**打破并列：先给 SA 着色（度最高）。  
   - 试 SA=R，前视删减邻居域；继续 3–4 次赋值；记录冲突与回溯。  
3) **分解**：大陆着色完成后，T 任取（R/G/B）。  
4) **提交**：一个一致着色方案与赋值顺序。

_说明_：任一一致解均可；请展示剪枝过程与理由。

# 课后作业 — 同题程序实现：澳大利亚地图着色（CSP）

实现一个小型 CSP 工具并求解该地图：

1) **CSP 核心**  
   - `variables`、`domains` 与二元 `constraints`（谓词）。  
   - 提供 `neighbors(var)`。

2) **回溯搜索**  
   - 变量顺序：**MRV + 度数打平**；  
   - 取值顺序：最少约束值（LCV）；  
   - **前视检查**（Forward Checking），带撤销栈。

3) **日志与指标**  
   - 统计节点数、回溯次数；输出赋值顺序与每步前视后的域。

4) **实验**  
   - 比较：朴素回溯 / +MRV / +MRV+LCV / +MRV+LCV+FC；  
   - 在澳大利亚数据集上报告节点数；可选再做 **N 皇后（N=8）**。

5) **（选做）**  
   - 实现 **AC-3** 并与前视比较；  
   - 增加**排程**（表述一），解一个小实例。

**提交**：代码 + ≤2 页小报告（节点数表格与简要分析）。

## 参考代码— 同题的程序实现
ref_csps1.py

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

## **Week 6-2：6-2csps2-w6-2**

# 课堂练习 — 同题：三步目标跟踪（回溯 · 前视/AC‑3 · Beam · ICM）

**设定** 变量 $X_1,X_2,X_3\in$ { $0,1,2$ }, 观测 $o=(0,2,2)$。  
观测因子 $O_i(x_i)=\max(0, 2-|x_i-o_i|)$（以 $o_i$ 为中心的 $[2,1,0]$）；
转移因子 $T_i(x_i,x_{i+1})$：相等 2，相差 1 给 1，否则 0。

## 任务
1) **部分权演示**：在 $x=$ { $X_1=0$ } 下扩展 $X_2\in$ { $0,1,2$ } 时的依赖因子与 $\delta$。  
2) **前视检查**：设 $X_2=2$ 后，用 $T_1,T_2$ 将 $X_1,X_3$ 中与之不相容的取值删除，给出新域。 
3) **AC‑3 跟踪**：从空赋值、初域 { $0,1,2$ } 出发，**利用 $T_i$ 中为 0 的对儿** 做一轮 AC‑3；记录删掉的取值。  
4) **束搜索（K=2）**：逐层扩展；在深度 1/2/3 分别列出前 2 个候选及其权重。  
5) **ICM（单次遍历）**：从 (0,0,0) 出发，依次更新 $X_2\rightarrow X_3\rightarrow X_1$，给出新赋值与权重。

# 课后作业 — 同题程序实现：回溯/AC‑3 · 束搜索 · ICM 的三步跟踪

1) **数据模型** 建立因子：域 {0,1,2}；观测 $O_i(x)=\max(0,2-|x-o_i|)$ ( $o=(0,2,2)$ )；
转移 $T_i(x,y)=2[x=y]+1[|x-y|=1]$。

2) **回溯**：MCV/MRV + LCV + **前视检查**，用依赖因子计算部分权。

3) **AC‑3**（二元、零支撑剪枝），接入回溯（域更新时触发 AC‑3）。

4) **束搜索**（K∈{1,2,3}）：基于部分权扩展；报告最优完整赋值与权重；与回溯的节点数对比。

5) **ICM**：从 5 个随机初值开始；报告达到的最佳权重与局部最优出现频率。

6) **（选做）** 加入“加速度”软约束 $A_i=|x_{i+1}-2x_i+x_{i-1}|$，因子取 $\exp(-\lambda A_i)$；重复 (2)-(5)。

**提交**：代码 + ≤2 页报告（节点/权重表；简要讨论）。

## 参考代码— 同题的程序实现
ref_csps2.py

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

# 所有作业（作业1，2，3，7，8必做，作业4，5，6必选一个，但鼓励大家都选）
1. (第二周周四截止）Homework 1--Pytorch Installation 简述!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx)<br/>
   [参考pytorch网页](https://pytorch.org/get-started/locally/)<br/>
2. (第三周周四截止）Homework 2--Learning PyTorch with Examples简述内容!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx)<br/>
   [参考pytorch网页](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)<br/>
3. (第四周周四截止）Homework 3--What is torch.nn really?简述内容!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx)<br/>
   [参考pytorch网页](https://pytorch.org/tutorials/beginner/nn_tutorial.html)<br/>
4. (第六周周四截止）Homework 4--图像识别简述内容!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx)<br/>
   [参考pytorch网页](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)<br/>
5. (第六周周四截止） Homework 5--语音识别简述内容!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx)<br/>
   [参考pytorch网页](https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html)<br/>
6. (第六周周四截止） Homework 6--文本识别简述内容!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx)<br/>
   [参考pytorch网页](https://pytorch.org/tutorials/beginner/translation_transformer.html)<br/>
7. (第七周周四截止） Homework 7--唐诗宋词GPT!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx)<br/>
   [poemGPT](https://github.com/jinqijinqi/poemGPT/tree/main)<br/>
8. (第八周周四截止） Homework 8--Mario play agent简述!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx)<br/>
   [参考pytorch网页](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br/>

---

# All Homework (Homework 1, 2, 3, 7，8 are required; Homework 4, 5, 6 you have to choose one, but all are encouraged)

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
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx)<br/>
   [poemGPT](https://github.com/jinqijinqi/poemGPT/tree/main)<br/>

8. (Due Thursday of Week 8) Homework 7 -- Mario play agent (brief description)!<br/>
   [Submission Template](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/homework1-week1-Pytorch%20Install.docx)<br/>
   [Reference PyTorch webpage](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br/>




