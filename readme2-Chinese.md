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




