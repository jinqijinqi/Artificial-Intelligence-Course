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



---

## **Instructor's  Grading:**

* Mathematical formulation: clear, correct (20)
* Code: runs, documented, reproducible (20)
* Results & analysis: matches theory, well explained (20)
* Reflection: insightful, connects AI to real task (20)
* Formatting: neat, professional (20)

---

# Artificial Intelligence Assignment Report Template

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

---------------------------
下面是**严格按照《Artificial Intelligence: A Modern Approach》经典教材章节**编排的**周项目/作业清单与模板**。每周作业包含**姓名、学号、实验题目、实验内容（代码）、数学推导（公式/理论）、实验结果、分析与总结**等模块，适合标准本科/研究生人工智能入门课程教学。
-----
## 📙[请下载作业空模板.docx (请恰当修改程序所用的原始数据，保证与参考代码所用数据不一致！！！）](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx) <br/>

## 课堂及课后作业

## **Week 1：机器学习1（回归与分类）**

## 课堂练习

1. **（线性回归，平方损失，GD 1 步）**  
  数据集 $(x,y)\in\{(1,1),(2,3),(4,3)\}$ ， $\phi(x)=[1,x]$ ，初始  $w^{(0)} =[0,0]$ ，步长  $\eta=0.1$ 。

   1) 计算 $\nabla \text{TrainLoss}(w^{(0)})$。

   2) 更新 $w^{(1)}=w^{(0)}-\eta\nabla \text{TrainLoss}(w^{(0)})$ 。
   3) 算  $\nabla\text{TrainLoss}(w^{(1)})$ 与  $w^{(2)}$ 。

2. **（线性分类，合页损失，次梯度 1 步）**  
样本 $(x,y)\in\{([0,2],+1),([-2,0],+1),([1,-1],-1)\}$ ，  $\phi(x)=[x_1,x_2]$ ，当前 $w=[0.5,1.0]$ 。  
   1)  分别计算每个样本的合页损失与次梯度。
   2)  求平均得到训练损失的（次）梯度；解释为何有的为 0。
   3)  用 $\eta=0.1$ 做一次更新。  
**提交**：关键式子 + 数值结果（保留两位小数）。

## 课后练习 — 同题的程序实现

1. **（线性回归：平方损失 + GD/SGD）**  
- 实现 `fit_linear_gd(X, y, lr=0.1, epochs=200)`（$\phi(x)=[1,x]$ 或通用 $\phi$ ）。  
- 打印/绘制训练损失曲线；报告最终 $w$。  
**基础**：与参考一致。  
**挑战**：加入 SGD（或小批）并比较与 GD 的速度。

2. **（线性分类：合页损失的次梯度）**  
- 实现 `fit_hinge_gd(X, y, lr=0.1, epochs=200)`；标签取 $\{\pm1\}$。  
- 报告训练合页损失与 0–1 准确率。  
**基础**：与参考一致。  
**挑战**：加入 L2 正则。

**提交**：代码 + 简短报告（≤1 页）含曲线/表格。  
**评分（基础/挑战）**：正确性 60，工程 20，分析 20。

## 参考代码— 同题的程序实现
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

···

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

 # 1) 线性回归 · 课堂版

* 模型： $f_w(x)=w^T[1,x]$ 
* 数据： $(-2,0),(-1,0.5),(0,1),(1,1.5),(2,2)$（即 $y=1+0.5x$）
* 损失： $\frac1n\sum(f_w(x)-y)^2$
* 训练：批量梯度下降（SGD 优化器）

**任务:**
写出总损失函数和相应的梯度下降迭代公式

**运行：**

```bash
python regression_classroom.py
```

**期望输出（示例）：**

* 学到的权重 $w=[w_1,w_2]\approx[1.0,0.5]$
* 训练 MSE 接近 0
* 对训练点的预测与真值一致

---

## 2) 线性分类 · 课堂版（合页损失）

* 特征： $\phi(x)=[x_1,x_2]$ （无偏置）
* 数据（严格同 PDF）： $[0,2]\to +1,\ [-2,0]\to +1,\ [1,-1]\to -1$
* 打分： $s(x)=w^\top\phi(x)$， 预测  $\mathrm{sign}(s)$
* 损失： $\frac1n\sum\max(0,1-y\,s)$

**任务:**
写出总损失函数和相应的梯度下降迭代公式

**运行：**

```bash
python classification_classroom.py
```

**期望输出（示例）：**

* 学到的权重 $w$ 能让三点全部间隔 $\ge 1$，训练合页损失 $\to 0$
* 训练集预测为 $[+1,+1,-1]$，精度 100%

---

# 课后练习（可调学习率/轮数，便于做实验报告）

## 3) 线性回归 · 课后版

* 与课堂版相同数学设定
* **命令行参数**：`--lr` 学习率，`--epochs` 轮数
* 任务：

  1. 在相同轮数下，画损失关于学习率的曲线，讨论学习率对收敛的影响（较小/较大对比）
  2. 在相同学习率下，画损失关于迭代轮数的曲线，讨论迭代轮数对收敛的影响

**运行示例：**

```bash
python regression_homework.py --lr 0.05 --epochs 500
```

---

## 4) 线性分类 · 课后版（合页损失）

* 
* **命令行参数**：`--lr` 学习率，`--epochs` 轮数
* 任务建议：

  1. 在相同轮数下，画损失关于学习率的曲线，讨论学习率对收敛的影响（较小/较大对比）
  2. 在相同学习率下，画损失关于迭代轮数的曲线，讨论迭代轮数对收敛的影响

**运行示例：**

```bash
python classification_homework.py --lr 0.2 --epochs 200
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




