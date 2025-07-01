# UESTC Artificial Intelligence Course (2024-2025)
# Lecturer: Jin Qi
这是人工智能课程相关信息的展示平台

---

# **教师评分标准：**

* 数学公式、推导清晰、准确（20分）
* 代码规范，运行无误，有适当注释（20分）
* 结果完整，分析透彻（20分）
* 心得体会有独立思考和联系应用（20分）
* 排版规范，结构清晰美观（20分）

---

# 人工智能作业报告模板

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

  $$
  d = \sqrt{(IR_{mean}^{(test)} - IR_{mean}^{(i)})^2 + (VIS_{mean}^{(test)} - VIS_{mean}^{(i)})^2}
  $$
* **kNN判决规则：**
  设 $X_{test}$ 为新样本特征，则预测标签为 $y_{i^*}$：

  $$
  i^* = \arg\min_i d(X_{test}, X_i)
  $$

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

  * 到\[0.7, 0.4]的距离 = sqrt((0.6-0.7)^2 + (0.5-0.4)^2) ≈ 0.14
  * 到\[0.3, 0.8]的距离 = sqrt((0.6-0.3)^2 + (0.5-0.8)^2) ≈ 0.42
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

  $$
  d = \sqrt{(IR_{mean}^{(test)} - IR_{mean}^{(i)})^2 + (VIS_{mean}^{(test)} - VIS_{mean}^{(i)})^2}
  $$
* **kNN prediction rule:**
  Let $X_{test}$ be the new patch feature.
  The predicted label is $y_{i^*}$, where

  $$
  i^* = \arg\min_i d(X_{test}, X_i)
  $$

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

  * Distance to \[0.7, 0.4] = sqrt((0.6-0.7)^2 + (0.5-0.4)^2) = sqrt(0.01 + 0.01) = sqrt(0.02) ≈ 0.14
  * Distance to \[0.3, 0.8] = sqrt((0.6-0.3)^2 + (0.5-0.8)^2) = sqrt(0.09 + 0.09) = sqrt(0.18) ≈ 0.42
    So, the classifier correctly predicts 0 (average fusion).
* This demonstrates that even a simple kNN can automate fusion method selection when given appropriate features and labels.

---

## 5. Reflection and Insights

This assignment helped me connect supervised learning (as presented in AIMA Ch.18) with practical decision-making in image fusion. I learned how to structure a dataset, implement kNN using scikit-learn, and interpret the results both mathematically and programmatically.

The exercise also made me realize that, while kNN is very intuitive and easy to implement, its effectiveness depends heavily on good feature design and sufficient labeled examples. In real-world fusion, we would need a larger and more diverse dataset, possibly including more features (such as edge strength or contrast), and we might experiment with larger values of k or more sophisticated classifiers.

This approach could be extended to more complex fusion tasks, or even as a first step before using deep learning methods. It also showed me the importance of explainable, reproducible AI pipelines in scientific and engineering applications.

---

# 🗓️ **AI + Image Fusion Project: Weekly Assignment Schedule**

---

## **Week 1: Agents and Percepts (AIMA Ch.2)**

**Learning goal:** Understand agents, environment, PEAS framework, and simple agent programs.

### **Assignment:**

1. **Run and understand the Table-driven agent code:**

   ```python
   class TableFusionAgent:
       def program(self, percept):
           return "average"
   agent = TableFusionAgent()
   print(agent.program({'ir': 0.7, 'vis': 0.4}))  # always outputs 'average'
   ```
2. **Mathematical prompt:**

   * *Describe the agent function $` f: P^\star \rightarrow A `$ (where $` P^\star `$ is the set of percept sequences and $` A `$ is the set of actions).*
   * *For this agent, what is the mapping in mathematical terms?*

---

## **Week 2: State Spaces and Search (AIMA Ch.3)**

**Learning goal:** Express fusion as a search; understand BFS.

### **Assignment:**

1. **Run and analyze the BFS code:**

   ```python
   def fuse_average(ir, vis): return 0.5 * ir + 0.5 * vis
   def fuse_max(ir, vis): return np.maximum(ir, vis)
   def image_entropy(img):
       h = np.histogram(img.flatten(), bins=10, range=(0,1))[0]
       p = h / h.sum()
       return -np.sum(p * np.log2(p+1e-10))
   methods = [fuse_average, fuse_max]
   ir = np.ones((3,3))*0.7; vis = np.ones((3,3))*0.3
   for m in methods:
       fused = m(ir, vis)
       if image_entropy(fused) > 0.1:
           print("BFS found:", m.__name__)
           break
   ```
2. **Mathematical prompt:**

   * *Write the state space as a graph/tree. What are the nodes, actions, and the goal test?*
   * *Define entropy mathematically: $H(X) = -\sum_x P(x) \log_2 P(x)$*

---

## **Week 3: Local Search (AIMA Ch.4)**

**Learning goal:** Tune parameters via hill climbing.

### **Assignment:**

1. **Run the hill climbing code:**

   ```python
   w, best = 0.5, 0
   for i in range(11):
       fused = w*ir + (1-w)*vis
       score = image_entropy(fused)
       if score > best:
           best, best_w = score, w
       w += 0.05
   print("Best weight:", round(best_w,2))
   ```
2. **Mathematical prompt:**

   * *Express the iterative improvement as $w_{k+1} = w_k + \Delta w$ if $f(w_{k+1}) > f(w_k)$, where $f$ is the objective function (entropy).*
   * *Show how you update $w$ step by step.*

---

## **Week 4: Constraints (AIMA Ch.6)**

**Learning goal:** Understand constraint satisfaction in fusion (CSP).

### **Assignment:**

1. **Run the CSP enumeration code:**

   ```python
   possible_ws = np.linspace(0, 1, 5)
   for w in possible_ws:
       fused = w*ir + (1-w)*vis
       if (fused <= 1).all():
           print("CSP: Feasible fusion at w=", w)
           break
   ```
2. **Mathematical prompt:**

   * *Formulate the constraint as $0 \leq w \leq 1$ and $(w \cdot ir_{ij} + (1-w) \cdot vis_{ij}) \leq 1$ for all pixels $(i,j)$.*

---

## **Week 5: Logical Agents (AIMA Ch.7)**

**Learning goal:** Use rules for fusion selection.

### **Assignment:**

1. **Run the rule-based code:**

   ```python
   def fusion_rule(ir, vis):
       if ir.mean() > 0.6 and vis.mean() < 0.5:
           return "max"
       else:
           return "average"
   print("Rule selected:", fusion_rule(ir, vis))
   ```
2. **Mathematical prompt:**

   * *Write the logic in propositional form: If $IR_{mean} > 0.6$ and $VIS_{mean} < 0.5$, then action = max fusion; else average.*

---

## **Week 6: Knowledge Representation (AIMA Ch.12)**

**Learning goal:** Use frames/objects to describe image regions.

### **Assignment:**

1. **Run this code:**

   ```python
   regionA = {'modality':'IR', 'edge':'high'}
   print(regionA)
   ```
2. **Mathematical prompt:**

   * *Express a region as a vector of features or a tuple: $regionA = (modality, edge)$.*

---

## **Week 7: Quantifying Uncertainty (AIMA Ch.13–14)**

**Learning goal:** Represent and reason with probabilities.

### **Assignment:**

1. **Run Bayesian CPT code:**

   ```python
   def P_HotObject(ir, vis):
       table = {(1,1):0.99, (1,0):0.8, (0,1):0.7, (0,0):0.05}
       return table[(ir,vis)]
   print("P(HotObject|IR=1,VIS=0):", P_HotObject(1,0))
   ```
2. **Mathematical prompt:**

   * *Write the formula for conditional probability: $P(H|IR, VIS)$ and explain the meaning of the CPT.*

---

## **Week 8: Learning from Examples (AIMA Ch.18)**

**Learning goal:** Train a classifier to select fusion method from data.

### **Assignment:**

1. **Run the kNN code:**

   ```python
   from sklearn.neighbors import KNeighborsClassifier
   X = [[0.7, 0.4], [0.3, 0.8]]
   y = [0, 1]
   clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
   print("Predicted fusion:", clf.predict([[0.6, 0.5]])[0])
   ```
2. **Mathematical prompt:**

   * *Define the kNN classification rule: $\hat{y} = y_{i^*}$ where $i^* = \arg\min_i ||X_i - X_{test}||$.*

---

## **Week 9: Deep Learning (AIMA Ch.19)**

**Learning goal:** Use a neural network to fuse pixel features.

### **Assignment:**

1. **Run the NN code:**

   ```python
   import torch
   import torch.nn as nn
   class SimpleFusionMLP(nn.Module):
       def __init__(self): super().__init__(); self.fc = nn.Linear(2, 1)
       def forward(self, x): return torch.sigmoid(self.fc(x))
   mlp = SimpleFusionMLP()
   input_feat = torch.tensor([[0.7, 0.4]])
   out = mlp(input_feat).item()
   print("NN fused value:", round(out, 3))
   ```
2. **Mathematical prompt:**

   * *Write the neural net equation: $y = \sigma(w_1 \cdot ir + w_2 \cdot vis + b)$, where $\sigma$ is the sigmoid.*

---

## **Week 10: Reinforcement Learning (AIMA Ch.21)**

**Learning goal:** Use Q-values to choose fusion actions.

### **Assignment:**

1. **Run the Q-table code:**

   ```python
   states = [(1,0), (0,1)]
   actions = ["average", "max"]
   Q = {(s,a):0 for s in states for a in actions}
   for s in states:
       for a in actions:
           ir, vis = np.ones((3,3))*s[0], np.ones((3,3))*s[1]
           fused = 0.5*ir + 0.5*vis if a=="average" else np.maximum(ir, vis)
           h = np.histogram(fused.flatten(), bins=10, range=(0,1))[0]
           p = h / h.sum()
           Q[(s,a)] = -np.sum(p * np.log2(p+1e-10))
   print("Q-table:", Q)
   ```
2. **Mathematical prompt:**

   * *Define the Q-value: $Q(s, a) = \text{expected reward}$ for state-action pair.*

---

## **Week 11: Explanation/NLP (AIMA Ch.22)**

**Learning goal:** Generate simple natural language explanations.

### **Assignment:**

1. **Run the explanation code:**

   ```python
   def explain(method, ir, vis):
       return f"Used {method} fusion because IR mean={ir.mean():.2f}, VIS mean={vis.mean():.2f}"
   print(explain("average", ir, vis))
   ```
2. **Mathematical prompt:**

   * *Write a simple mapping: $(ir, vis, method) \rightarrow \text{text explanation}$.*

---

## **Week 12: Ethics and Reflection (AIMA Ch.24)**

**Learning goal:** Reflect on transparency and fairness.

### **Assignment:**

1. **Prompt:**

   * *Write a short paragraph: “What ethical considerations arise in using automated fusion algorithms for safety-critical tasks?”*

---








# 所有作业（作业1，2，3，7必做，作业4，5，6可以选一个，但鼓励大家都选）
1. (第二周周四截止）Homework 1--Pytorch Installation 简述!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%201-%E5%91%A81-pytorch%E5%AE%89%E8%A3%85.docx)<br/>
   [参考pytorch网页](https://pytorch.org/get-started/locally/)<br/>
2. (第三周周四截止）Homework 2--Learning PyTorch with Examples简述内容!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%202-%E5%91%A82-Learning%20PyTorch%20with%20Examples.docx)<br/>
   [参考pytorch网页](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)<br/>
3. (第四周周四截止）Homework 3--What is torch.nn really?简述内容!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%203-%E5%91%A83-torch-nn.docx)<br/>
   [参考pytorch网页](https://pytorch.org/tutorials/beginner/nn_tutorial.html)<br/>
4. (第六周周四截止）Homework 4--图像识别简述内容!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%204(optional)%E5%91%A85-%E5%9B%BE%E5%83%8F%E8%AF%86%E5%88%AB.docx)<br/>
   [参考pytorch网页](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)<br/>
5. (第六周周四截止） Homework 5--语音识别简述内容!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%205(optional)-%E5%91%A85-%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB.docx)<br/>
   [参考pytorch网页](https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html)<br/>
6. (第六周周四截止） Homework 6--文本识别简述内容!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%206(optional)-%E5%91%A85-%E6%96%87%E6%9C%AC%E8%AF%86%E5%88%AB.docx)<br/>
   [参考pytorch网页](https://pytorch.org/tutorials/beginner/translation_transformer.html)<br/>
6. (第八周周四截止） Homework 6--Mario play agent简述!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%207-%E5%91%A86-%E5%91%A88-%E8%AF%BE%E7%A8%8B%E6%9C%80%E7%BB%88%E6%8A%A5%E5%91%8A-%E9%A9%AC%E9%87%8C%E5%A5%A5%E7%8E%A9%E5%AE%B6.docx)<br/>
   [参考pytorch网页](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br/>
   
   


