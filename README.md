# UESTC Artificial Intelligence Course (2024-2025)
# Lecturer: Jin Qi
这是人工智能课程相关信息的展示平台

---

# **教师评分标准：**

* **数学公式**：推导清晰、准确（20分）
* **代码**：规范，运行无误，有适当注释（20分）
* **结果与分析**：结果完整，分析透彻（20分）
* **心得与思考**：有独立思考和AI联系实际应用（20分）
* **格式**：排版规范，结构清晰美观（20分）

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
7. (第八周周四截止） Homework 6--Mario play agent简述!<br/>
   [提交作业模板](https://github.com/jinqijinqi/Artificial-Intelligence-Course/blob/main/homework/%E4%BD%9C%E4%B8%9A%207-%E5%91%A86-%E5%91%A88-%E8%AF%BE%E7%A8%8B%E6%9C%80%E7%BB%88%E6%8A%A5%E5%91%8A-%E9%A9%AC%E9%87%8C%E5%A5%A5%E7%8E%A9%E5%AE%B6.docx)<br/>
   [参考pytorch网页](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br/>
   
   


