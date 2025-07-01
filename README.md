# UESTC Artificial Intelligence Course (2024-2025)
# Lecturer: Jin Qi
这是人工智能课程相关信息的展示平台

Absolutely! Here’s a **project plan for a semester/quarter**, organized as **weekly assignments**—each aligned to one or two AIMA chapters and mapped to a key image fusion concept, with **minimal code demo**, **mathematical explanation prompt**, and a **sample writeup template** for students.
The goal:

* Students understand and run each code snippet.
* For each algorithm, they explain the *underlying math* (e.g., formula, search tree, probability equation) and relate it to the code.
* They complete a short written report using a unified template.

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

   * *Describe the agent function $' f: P^\star \rightarrow A $' (where $ P^* $ is the set of percept sequences and $ A $ is the set of actions).*
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

---

# 📝 **Sample Student Writeup Template**

---

**Assignment Week X: \[Topic, e.g., Search or Deep Learning]**

1. **Code Output (screenshot or copy):**

   ```
   [Paste output or screenshot here]
   ```

2. **Math Explanation:**

   * *Write the relevant algorithm mathematically:*
   * \[E.g.,\* “Entropy: $H(X) = -\sum_x P(x)\log_2 P(x)$”\*]
   * *Describe how the code implements this.*

3. **Reflection:**

   * *What did you learn? How does the AI method relate to the fusion problem?*
   * *Where else might you apply this idea?*

---

# ⭐️ **How to Use**

* Each week: students **run, understand, and explain** a concept from both code and math.
* The template ensures students connect **AIMA theory → algorithm → code → application**.
* You may expand each into full lab, homework, or as part of a capstone project.

---

If you want this as a formatted Jupyter notebook with markdown and editable prompts for each week, **just say the word!**





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
   
   


