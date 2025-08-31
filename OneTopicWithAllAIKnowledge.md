---
marp: false
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

   * *Write a simple mapping: $(ir, vis, method) 
   * \rightarrow \text{text explanation}$.*

---

## **Week 12: Ethics and Reflection (AIMA Ch.24)**

**Learning goal:** Reflect on transparency and fairness.

### **Assignment:**

1. **Prompt:**

   * *Write a short paragraph: “What ethical considerations arise in using automated fusion algorithms for safety-critical tasks?”*

---

