# 板书（中文）— 对弈搜索 II（机会结点 · 期望极大 · 估值 · 风险偏好 · 采样）

## 随机博弈树
除 MAX/MIN 外还有 **机会结点**，具已知分布 \(P(o\mid s)\)。终止效用 \(U(s)\)（对 MAX）。

## 期望极大（Expectimax）递推
\[
V(s)=
\begin{cases}
U(s), & s \text{ 为终局}\\
\max_{a\in A(s)} V(\mathrm{Succ}(s,a)), & s \text{ 为 MAX}\\
\min_{a\in A(s)} V(\mathrm{Succ}(s,a)), & s \text{ 为 MIN}\\
\sum_o P(o\mid s)\, V(\mathrm{Succ}(s,o)), & s \text{ 为机会结点}
\end{cases}
\]

## 深度截断与估值
限制深度 \(D\)：叶子用 \(\mathrm{Eval}(s)=w^\top\phi(s)\) 代替。  
特征需贴合任务（威胁、距离、风险）；用小范围网格搜索/回归调权。

## 风险偏好（效用）
把收益 \(x\) 换成效用 \(U(x)\)。  
- **风险中性**：\(U(x)=x\)；  
- **风险厌恶**：凹函数 \(U(x)=\sqrt{x_+}-\lambda x_-\)。  
决策仍为**最大化期望效用**（MAX）/最小化（MIN）。

## Alpha–Beta？
一般对机会结点**无**安全 αβ 剪枝（max/min 与期望不可互换）。  
实践：**概率截断**、结果 beam、**蒙特卡洛 expectimax**（采样叶子）。

## 蒙特卡洛与迭代加深
- 在机会结点**采样**：用 \(k\) 次样本估计 \(\mathbb{E}[V]\)。  
- **迭代加深**：D=1,2,…；利用上一层主变排序。
