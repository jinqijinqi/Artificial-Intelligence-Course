# 板书（中文）— MDPs 1（概览 · 建模 · 策略评估 · 价值迭代）

## MDP（有限、回合式）定义
状态 **S**、起始 **s₀**、动作 **A(s)**；  
转移 **T(s,a,s')=P(s'|s,a)**（对固定 \(s,a\) 有 \(\sum_{s'}T=1\)）；  
奖励 **R(s,a,s')**；终止 **IsEnd(s)**；折扣 **γ∈[0,1]**。

搜索是特例：
- 确定性：**Succ(s,a)** ⇒ **T(s,a,s')=1[s'=Succ(s,a)]**；
- 成本与奖励互换：**R = −Cost**。

## 策略、价值与 Q 值
策略 **π: S→A**。  
价值（期望折扣效用）：
\[
V^{\pi}(s)=
\begin{cases}
0, & \text{若 IsEnd}(s)\\
\sum_{s'} T(s,\pi(s),s')\,[R(s,\pi(s),s')+\gamma V^{\pi}(s')], & \text{否则}
\end{cases}
\]
Q 值：
\[
Q^{\pi}(s,a)=\sum_{s'} T(s,a,s')\,[R(s,a,s')+\gamma V^{\pi}(s')].
\]

## 策略评估（迭代）
\(V^{(0)}\equiv0\)，迭代
\[
V^{(t)}(s)\leftarrow \sum_{s'} T(s,\pi(s),s')\,[R(s,\pi(s),s')+\gamma V^{(t-1)}(s')].
\]
收敛准则：\(\max_s |V^{(t)}-V^{(t-1)}|\le\varepsilon\)。  
时间 \(O(t_{\text{PE}}\cdot |S|\cdot S')\)。

## 最优性与价值迭代
\[
Q^*(s,a)=\sum_{s'} T(s,a,s')[R(s,a,s')+\gamma V^*(s')],\quad
V^*(s)=\begin{cases}
0,&\text{若 IsEnd}(s)\\
\max_a Q^*(s,a),&\text{否则。}
\end{cases}
\]
迭代
\[
V^{(t)}(s)\leftarrow \max_a\sum_{s'} T(s,a,s')[R(s,a,s')+\gamma V^{(t-1)}(s')].
\]
策略：\(\pi^*(s)=\arg\max_a Q^*(s,a)\)。  
时间 \(O(t_{\text{VI}}\cdot |S|\cdot |A|\cdot S')\)。

## 收敛（充分条件）
若 **γ<1** 或 **MDP 图无环**，策略评估与价值迭代收敛；可将 \(1-γ\) 看作每步以该概率**终止**。
