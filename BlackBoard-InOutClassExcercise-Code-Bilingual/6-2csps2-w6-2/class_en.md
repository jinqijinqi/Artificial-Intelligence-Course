# Class — SAME Problem: 3‑Step Object Tracking (Backtracking · FC/AC‑3 · Beam · ICM)

**Setup** Three variables \(X_1,X_2,X_3\in\{0,1,2\}\). Observations \(o=(0,2,2)\).
Observation factors \(O_i(x_i)=\max(0, 2-|x_i-o_i|)\) giving \([2,1,0]\) around \(o_i\).  
Transition factors \(T_i(x_i,x_{i+1}) = 2\) if equal, \(1\) if \(|x_i-x_{i+1}|=1\), else \(0\).

## Tasks
1) **Partial‑weight demo**: with assignment \(x=\{X_1=0\}\), compute dependent factors when extending \(X_2\) by v∈{0,1,2}.  
2) **Forward checking**: after setting \(X_2=2\), cross out from \(X_1,X_3\) any values with \(T_1,T_2=0\). Show remaining domains.  
3) **AC‑3 trace**: starting from empty domains \(\{0,1,2\}\), run AC‑3 once **using zeros in \(T_i\)**; write any values removed.  
4) **Beam (K=2)**: expand level by level; list top‑2 partials and their weights at depths 1,2,3.  
5) **ICM (one pass)**: from initial (0,0,0), update \(X_2\) then \(X_3\) then \(X_1\) using local products; show new assignment and weight.
