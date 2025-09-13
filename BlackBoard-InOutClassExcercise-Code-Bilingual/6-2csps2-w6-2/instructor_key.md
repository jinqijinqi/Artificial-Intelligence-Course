# Instructor Key — 3‑Step Tracking

- **Partial weights** (Task 1) at \(x=\{X_1=0\}\):  
  - Extend \(X_2=0\): \(\delta = O_2(0)\cdot T_1(0,0)=1\cdot 2=2\)  
  - \(X_2=1\): \(\delta = O_2(1)\cdot T_1(0,1)=1\cdot 1=1\)  
  - \(X_2=2\): \(\delta = O_2(2)\cdot T_1(0,2)=2\cdot 0=0\) ⇒ pruned.
- **Forward checking** after \(X_2=2\): remove from \(X_1\{0\}\) (since \(T_1(0,2)=0\)) and from \(X_3\{0\}\) (since \(T_2(2,0)=0\)).
- **AC‑3** prunes any value with no non‑zero support across a neighbor (due to zeros in \(T_i\)); domains shrink accordingly.
- **Beam (K=2)** should keep partials ending \(X_1=0\) and \(X_1=1\) first; final best full often \((0,1,2)\) or \((0,2,2)\) depending on weights.
- **ICM** from (0,0,0): update \(X_2\to1\), \(X_3\to2\), \(X_1\) stays 0 ⇒ (0,1,2) with higher total weight than start.
