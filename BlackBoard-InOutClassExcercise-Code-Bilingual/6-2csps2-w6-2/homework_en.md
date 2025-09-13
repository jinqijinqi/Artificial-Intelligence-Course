# Homework — SAME Problem Programmatically: Tracking with Backtracking/AC‑3 · Beam · ICM

1) **Data model** Build factors: domains {0,1,2}; \(O_i(x)=\max(0,2-|x-o_i|)\) for \(o=(0,2,2)\);
\(T_i(x,y)=2\,[x=y]+1\,[|x-y|=1]\).

2) **Backtracking** with MCV/MRV + LCV + **forward checking**, computing partial weights via dependent factors.

3) **AC‑3** (binary, zero‑support pruning) and plug into backtracking (run AC‑3 on domain updates).

4) **Beam search** (K∈{1,2,3}) on partial weights; report best full assignment and weight; compare node counts vs backtracking.

5) **ICM** starting from 5 random initializations; report best weight reached and frequency of local optima.

6) **(Optional)** Add soft constraints on “acceleration” \(A_i=|x_{i+1}-2x_i+x_{i-1}|\) with factor \(\exp(-\lambda A_i)\); redo (2)-(5).

**Deliverables**: code + ≤2‑page report (tables: nodes/weights; short discussion).
