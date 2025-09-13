# Homework — SAME Problem Programmatically: Australia Map Coloring (CSP)

Implement a tiny CSP toolkit and solve the Australia map:

1) **CSP core**  
   - Structures for `variables`, `domains`, and `constraints` (binary predicates).  
   - A function `neighbors(var)`.

2) **Backtracking search**  
   - Variable ordering: **MRV + degree tie-break**.  
   - Value ordering: try least-constraining value (LCV).  
   - **Forward checking** (propagate domain wipe-outs) with undo stack.

3) **Logs & metrics**  
   - Count nodes, backtracks; print the assignment order and domains after forward checking.

4) **Experiments**  
   - Compare plain backtracking vs +MRV vs +MRV+LCV vs +MRV+LCV+FC (forward checking).  
   - Report node counts on Australia; optionally add **N-Queens (N=8)** as a second CSP.

5) **(Optional)**  
   - Implement **AC-3** and compare with forward checking.  
   - Add **event scheduling** (Formulation 1) and solve a toy instance.

**Deliverables**: code + ≤2-page note (tables with node counts and brief analysis).
