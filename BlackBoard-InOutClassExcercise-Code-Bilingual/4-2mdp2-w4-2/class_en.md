# Class — SAME Problem: 3×4 Volcano GridWorld (PI vs VI vs Q-Iteration)

World: 3×4 grid, wall at (2,2); Goal G=(1,4) reward +1 (absorbing); Lava L=(2,4) reward −1 (absorbing);
step reward r_step=−0.04; slip p=0.2; γ=0.99. Actions: U/D/L/R with perpendicular slip.

Tasks:
1) One VI sweep from V^(0)=0 → compute V^(1) (show one full cell calculation).
2) Policy evaluation (uniform-right policy) for 3 iterations; report max_s |V^(t)−V^(t−1)|.
3) Policy improvement using current V; draw greedy arrows.
4) Compare two PI improvement rounds (PE 5 iters each) vs two VI sweeps (values/policies).
5) With residual ε=0.01, give ||V−V*||_∞ bound for γ=0.99.
