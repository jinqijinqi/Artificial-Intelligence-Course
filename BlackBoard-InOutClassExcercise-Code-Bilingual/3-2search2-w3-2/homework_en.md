# Homework — A* & Relaxed Heuristics (Same Problem)

Implement:
1) **A\***(`astar(problem, h)`) as UCS on modified cost: `cost' = cost + h(s') - h(s)`; return optimal path/cost and node counts.
2) **Relaxed heuristic** \(h_{\text{rel}}\): compute \(\mathrm{FutureCost}_{\text{rel}}(\text{loc})\) via UCS on the **reversed relaxed** problem (don’t stop early).
3) Baselines: \(h_0=0\); \(h_{\text{walk}}(\text{loc})=n-\text{loc}\).
4) Combine: \(h_{\max}=\max(h_{\text{rel}},h_{\text{walk}})\).

**Experiments** \(n\in\{50,200,1000\}\):
- Optimal cost from A* must match UCS;  
- **Expanded states** & **peak frontier**: UCS vs A* with \(h_0,h_{\text{walk}},h_{\text{rel}},h_{\max}\);  
- Verify programmatically that all expanded edges satisfy `cost' ≥ 0` (consistency).  
- Optional: plot expansions vs heuristic.

**Optional (Structured Perceptron)**: given target paths, learn walk/tram costs and re-run A*.
