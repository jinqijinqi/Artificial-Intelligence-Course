# Class — SAME Problem: Australia Map Coloring

**Variables & domains**  
\(X=\{\mathrm{WA},\mathrm{NT},\mathrm{SA},\mathrm{Q},\mathrm{NSW},\mathrm{V},\mathrm{T}\}\).  
\(\mathrm{Dom}=\{\text{R},\text{G},\text{B}\}\).

**Binary constraints** (neighbors must differ)  
Edges: (WA,NT), (WA,SA), (NT,SA), (NT,Q), (SA,Q), (SA,NSW), (SA,V), (Q,NSW), (NSW,V).  
Tasmania (T) is isolated ⇒ independent.

## Tasks
1) **Factor-graph sketch**: circles for provinces, squares for \([u\neq v]\).  
2) **Backtracking by hand (first few steps)**  
   - Use **MRV** (minimum remaining values) + **degree** tie-break: start with SA (highest degree).  
   - Try SA=R. Forward-check neighbors’ domains; continue 3–4 assignments; document any dead ends & backtracks.  
3) **Decomposition**: after mainland is colored, color T arbitrarily (any of R/G/B).  
4) **Deliverable**: your chosen consistent coloring and the order you assigned variables.

_Note_: Any consistent coloring earns full credit; show your reasoning and pruning.
