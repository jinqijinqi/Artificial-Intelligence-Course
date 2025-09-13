# Board (EN) — CSPs 1 (Modeling · Factor Graphs · Weights · Examples)

## Factor graph (variables & factors)
Variables \(X=(X_1,\dots,X_n)\), domains \(\mathrm{Dom}_i\).
Factors \(f_1,\dots,f_m\) (non-negative), each has **scope** (subset of variables) and **arity** (scope size).
Constraints are 0/1 factors.

## Assignment weight & objective
For assignment \(x=(x_1,\dots,x_n)\):
\[
\mathrm{Weight}(x)=\prod_{j=1}^m f_j(x).
\]
**Consistent** iff \(\mathrm{Weight}(x)>0\).
CSP objective: \(\arg\max_x \mathrm{Weight}(x)\); **satisfiable** iff \(\max_x \mathrm{Weight}(x) > 0\).

## Modeling pattern
- Choose variables & domains so that an assignment **is** a solution candidate.
- Translate desiderata into **local factors** (unary/binary preferred).
- Keep graph small: fewer variables, smaller domains, low-arity factors.

## Examples (from slides)
- Map coloring (binary \([u\neq v]\) constraints).
- Voting toy (weighted factors → max-weight assignment).
- Object tracking (transition + observation factors).
- Event scheduling (two formulations: event-centric vs slot-centric).
- Program verification (equation constraints + negated spec).
