# Homework — SAME Problem Programmatically: A Tiny Propositional Logic Engine

Implement a minimal engine to support **Ask/Tell via SAT** and **forward chaining (modus ponens)**.

1) **AST & evaluator**: atoms, Not/And/Or/Imp/Iff; `eval(formula, model)` with model as dict.
2) **Truth-table SAT & entailment**: 
   - `satisfiable(KB)`, `entails(KB, f)` using: `KB ⊨ f` iff `KB ∪ {¬f}` is UNSAT.
   - Return a countermodel if satisfiable (to explain *not* entailed / *not* contradicted).
3) **Forward chaining** with just **modus ponens**, returning all derived formulas; show soundness (derived ⊆ entailed). 
   Discuss incompleteness by example (e.g., KB={Rain, Rain∨Snow → Wet}, cannot derive Wet).
4) **Experiments on the class KB**: verify answers to 2a–2c; list models that witness contingency.
5) **(Optional)** Implement DPLL and/or WalkSAT; compare node counts vs truth-table.

Deliverables: code + ≤2 pages (answers + short discussion of soundness/completeness).
