# Board (EN) — Logic I (Propositional Logic · Semantics · SAT · Inference · Sound/Complete)

## Syntax (formulas)
Atoms: A,B,C,...  Connectives: ¬ ∧ ∨ → ↔.
If f,g are formulas, so are ¬f, (f∧g), (f∨g), (f→g), (f↔g).

## Semantics (models & interpretation)
Model w: truth assignment to atoms.  Interpretation I(f,w) ∈ {0,1}; define recursively:
- Base: I(p,w)=w(p) for atom p.
- Recursion (by truth tables): ¬, ∧, ∨, →, ↔.
Formula denotes set of models M(f)={w: I(f,w)=1}.  KB (set of formulas) denotes M(KB)=⋂_{f∈KB} M(f).

## Ask/Tell via entailment/contradiction/contingency
- **Entailment** KB ⊨ f  ⇔  M(KB) ⊆ M(f).
- **Contradiction** KB ⊨ ¬f  ⇔  M(KB) ∩ M(f)=∅.
- **Contingent** otherwise (both f, ¬f compatible).

## Reduce to SAT (model checking)
Satisfiable: M(KB)≠∅.  Decide Ask/Tell by SAT:
- KB ⊨ f  ⇔  KB ∪ {¬f} is UNSAT
- KB ⊨ ¬f ⇔  KB ∪ {f}  is UNSAT
Propositional SAT ≈ CSP: symbols ↔ variables; formulas ↔ constraints.

## Algorithms
Exact model checking: truth-table / DPLL; Local search: WalkSAT.

## Inference rules on syntax
Example (modus ponens): from p and (p→q), derive q.
**Soundness**: rules derive only truths ({f: KB ⊢ f} ⊆ {f: KB ⊨ f}).  
**Completeness**: rules derive all truths ({f: KB ⊢ f} ⊇ {f: KB ⊨ f}).
