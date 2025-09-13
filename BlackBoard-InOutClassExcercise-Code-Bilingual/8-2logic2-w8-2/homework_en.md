# Homework — SAME Problem Programmatically: Logic II Toolkit

Implement a compact toolkit:

1) **Propositional CNF + resolution**
   - AST → CNF (↔/→ elimination, push ¬, distribute).
   - Resolution refutation `entails_via_resolution(KB, f)` that returns a proof trace (pairs of parent clauses → resolvent), or a counterexample if not derived.

2) **Horn forward chaining**
   - Represent rules `(premises -> head)` and facts; derive all entailed atoms and a DAG of justifications.

3) **FOL unification + FO-MP**
   - Implement terms (Const/Var/Fun) and atoms `Pred(name,args)`.
   - `unify(a,b)` with occurs-check (basic); `subst(theta, obj)`.
   - `fo_modus_ponens(facts, rule)` returns new facts via most-general unifier.

4) **Experiments on the class KB**
   - Show: (i) FC derives `Knows(alice,mdp)`; (ii) CNF+resolution refutes KB∪{¬Knows(alice,mdp)}`; (iii) FO-MP derives `Knows(alice,mdp)` without grounding.

(**Optional**) FO-CNF (Skolemization) and one FO-resolution step.

**Deliverables**: code + ≤2-page note (CNF steps, resolution trace, FC graph, FO-MP unifiers).
