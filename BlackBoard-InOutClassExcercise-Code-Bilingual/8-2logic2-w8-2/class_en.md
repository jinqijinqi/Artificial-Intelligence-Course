# Class — SAME Problem: Students–Courses–Knows

**Domain**  
Constants: `alice, bob, cs221, mdp`.  Predicates: `Takes(x,y)`, `Course(y)`, `Covers(y,z)`, `Knows(x,z)`.

**KB (Horn)**  
1) ∀x∀y∀z  (Takes(x,y) ∧ Covers(y,z)) → Knows(x,z)  
2) Takes(alice, cs221)  
3) Covers(cs221, mdp)  
4) Course(cs221)

### Tasks
A) **Propositional (Horn) via MP completeness**  
- Ground the KB (replace variables with constants) to propositional atoms like `Takes_alice_cs221`.  
- Using **forward chaining (MP only)**, derive `Knows(alice, mdp)`. Draw a small derivation DAG.

B) **CNF conversion (practice)**  
Convert \((A∧B)→(C∨D)\) to CNF step by step, then show how a Horn rule becomes a single clause.

C) **Propositional resolution (non-Horn add-on)**  
Augment with `¬Knows(alice, mdp)`. Convert everything to CNF and resolve to the **empty clause** \(\Box\).

D) **FOL MP with unification**  
Without propositionalizing, show the **unifier** θ for premises `{Takes(alice,cs221), Covers(cs221,mdp)}` and rule (1), and derive `Knows(alice,mdp)`.

E) (**Optional, FO-resolution**)  
From clauses `[¬Takes(x,y) ∨ ¬Covers(y,z) ∨ Knows(x,z)]` and `[Takes(alice,cs221)]`, `[Covers(cs221,mdp)]`, perform one **FO-resolution** step to obtain the ground fact.
