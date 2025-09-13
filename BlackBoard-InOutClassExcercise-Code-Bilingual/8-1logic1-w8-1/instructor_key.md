# Instructor Key — Rain–Wet–Slippery KB

**Derived by Modus Ponens**: from `Rain` and `Rain→Wet` derive `Wet`; then with `Wet→Slippery` derive `Slippery`.
`Rain→Slippery` is **entailed** semantically (KB ⊨ Rain→Slippery) though **not produced** by MP alone (illustrates incompleteness of that single rule).

**Ask/Tell (via SAT reasoning)**
- (a) KB ⊨ **Wet**: Yes, since KB ∪ {¬Wet} is UNSAT (any model of KB makes Wet true).
- (b) KB ⊨ **Rain → Slippery**: Yes (every model of KB has Rain ⇒ Wet ⇒ Slippery).
- (c) KB ⊨ ¬(**¬Rain**): i.e., **¬Rain** contradicts KB because KB already contains Rain.

**Contingency**: `Snow` is contingent—both KB∪{Snow} and KB∪{¬Snow} are satisfiable (doesn’t affect existing implications).

**Shrink M(KB) if Tell[¬Wet]**: This contradicts KB (since KB entails Wet), so the correct response is “don’t believe that” / reject update.

**Notes**: Truth-based checks line up with semantics; forward chaining with only MP is sound but not complete—needs stronger rules (e.g., resolution) or formula restrictions (e.g., Horn).
