# Instructor Key — Logic II (Students–Courses–Knows)

- **A (Horn + MP)**: Forward chaining derives `Knows_alice_mdp` from facts 2–3 and rule (1).  
- **B (CNF)**: \((A∧B)→(C∨D)\)  
  → \(¬(A∧B)∨(C∨D)\)  
  → \((¬A∨¬B∨C∨D)\) (already a clause); the Horn rule (1) becomes \([¬Takes(x,y) ∨ ¬Covers(y,z) ∨ Knows(x,z)]\).
- **C (Resolution)**: CNF(KB) ∪ {¬Knows_alice_mdp} derives \(\Box\); thus KB ⊨ Knows_alice_mdp.
- **D (FOL MP)**: Unifier θ = {x/alice, y/cs221, z/mdp}; conclude `Knows(alice,mdp)`.
- **E (FO-Resolution)**: Resolve `[¬Covers(y,z)∨…]` with `Covers(cs221,mdp)` etc., after unifying \(y\mapsto cs221, z\mapsto mdp\), to obtain the ground head.

**Complexity**: MP on Horn is linear; resolution may add longer clauses—can blow up exponentially; FOL is semi-decidable.
