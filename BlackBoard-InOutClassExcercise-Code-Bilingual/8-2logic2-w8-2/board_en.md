# Board (EN) — Logic II (Horn · CNF · Resolution · FOL · Unification · FO-Resolution)

## Horn & Modus Ponens (MP)
- **Definite clause**: \((p_1 \land \cdots \land p_k) \to q\).  **Horn clause**: definite or **goal** \((p_1 \land \cdots \land p_k) \to \bot\).
- **MP rule**: from \(p_1,\dots,p_k,(p_1\land\cdots\land p_k)\to q\) derive \(q\).
- **Theorem (completeness on Horn)**: if KB has only Horn clauses and symbol \(p\) is entailed, MP derives \(p\).
- **Tradeoff**: Horn + MP ⇒ linear-time forward chaining; general propositional logic needs stronger rules.

## Resolution (propositional)
- Write implication as disjunction: \(A\to C \equiv \lnot A \lor C\); a **clause** is a disjunction of **literals** (e.g., \(\lnot A \lor B \lor \lnot C\)).
- **Resolution rule**: \((f_1\vee\cdots\vee f_n \vee p),(\lnot p \vee g_1\vee\cdots\vee g_m)\Rightarrow (f_1\vee\cdots\vee f_n\vee g_1\vee\cdots\vee g_m)\).
- **CNF**: conjunction of clauses. **Entailment by refutation**: KB ⊨ \(f\) iff KB ∪ {¬\(f\)} derives **empty clause** \(\Box\).

### CNF conversion steps
Eliminate ↔, then →; push ¬ inward (de Morgan); remove double ¬; **distribute** ∨ over ∧; (optionally standardize names).  

## First-Order Logic (FOL) essentials
- **Terms**: constants, variables, and functions.  **Atoms**: predicates on terms.  **Quantifiers**: ∀, ∃.
- **Substitution**: Subst[\(\theta\), \(f\)] replaces variables with terms.  **Unification**: Unify[\(f,g\)] = most general θ with Subst[θ,f]=Subst[θ,g].
- **FO-MP**: unify premises \(a'_1\!\land\!\cdots\!\land\!a'_k\) with \(a_1\!\land\!\cdots\!\land\!a_k\) to get θ, conclude Subst[θ,\(b\)].
- **FO to CNF**: eliminate →/↔, push ¬, **Skolemize** (replace ∃ with Skolem functions of surrounding ∀ vars), drop ∀, distribute, get clauses.
- **FO-resolution**: as propositional resolution but first **unify** complementary literals; apply θ to resolvent.

## Complexity snapshot
- MP on Horn: linear-time (each symbol derived once).  
- Resolution (prop/FOL): in worst case exponential; FOL is **semi-decidable** (may not halt when not entailed).
