# Board (EN) — MDPs 2 (Policy Iteration · Bellman Operators · Contraction · Q-Iteration)

**Bellman operators**
- Policy operator: (T^π V)(s)=Σ_{s'} T(s,π(s),s') [ R(s,π(s),s') + γ V(s') ]  (γ-contraction in ||·||_∞)
- Optimality: (T^* V)(s)=max_a Σ_{s'} T(s,a,s') [ R(s,a,s') + γ V(s') ]  (γ-contraction)

**Error bound**
If ||T^*V − V||_∞ ≤ ε ⇒ ||V − V^*||_∞ ≤ ε/(1−γ).

**Policy Iteration (PI)**
repeat:  Policy Evaluation → Policy Improvement (greedy wrt V^π).  Finite MDP ⇒ finite termination.

**Q-Iteration**
Q_{t+1}(s,a)=Σ_{s'} T(s,a,s') [ R(s,a,s') + γ max_{a'} Q_t(s',a') ] ;  π(s)=argmax_a Q(s,a).

**Practical**
- Asynchronous (in-place) VI; prioritized sweeping.
- Residual stopping with Bellman residual; optimistic init to speed policy improvement.
