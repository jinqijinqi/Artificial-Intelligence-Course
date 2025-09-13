from typing import Dict, List, Tuple, Iterable

State = str
Action = str
Transition = Tuple[State, float, float]  # (next_state, prob, reward)

# --------- MDP base ---------
class MDP:
    def states(self) -> Iterable[State]: ...
    def actions(self, s: State) -> Iterable[Action]: ...
    def transitions(self, s: State, a: Action) -> Iterable[Transition]:
        """Yield (s', prob, reward). Probabilities over s' must sum to 1 for each (s,a)."""
        ...
    def is_end(self, s: State) -> bool: ...
    @property
    def start_state(self) -> State: ...

# --------- Dice Game MDP ---------
class DiceMDP(MDP):
    """
    States: 'in', 'end'
    Actions at 'in': 'stay' or 'quit'; 'end' has no actions.
    Rewards: stay gives +4 then stochastic termination; quit gives +10 then terminate.
    """
    def __init__(self): pass
    def states(self): return ['in', 'end']
    def actions(self, s): return ['stay','quit'] if s == 'in' else []
    def transitions(self, s, a):
        if s == 'end': return []
        if a == 'quit':
            yield ('end', 1.0, 10.0)
        elif a == 'stay':
            yield ('in', 2/3, 4.0)
            yield ('end', 1/3, 4.0)
        else:
            raise ValueError(a)
    def is_end(self, s): return s == 'end'
    @property
    def start_state(self): return 'in'

# --------- Policy evaluation ---------
def policy_evaluation(mdp: MDP, policy: Dict[State, Action], gamma: float=1.0, eps: float=1e-8,
                      max_iters: int=10_000) -> Dict[State, float]:
    V = {s: 0.0 for s in mdp.states()}
    for t in range(max_iters):
        delta = 0.0
        V_prev = V.copy()
        for s in mdp.states():
            if mdp.is_end(s):
                V[s] = 0.0
                continue
            a = policy[s]
            val = 0.0
            for sp, p, r in mdp.transitions(s, a):
                val += p * (r + gamma * V_prev[sp])
            delta = max(delta, abs(val - V_prev[s]))
            V[s] = val
        if delta <= eps:
            break
    return V

# --------- Value iteration ---------
def value_iteration(mdp: MDP, gamma: float=1.0, eps: float=1e-8, max_iters: int=10_000):
    V = {s: 0.0 for s in mdp.states()}
    for t in range(max_iters):
        delta = 0.0
        V_prev = V.copy()
        for s in mdp.states():
            if mdp.is_end(s):
                V[s] = 0.0
                continue
            best = float('-inf')
            for a in mdp.actions(s):
                q = 0.0
                for sp, p, r in mdp.transitions(s, a):
                    q += p * (r + gamma * V_prev[sp])
                if q > best:
                    best = q
            delta = max(delta, abs(best - V_prev[s]))
            V[s] = best
        if delta <= eps:
            break
    # greedy policy
    policy = {}
    Q = {}
    for s in mdp.states():
        if mdp.is_end(s): continue
        best_a, best_q = None, float('-inf')
        for a in mdp.actions(s):
            q = 0.0
            for sp, p, r in mdp.transitions(s, a):
                q += p * (r + gamma * V[sp])
            Q[(s,a)] = q
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a
    return V, policy, Q

# --------- Demo ---------
if __name__ == "__main__":
    mdp = DiceMDP()
    # Policy: always stay
    pi_stay = {'in':'stay'}
    V_stay = policy_evaluation(mdp, pi_stay, gamma=1.0, eps=1e-10)
    print("V^pi(stay) at 'in':", V_stay['in'])  # 12 (γ=1)

    # Policy: always quit
    pi_quit = {'in':'quit'}
    V_quit = policy_evaluation(mdp, pi_quit, gamma=1.0)
    print("V^pi(quit) at 'in':", V_quit['in'])  # 10

    # Value iteration
    Vstar, pistar, Q = value_iteration(mdp, gamma=1.0, eps=1e-10)
    print("V* at 'in':", Vstar['in'], "pi*:", pistar['in'])
