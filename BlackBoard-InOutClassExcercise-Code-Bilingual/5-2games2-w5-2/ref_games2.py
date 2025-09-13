from typing import Dict, Tuple, Callable
import random, math

# --------- Dice-to-21 ---------
TARGET = 21

def utility_risk_neutral(x: int) -> float:
    return float(x)

def utility_risk_averse(x: int, lam: float=2.0) -> float:
    xp = max(x, 0); xn = max(-x, 0)
    return math.sqrt(xp) - lam * xn

def succ_max(s: int):
    """Return list of actions at MAX state s."""
    return ["stop", "roll"]

def succ_chance(s: int, a: str):
    """Return list of outcomes (prob, next_state, immediate_reward_flag)."""
    if a == "stop":
        # terminal handled by is_terminal/utility; no chance children
        return []
    # roll: outcomes 1..6, uniform
    outcomes = []
    for x in range(1,7):
        sp = s + x
        outcomes.append((1/6.0, sp))
    return outcomes

def is_terminal(s: int, last_action: str=None) -> bool:
    if last_action == "stop": return True
    return s > TARGET

def terminal_payoff(s: int, last_action: str=None) -> int:
    if last_action == "stop":
        return s
    # bust
    return -10

# --------- Expectimax ---------
def expectimax(state: int, depth: int,
               utility_fn: Callable[[int], float]=utility_risk_neutral,
               eval_fn: Callable[[int], float]=lambda s: s,
               last_action: str=None):
    """
    Returns (value, best_action, nodes) from MAX perspective under chance.
    Depth-limited: at depth==0, use eval_fn(state).
    """
    nodes = 0
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def max_node(s: int, d: int):
        nonlocal nodes
        nodes += 1
        if is_terminal(s, None):  # bust state (s>TARGET)
            return utility_fn(terminal_payoff(s, None)), None
        if d == 0:
            return eval_fn(s), None
        best_val = -1e18
        best_act = None
        for a in succ_max(s):
            if a == "stop":
                val = utility_fn(terminal_payoff(s, "stop"))
            else:
                val = chance_node(s, a, d-1)
            if val > best_val:
                best_val, best_act = val, a
        return best_val, best_act

    @lru_cache(maxsize=None)
    def chance_node(s: int, a: str, d: int):
        nonlocal nodes
        nodes += 1
        # expectation over outcomes
        ev = 0.0
        for p, sp in succ_chance(s, a):
            if is_terminal(sp, None):
                ev += p * utility_fn(terminal_payoff(sp, None))
            elif d == 0:
                ev += p * eval_fn(sp)
            else:
                val, _ = max_node(sp, d)
                ev += p * val
        return ev

    val, act = max_node(state, depth)
    return val, act, nodes

# --------- Sampling Expectimax ---------
def expectimax_sample(state: int, depth: int, k: int=4,
                      utility_fn: Callable[[int], float]=utility_risk_neutral,
                      eval_fn: Callable[[int], float]=lambda s: s):
    """
    Monte Carlo at chance nodes: sample k outcomes (with replacement).
    """
    nodes = 0
    def max_node(s: int, d: int):
        nonlocal nodes
        nodes += 1
        if is_terminal(s, None):
            return utility_fn(terminal_payoff(s, None)), None
        if d == 0:
            return eval_fn(s), None
        best_val, best_act = -1e18, None
        for a in succ_max(s):
            if a == "stop":
                val = utility_fn(terminal_payoff(s, "stop"))
            else:
                val = chance_node(s, a, d-1)
            if val > best_val:
                best_val, best_act = val, a
        return best_val, best_act

    def chance_node(s: int, a: str, d: int):
        nonlocal nodes
        nodes += 1
        ev = 0.0
        for _ in range(k):
            x = random.randint(1,6)
            sp = s + x
            if is_terminal(sp, None):
                ev += utility_fn(terminal_payoff(sp, None))
            elif d == 0:
                ev += eval_fn(sp)
            else:
                v, _ = max_node(sp, d)
                ev += v
        return ev / k

    return max_node(state, depth) + (nodes,)

# --------- Simple evals ---------
def eval_linear(s: int):
    # features: (score, near_target, far)
    return 1.0*s + 5.0*(1 if s>=20 else 0) - 3.0*(1 if s<=15 else 0)

if __name__ == "__main__":
    for depth in [2,3,4]:
        v, a, n = expectimax(18, depth, utility_fn=utility_risk_neutral, eval_fn=eval_linear)
        print(f"depth={depth} -> value={v:.3f}, act={a}, nodes={n}")
    v2, a2, n2 = expectimax(18, 3, utility_fn=lambda x: (x if x>=0 else -2*abs(x)), eval_fn=eval_linear)
    print("risk-averse depth=3 ->", v2, a2, n2)
    vs, as_, ns = expectimax_sample(18, 4, k=4, eval_fn=eval_linear)
    print("sampling depth=4 k=4 ->", vs, as_, ns)
