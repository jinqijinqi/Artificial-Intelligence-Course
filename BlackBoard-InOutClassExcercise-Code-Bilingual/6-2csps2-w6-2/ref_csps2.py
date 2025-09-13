from typing import Dict, List, Tuple, Callable, Optional
import itertools, random

Var = str
Val = int
Assignment = Dict[Var, Val]

class WeightedCSP:
    def __init__(self, variables: List[Var], domains: Dict[Var, List[Val]]):
        self.variables = variables
        self.domains = {v:list(domains[v]) for v in variables}
        # factors: unary[var] -> dict[val]->w ; binary[(u,v)] -> dict[(a,b)]->w
        self.unary = {v:{} for v in variables}
        self.binary = {}  # key is ordered pair (u,v)
        self.neigh = {v:set() for v in variables}

    def add_unary(self, v: Var, table: Dict[Val, float]):
        self.unary[v] = dict(table)

    def add_binary(self, u: Var, v: Var, table: Dict[Tuple[Val,Val], float]):
        self.binary[(u,v)] = dict(table)
        self.binary[(v,u)] = {(b,a):w for (a,b),w in table.items()}
        self.neigh[u].add(v); self.neigh[v].add(u)

    # ----- factor evaluation -----
    def dep_weight(self, x: Assignment, var: Var, val: Val) -> float:
        """Product of factors touching var whose other vars are already assigned in x."""
        w = 1.0
        # unary
        if self.unary[var]:
            w *= self.unary[var].get(val, 0.0)
        # binary with assigned neighbors
        for nb in self.neigh[var]:
            if nb in x:
                w *= self.binary[(var,nb)].get((val, x[nb]), 0.0)
        return w

    def full_weight(self, x: Assignment) -> float:
        # assumes all variables assigned
        w = 1.0
        for v in self.variables:
            if self.unary[v]: w *= self.unary[v].get(x[v], 0.0)
        for (u,v), tab in self.binary.items():
            if (u < v):  # count each undirected pair once
                w *= tab.get((x[u], x[v]), 0.0)
        return w

# ---------- Lookahead: forward checking ----------
def forward_check(csp: WeightedCSP, x: Assignment, var: Var, val: Val):
    """Return (ok, removed) where removed is list of (y, values) pruned; prune only 0-supported values."""
    removed = []
    for y in csp.neigh[var]:
        if y in x: continue
        to_rm = []
        for b in list(csp.domains[y]):
            # check if any factor forbids (var=val, y=b)
            w = csp.binary[(var,y)].get((val,b), 0.0)
            if w == 0.0:
                to_rm.append(b)
        if to_rm:
            removed.append((y, to_rm))
            csp.domains[y] = [b for b in csp.domains[y] if b not in to_rm]
            if not csp.domains[y]:
                return False, removed
    return True, removed

def undo_fc(csp: WeightedCSP, removed):
    for y, vals in removed:
        for b in vals:
            if b not in csp.domains[y]:
                csp.domains[y].append(b)

# ---------- AC-3 ----------
from collections import deque
def enforce_arc_consistency(csp: WeightedCSP):
    """AC-3 using zero-support pruning on binary factors."""
    q = deque()
    for (u,v) in csp.binary.keys():
        q.append((u,v))
    changed = False
    while q:
        u,v = q.popleft()
        dom_u = list(csp.domains[u])
        removed = False
        for a in dom_u:
            # check if a has any supporting b in v's domain with nonzero factor
            ok = any(csp.binary[(u,v)].get((a,b),0.0) > 0.0 for b in csp.domains[v])
            if not ok:
                csp.domains[u].remove(a)
                removed = True
                changed = True
        if removed:
            for w in csp.neigh[u]:
                if w != v:
                    q.append((w,u))
    return changed

# ---------- Heuristics ----------
def mrv(csp: WeightedCSP, x: Assignment) -> Var:
    unassigned = [v for v in csp.variables if v not in x]
    # MRV: smallest domain size
    k = min(len(csp.domains[v]) for v in unassigned)
    cands = [v for v in unassigned if len(csp.domains[v]) == k]
    # tie-break by degree
    cands.sort(key=lambda v: -len([nb for nb in csp.neigh[v] if nb not in x]))
    return cands[0]

def lcv_values(csp: WeightedCSP, x: Assignment, var: Var) -> List[Val]:
    def score(val):
        # count how many neighbor values remain nonzero-compatible
        s = 0
        for nb in csp.neigh[var]:
            if nb in x: continue
            s += sum(1 for b in csp.domains[nb] if csp.binary[(var,nb)].get((val,b),0.0) > 0.0)
        return -s  # smaller is worse
    return sorted(list(csp.domains[var]), key=score)

# ---------- Backtracking ----------
def backtracking(csp: WeightedCSP):
    x: Assignment = {}
    best = (0.0, None)  # (weight, assignment)
    nodes = 0; backs = 0

    # optional AC-3 before search
    enforce_arc_consistency(csp)

    def dfs():
        nonlocal nodes, backs, best
        if len(x) == len(csp.variables):
            w = csp.full_weight(x)
            if w > best[0]: best = (w, dict(x))
            return True
        var = mrv(csp, x)
        for val in lcv_values(csp, x, var):
            nodes += 1
            delta = csp.dep_weight(x, var, val)
            if delta == 0.0: 
                continue
            x[var] = val
            # forward check + AC-3
            ok, removed = forward_check(csp, x, var, val)
            if ok:
                enforce_arc_consistency(csp)
                dfs()
            undo_fc(csp, removed)
            x.pop(var, None)
        backs += 1
        return False

    dfs()
    return best, nodes, backs

# ---------- Beam search ----------
def beam_search(csp: WeightedCSP, K: int):
    # candidates are (assignment, weight)
    cand = [({}, 1.0)]
    for var in csp.variables:
        # extend all
        ext = []
        for x, w in cand:
            for val in csp.domains[var]:
                delta = csp.dep_weight(x, var, val)
                if delta == 0.0: 
                    continue
                x2 = dict(x); x2[var] = val
                ext.append((x2, w*delta))
        # keep top-K by weight
        ext.sort(key=lambda t: t[1], reverse=True)
        cand = ext[:K] if ext else []
        if not cand: break
    # pick best full if exists
    best = max(cand, key=lambda t: t[1]) if cand else ({}, 0.0)
    return best

# ---------- Local search (ICM) ----------
def icm(csp: WeightedCSP, iters: int=10, seed: int=0):
    random.seed(seed)
    # random full assignment (not guaranteed positive weight)
    x = {v: random.choice(csp.domains[v]) for v in csp.variables}
    def local_weight(var, val):
        # local product: unary(var) * binaries with neighbors
        w = csp.unary[var].get(val, 1.0) if csp.unary[var] else 1.0
        for nb in csp.neigh[var]:
            b = x[nb]
            w *= csp.binary[(var,nb)].get((val, b), 0.0)
        return w
    improved = True
    steps = 0
    while improved and steps < iters:
        improved = False; steps += 1
        for v in csp.variables:
            best = max(csp.domains[v], key=lambda a: local_weight(v,a))
            if local_weight(v, best) > local_weight(v, x[v]):
                x[v] = best; improved = True
    # compute full weight at end (includes unary of all vars and binaries once)
    return x

# ---------- Instance: 3-step tracking ----------
def build_tracking_instance():
    vars = ["X1","X2","X3"]
    doms = {v:[0,1,2] for v in vars}
    csp = WeightedCSP(vars, doms)
    obs = { "X1":0, "X2":2, "X3":2 }
    # unary obs factors: 2,1,0 by distance
    for v in vars:
        table = {a: max(0, 2-abs(a-obs[v])) for a in doms[v]}
        csp.add_unary(v, table)
    # binary transitions
    def trans(a,b):
        if a==b: return 2
        if abs(a-b)==1: return 1
        return 0
    for (u,v) in [("X1","X2"),("X2","X3")]:
        tab = {}
        for a in doms[u]:
            for b in doms[v]:
                tab[(a,b)] = trans(a,b)
        csp.add_binary(u,v, tab)
    return csp

if __name__ == "__main__":
    csp = build_tracking_instance()
    best, nodes, backs = backtracking(csp)
    print("Backtracking best:", best, "nodes:", nodes, "backs:", backs)
    csp2 = build_tracking_instance()
    print("Beam K=2:", beam_search(csp2, K=2))
    csp3 = build_tracking_instance()
    print("ICM:", icm(csp3, iters=10, seed=0))
