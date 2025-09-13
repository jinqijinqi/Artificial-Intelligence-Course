from typing import Dict, List, Callable, Set, Tuple, Optional, Iterable

Assignment = Dict[str, str]
Domain = Dict[str, List[str]]
Constraint = Callable[[str, str, str, str], bool]  # (xi, vi, xj, vj) -> ok?

class CSP:
    def __init__(self, variables: List[str], domains: Domain):
        self.variables = variables
        self.domains = {v: list(domains[v]) for v in variables}
        self.neigh: Dict[str, Set[str]] = {v: set() for v in variables}
        self.binary_constraints: List[Tuple[str, str, Constraint]] = []

    def add_binary_constraint(self, xi: str, xj: str, pred: Constraint):
        self.neigh[xi].add(xj)
        self.neigh[xj].add(xi)
        self.binary_constraints.append((xi, xj, pred))
        self.binary_constraints.append((xj, xi, lambda a,va,b,vb,pred=pred: pred(b,vb,a,va)))

    def neighbors(self, x: str) -> Set[str]:
        return self.neigh[x]

    def consistent_pair(self, xi: str, vi: str, xj: str, vj: str) -> bool:
        # check constraints involving (xi,xj)
        for a,b,p in self.binary_constraints:
            if a==xi and b==xj:
                if not p(xi,vi,xj,vj): return False
        return True

# ---------- Heuristics ----------
def mrv(assignment: Assignment, csp: CSP) -> str:
    unassigned = [x for x in csp.variables if x not in assignment]
    # min remaining values
    lens = {x: sum(all(csp.consistent_pair(x,v, y, assignment[y]) for y in csp.neighbors(x) if y in assignment)
                   for v in csp.domains[x]) for x in unassigned}
    m = min(lens.values())
    candidates = [x for x in unassigned if lens[x]==m]
    if len(candidates)==1:
        return candidates[0]
    # degree tie-break: choose variable with most constraints on unassigned vars
    def degree(x): 
        return sum(1 for y in csp.neighbors(x) if y not in assignment)
    candidates.sort(key=lambda x: -degree(x))
    return candidates[0]

def lcv(x: str, assignment: Assignment, csp: CSP) -> List[str]:
    # least-constraining value ordering
    def score(v):
        cnt = 0
        for y in csp.neighbors(x):
            if y in assignment: 
                continue
            for w in csp.domains[y]:
                if not csp.consistent_pair(x,v,y,w):
                    cnt += 1
        return cnt
    return sorted(csp.domains[x], key=score)

# ---------- Forward Checking with undo ----------
def forward_check(x: str, v: str, assignment: Assignment, csp: CSP):
    # prune domains of neighbors; return list of (var, removed_values) to undo
    removed = []
    for y in csp.neighbors(x):
        if y in assignment: 
            continue
        to_remove = [w for w in csp.domains[y] if not csp.consistent_pair(x,v,y,w)]
        if to_remove:
            csp.domains[y] = [w for w in csp.domains[y] if w not in to_remove]
            removed.append((y, to_remove))
            if not csp.domains[y]:
                return False, removed
    return True, removed

def undo(removed, csp: CSP):
    for y, vals in removed:
        # restore in any order; keep unique
        cur = set(csp.domains[y])
        for w in vals:
            if w not in cur:
                csp.domains[y].append(w)

# ---------- Backtracking ----------
def backtracking_search(csp: CSP, use_lcv=True, use_fc=True):
    assignment: Assignment = {}
    nodes = 0; backtracks = 0
    order_log = []

    def backtrack():
        nonlocal nodes, backtracks
        if len(assignment)==len(csp.variables):
            return True
        x = mrv(assignment, csp)
        values = lcv(x, assignment, csp) if use_lcv else list(csp.domains[x])
        for v in values:
            nodes += 1
            # check consistency with assigned neighbors
            ok = all(csp.consistent_pair(x,v,y,assignment[y]) for y in csp.neighbors(x) if y in assignment)
            if not ok: 
                continue
            assignment[x]=v; order_log.append((x,v))
            removed = []
            if use_fc:
                ok, removed = forward_check(x,v,assignment,csp)
            if ok:
                if backtrack():
                    return True
            # undo
            if use_fc:
                undo(removed, csp)
            order_log.pop(); assignment.pop(x, None)
        backtracks += 1
        return False

    success = backtrack()
    return success, assignment, nodes, backtracks, order_log

# ---------- Australia instance ----------
def australia_csp():
    vars = ["WA","NT","SA","Q","NSW","V","T"]
    dom = {v:["R","G","B"] for v in vars}
    csp = CSP(vars, dom)
    edges = [("WA","NT"),("WA","SA"),("NT","SA"),("NT","Q"),
             ("SA","Q"),("SA","NSW"),("SA","V"),("Q","NSW"),("NSW","V")]
    ne = lambda xi,vi,xj,vj: vi != vj
    for a,b in edges:
        csp.add_binary_constraint(a,b,ne)
    return csp

if __name__ == "__main__":
    csp = australia_csp()
    ok, sol, nodes, backs, log = backtracking_search(csp, use_lcv=True, use_fc=True)
    print("Solved:", ok, "nodes:", nodes, "backtracks:", backs)
    print("Solution:", sol)
    print("Order:", log)
