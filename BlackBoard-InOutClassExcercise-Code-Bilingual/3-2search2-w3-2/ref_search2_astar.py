from collections import deque
import heapq

# ---------- Base ----------
class SearchProblem:
    def start_state(self): raise NotImplementedError
    def is_end(self, s):   raise NotImplementedError
    def succ_and_cost(self, s):
        """yield (action, s', cost)"""
        raise NotImplementedError

# ---------- Constrained Transportation ----------
class ConstrainedTransportation(SearchProblem):
    """
    State: (loc, delta), delta = #walk - #tram >= 0
    Start: (1, 0); End: any (n, delta>=0)
    Actions:
      walk: (loc, d) -> (loc+1, d+1)  cost 1
      tram: (loc, d) -> (2*loc, d-1)  cost 2, only if d-1 >= 0 and 2*loc <= n
    """
    def __init__(self, n):
        assert n >= 1
        self.n = n
    def start_state(self): return (1, 0)
    def is_end(self, s): loc, d = s; return loc == self.n and d >= 0
    def succ_and_cost(self, s):
        loc, d = s
        if loc < self.n:
            yield ("walk", (loc+1, d+1), 1)
            if d-1 >= 0 and 2*loc <= self.n:
                yield ("tram", (2*loc, d-1), 2)

# ---------- Relaxed Transportation (drop delta constraint) ----------
class RelaxedTransportation:
    def __init__(self, n): self.n = n
    def neighbors(self, s):
        if s < self.n:
            yield (s+1, 1)
            if 2*s <= self.n:
                yield (2*s, 2)

# ---------- UCS (for reference) ----------
def ucs(problem):
    start = problem.start_state()
    pq = [(0, start)]
    best = {start: 0}
    parent = {}
    explored = set()
    expansions = 0
    while pq:
        cost, s = heapq.heappop(pq)
        if s in explored: continue
        explored.add(s); expansions += 1
        if problem.is_end(s):
            return reconstruct(parent, s), cost, expansions
        for a, sp, c in problem.succ_and_cost(s):
            nc = cost + c
            if nc < best.get(sp, float("inf")):
                best[sp] = nc
                parent[sp] = (s, a, c)
                heapq.heappush(pq, (nc, sp))
    return None, float("inf"), expansions

def reconstruct(parent, s):
    path = []
    while s in parent:
        ps, a, c = parent[s]
        path.append((a, s, c))
        s = ps
    path.reverse()
    return path

# ---------- A* (UCS with f=g+h) ----------
def astar(problem, h):
    start = problem.start_state()
    pq = [(h(start), 0, start)]  # (f, g, s)
    best_g = {start: 0}
    parent = {}
    explored = set()
    expansions = 0
    while pq:
        f, g, s = heapq.heappop(pq)
        if s in explored: continue
        explored.add(s); expansions += 1
        if problem.is_end(s):
            return reconstruct(parent, s), g, expansions
        for a, sp, c in problem.succ_and_cost(s):
            new_g = g + c
            new_f = new_g + h(sp)
            if new_g < best_g.get(sp, float("inf")):
                best_g[sp] = new_g
                parent[sp] = (s, a, c)
                heapq.heappush(pq, (new_f, new_g, sp))
    return None, float("inf"), expansions

# ---------- Heuristics ----------
def h_zero(s): return 0

def make_h_walk(n):
    def h(s):
        loc = s if isinstance(s, int) else s[0]
        return max(0, n - loc)
    return h

def make_h_relaxed(n):
    """
    Compute relaxed FutureCost via UCS on the REVERSED relaxed problem.
    Equivalent to Dijkstra from goal node n on edges:
      (s-1)->s cost 1
      (2*s)->s cost 2
    """
    INF = 10**18
    dist = [INF]*(n+1)
    dist[n] = 0
    pq = [(0, n)]
    while pq:
        d, s = heapq.heappop(pq)
        if d != dist[s]: continue
        # reversed edges
        if s - 1 >= 1:
            v = s - 1; nd = d + 1
            if nd < dist[v]:
                dist[v] = nd; heapq.heappush(pq, (nd, v))
        if 2*s <= n:
            v = 2*s; nd = d + 2
            if nd < dist[v]:
                dist[v] = nd; heapq.heappush(pq, (nd, v))
    def h(s):
        loc = s if isinstance(s, int) else s[0]
        return dist[loc]
    return h

def make_h_max(h1, h2):
    return lambda s: max(h1(s), h2(s))

# ---------- Consistency checker ----------
def check_consistency(problem, h, samples=2000, seed=0):
    import random
    rnd = random.Random(seed)
    for _ in range(samples):
        s = problem.start_state()
        for __ in range(100):
            for a, sp, c in problem.succ_and_cost(s):
                cprime = c + h(sp) - h(s)
                if cprime < -1e-9:
                    return False
            succs = list(problem.succ_and_cost(s))
            if not succs: break
            a, sp, c = rnd.choice(succs)
            s = sp
            if problem.is_end(s): break
    return True

if __name__ == "__main__":
    n = 60
    prob = ConstrainedTransportation(n)
    h0 = h_zero
    hw = make_h_walk(n)
    hr = make_h_relaxed(n)
    hm = make_h_max(hw, hr)
    _, cu, _ = ucs(prob)
    _, c0, _ = astar(prob, h0)
    _, cw, _ = astar(prob, hw)
    _, cr, _ = astar(prob, hr)
    _, cm, _ = astar(prob, hm)
    print("Costs:", cu, c0, cw, cr, cm)
    print("Consistent(walk)?", check_consistency(prob, hw))
    print("Consistent(relaxed)?", check_consistency(prob, hr))
