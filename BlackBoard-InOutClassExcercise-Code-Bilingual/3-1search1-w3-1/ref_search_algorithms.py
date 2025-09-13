from collections import deque
import heapq

# ---------- SearchProblem interface ----------
class SearchProblem:
    def start_state(self):
        raise NotImplementedError
    def is_end(self, s):
        raise NotImplementedError
    def succ_and_cost(self, s):
        '''Yield (action, s', cost).'''
        raise NotImplementedError

# ---------- TransportationProblem ----------
class TransportationProblem(SearchProblem):
    '''
    States: integers s in [1..n]
    Actions:
      - 'walk' to s+1 with cost 1
      - 'tram' to 2*s with cost 2 (only if 2*s <= n)
    If unit_cost=True, treat both actions as cost=1 (for BFS/DFID demonstrations).
    '''
    def __init__(self, n, unit_cost=False):
        assert n >= 1
        self.n = n
        self.unit_cost = unit_cost

    def start_state(self):
        return 1

    def is_end(self, s):
        return s == self.n

    def succ_and_cost(self, s):
        if s < self.n:
            # walk
            c = 1 if not self.unit_cost else 1
            yield ('walk', s+1, c)
            # tram
            if 2*s <= self.n:
                c = 2 if not self.unit_cost else 1
                yield ('tram', 2*s, c)

# ---------- Utilities ----------
def reconstruct_path(parent, end_state):
    path = []
    s = end_state
    while s in parent:
        s_prev, action, cost = parent[s]
        path.append((action, s, cost))
        s = s_prev
    path.reverse()
    return path

# ---------- Backtracking (exponential) ----------
def backtracking_min_cost(problem):
    best = {'cost': float('inf'), 'path': None}
    expansions = 0

    def dfs(s, cost_so_far, parent):
        nonlocal expansions
        if cost_so_far >= best['cost']:
            return
        expansions += 1
        if problem.is_end(s):
            best['cost'] = cost_so_far
            best['path'] = reconstruct_path(parent, s)
            return
        for action, sp, c in problem.succ_and_cost(s):
            parent[sp] = (s, action, c)
            dfs(sp, cost_so_far + c, parent)
            parent.pop(sp, None)

    dfs(problem.start_state(), 0, {})
    return best['path'], best['cost'], expansions

# ---------- DFS (unit-cost, stop at first goal) ----------
def dfs_first_solution(problem, max_depth=10**6):
    start = problem.start_state()
    stack = [(start, 0)]
    parent = {}
    seen = set([start])
    expansions = 0
    while stack:
        s, depth = stack.pop()
        expansions += 1
        if problem.is_end(s):
            return reconstruct_path(parent, s), depth, expansions
        if depth == max_depth:
            continue
        for action, sp, c in problem.succ_and_cost(s):
            if sp not in seen:
                seen.add(sp)
                parent[sp] = (s, action, c)
                stack.append((sp, depth+1))
    return None, None, expansions

# ---------- BFS (unit-cost optimal) ----------
def bfs_unit_cost(problem):
    start = problem.start_state()
    q = deque([start])
    parent = {}
    seen = set([start])
    expansions = 0
    depth = {start: 0}
    while q:
        s = q.popleft()
        expansions += 1
        if problem.is_end(s):
            return reconstruct_path(parent, s), depth[s], expansions
        for action, sp, c in problem.succ_and_cost(s):
            if sp not in seen:
                seen.add(sp)
                parent[sp] = (s, action, c)
                depth[sp] = depth[s] + 1
                q.append(sp)
    return None, None, expansions

# ---------- DFS with Iterative Deepening (unit-cost optimal) ----------
def dfid_unit_cost(problem, max_depth=10**6):
    start = problem.start_state()
    expansions_total = 0
    for limit in range(max_depth+1):
        stack = [(start, 0)]
        parent = {}
        seen = {start}
        while stack:
            s, depth = stack.pop()
            expansions_total += 1
            if problem.is_end(s):
                return reconstruct_path(parent, s), depth, expansions_total
            if depth == limit:
                continue
            for action, sp, c in problem.succ_and_cost(s):
                if sp not in seen:
                    seen.add(sp)
                    parent[sp] = (s, action, c)
                    stack.append((sp, depth+1))
    return None, None, expansions_total

# ---------- Dynamic Programming (acyclic) ----------
def dp_future_cost(problem):
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def F(s):
        if problem.is_end(s):
            return 0
        best = float('inf')
        for action, sp, c in problem.succ_and_cost(s):
            best = min(best, c + F(sp))
        return best
    cost = F(problem.start_state())
    # reconstruct greedily
    path = []
    s = problem.start_state()
    while not problem.is_end(s):
        best_act = None
        best_val = float('inf')
        for action, sp, c in problem.succ_and_cost(s):
            val = c + F(sp)
            if val < best_val:
                best_val = val; best_act = (action, sp, c)
        action, sp, c = best_act
        path.append((action, sp, c))
        s = sp
    return path, cost

# ---------- Uniform Cost Search (Dijkstra on implicit graph) ----------
def ucs(problem):
    start = problem.start_state()
    frontier = [(0, start)]
    parent = {}
    best_cost = {start: 0}
    explored = set()
    expansions = 0
    while frontier:
        cost, s = heapq.heappop(frontier)
        if s in explored:
            continue
        explored.add(s)
        expansions += 1
        if problem.is_end(s):
            return reconstruct_path(parent, s), cost, expansions
        for action, sp, c in problem.succ_and_cost(s):
            new_cost = cost + c
            if new_cost < best_cost.get(sp, float('inf')):
                best_cost[sp] = new_cost
                parent[sp] = (s, action, c)
                heapq.heappush(frontier, (new_cost, sp))
    return None, float('inf'), expansions

if __name__ == "__main__":
    # Sanity: n=10 should have optimal total cost 6
    prob = TransportationProblem(10, unit_cost=False)
    path_dp, cost_dp = dp_future_cost(prob)
    path_ucs, cost_ucs, exp_ucs = ucs(prob)
    print("DP:", cost_dp, path_dp)
    print("UCS:", cost_ucs, path_ucs, "expansions:", exp_ucs)
