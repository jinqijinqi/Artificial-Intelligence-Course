from typing import List, Tuple, Dict, Any
import math, heapq, random

# ---------- 1) Tool recommender ----------
def recommend_tools(spec: Dict[str, bool]) -> Dict[str, Any]:
    """
    spec flags: perception, path_planning, stochastic, hidden_state, logic_rules
    """
    rec = {"paradigms": [], "algorithms": []}
    if spec.get("perception"):
        rec["paradigms"].append("reflex")
        rec["algorithms"].append({"inference":"feedforward", "learning":"SGD", "models":["linear","CNN","kNN"]})
    if spec.get("path_planning"):
        rec["paradigms"].append("state")
        rec["algorithms"].append({"inference":"A* / UCS", "learning":"-", "models":["search"]})
    if spec.get("stochastic"):
        rec["paradigms"].append("state")
        rec["algorithms"].append({"inference":"value iteration", "learning":"TD / Q-learning", "models":["MDP"]})
    if spec.get("hidden_state"):
        rec["paradigms"].append("variable")
        rec["algorithms"].append({"inference":"forward-backward / particle / Gibbs", "learning":"MLE / EM", "models":["HMM/BN/MN"]})
    if spec.get("logic_rules"):
        rec["paradigms"].append("logic")
        rec["algorithms"].append({"inference":"model checking / MP / resolution", "learning":"-", "models":["prop/FOL"]})
    return rec

# ---------- 2) Grid A* demo ----------
def astar(grid: List[str], s: Tuple[int,int], t: Tuple[int,int]) -> Tuple[int, List[Tuple[int,int]]]:
    H, W = len(grid), len(grid[0])
    def h(p): return abs(p[0]-t[0]) + abs(p[1]-t[1])
    def nbrs(p):
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            x,y = p[0]+dx, p[1]+dy
            if 0<=x<H and 0<=y<W and grid[x][y] != '#':
                yield (x,y)
    g = {s:0}; came = {}; openq = [(h(s), 0, s)]
    seen=set()
    while openq:
        _, gc, u = heapq.heappop(openq)
        if u in seen: continue
        seen.add(u)
        if u==t: break
        for v in nbrs(u):
            ng = gc+1
            if ng < g.get(v, 1e9):
                g[v]=ng; came[v]=u
                heapq.heappush(openq, (ng+h(v), ng, v))
    if t not in came and s!=t: return (math.inf, [])
    path = [t]
    while path[-1]!=s:
        path.append(came[path[-1]])
    path.reverse()
    return (len(path)-1, path)

# ---------- 3) HMM forward-backward (domain {0,1,2}) ----------
def fb_1d(evidence: List[int]) -> List[Dict[int,float]]:
    dom = [0,1,2]
    def trans(hp,h): 
        if h==hp: return 0.5
        if abs(h-hp)==1: return 0.25
        return 0.0
    def emit(h,e):
        if e==h: return 0.5
        if abs(e-h)==1: return 0.25
        return 0.0
    n=len(evidence)
    prior = {h:1/3 for h in dom}
    F=[{h:0.0 for h in dom} for _ in range(n)]
    B=[{h:1.0 for h in dom} for _ in range(n)]
    for h in dom: F[0][h]=prior[h]*emit(h,evidence[0])
    s=sum(F[0].values()); F[0]={h:F[0][h]/s for h in dom}
    for i in range(1,n):
        for h in dom:
            F[i][h]=emit(h,evidence[i])*sum(F[i-1][hp]*trans(hp,h) for hp in dom)
        s=sum(F[i].values()); F[i]={h:F[i][h]/s for h in dom}
    for i in reversed(range(n-1)):
        for h in dom:
            B[i][h]=sum(B[i+1][hn]*trans(h,hn)*emit(hn,evidence[i+1]) for hn in dom)
        s=sum(B[i].values()); B[i]={h:B[i][h]/s for h in dom}
    post=[]
    for i in range(n):
        S={h:F[i][h]*B[i][h] for h in dom}; s=sum(S.values()); post.append({h:S[h]/s for h in dom})
    return post

# ---------- 4) Ethics checklist ----------
def audit(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    out = {"data":[], "objective":[], "inequality":[], "harm":[], "ia":[], "actions":[]}
    ds = spec.get("data_sources","")
    if "web" in ds.lower():
        out["data"].append("Web-scraped data can contain offensive content and historical bias; curate & filter.")
        out["actions"].append("Add data filters; human-in-the-loop review; document datasheets.")
    if spec.get("objective","").lower() in {"clicks","views"}:
        out["objective"].append("Surrogate objective may misalign with user welfare.")
        out["actions"].append("Use multi-objective optimization; long-term user value metrics.")
    if spec.get("users") and "underrepresented" in spec["users"]:
        out["inequality"].append("Potential disparity on under-represented groups.")
        out["actions"].append("Audit by group; collect balanced data; min-max (worst-group) loss.")
    if spec.get("potential_misuse"):
        out["harm"].append("Dual-use risks present.")
        out["actions"].append("Red-team; restrict API; watermarking/traceability.")
    out["ia"].append("Prefer IA: keep humans-in-the-loop; design interpretable controls.")
    return out

# ---------- 5) Course triad ----------
def next_courses(goal: str="robotics") -> Dict[str, List[str]]:
    M = {
        "robotics": {
            "Methods": ["CS229", "CS230", "CS234", "CS238"],
            "Applications": ["CS237AB", "CS223A"],
            "Foundations": ["EE364/CS334", "STATS200"]
        },
        "nlp": {
            "Methods": ["CS229", "CS230", "CS228", "CS236"],
            "Applications": ["CS224N", "CS224U", "CS224V", "CS224C", "CS324"],
            "Foundations": ["EE364/CS334", "STATS214/CS229M"]
        },
        "vision": {
            "Methods": ["CS229", "CS230", "CS228"],
            "Applications": ["CS231N", "CS231A", "CS348I"],
            "Foundations": ["EE364/CS334", "STATS200"]
        }
    }
    return M.get(goal.lower(), M["robotics"])
