from typing import List, Tuple, Optional, Dict

MAX, MIN = 'X', 'O'

def pretty(s: str) -> str:
    g = [s[i:i+3] for i in range(0,9,3)]
    return "\n".join(" ".join(c if c != '.' else '_' for c in row) for row in g)

def player_to_move(s: str) -> str:
    return MAX if s.count(MAX) == s.count(MIN) else MIN

def legal_moves(s: str) -> List[int]:
    return [i for i,c in enumerate(s) if c == '.']

def next_state(s: str, a: int) -> str:
    p = player_to_move(s)
    return s[:a] + p + s[a+1:]

def lines() -> List[Tuple[int,int,int]]:
    return [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def winner(s: str) -> Optional[str]:
    for a,b,c in lines():
        if s[a] != '.' and s[a] == s[b] == s[c]:
            return s[a]
    return None

def is_terminal(s: str) -> bool:
    return winner(s) is not None or '.' not in s

def utility(s: str) -> int:
    w = winner(s)
    if w == MAX: return +1
    if w == MIN: return -1
    return 0

# ---- Evaluation ----
def eval_features(s: str) -> Tuple[int,int,int,int]:
    # (open-X-2s, open-O-2s, centerX, cornerX)
    openX = openO = 0
    for a,b,c in lines():
        line = s[a]+s[b]+s[c]
        if line.count(MIN)==0 and line.count(MAX)==2: openX += 1
        if line.count(MAX)==0 and line.count(MIN)==2: openO += 1
    centerX = 1 if s[4]==MAX else 0
    corners = [0,2,6,8]
    cornerX = sum(1 for i in corners if s[i]==MAX)
    return (openX, openO, centerX, cornerX)

def eval_linear(s: str, w=(3, -3, 1, 1)) -> int:
    f = eval_features(s)
    return sum(wi*fi for wi,fi in zip(w,f))

# ---- Minimax / Alpha-Beta ----
def minimax(s: str, depth: int) -> Tuple[int, Optional[int], int]:
    """Return (value, best_move, nodes) from perspective of player_to_move(s)."""
    nodes = 0
    def mm(state, d) -> int:
        nonlocal nodes
        nodes += 1
        if is_terminal(state) or d == 0:
            return utility(state) if is_terminal(state) else eval_linear(state)
        p = player_to_move(state)
        moves = legal_moves(state)
        if p == MAX:
            best = -10**9
            for a in moves:
                best = max(best, mm(next_state(state,a), d-1))
            return best
        else:
            best = 10**9
            for a in moves:
                best = min(best, mm(next_state(state,a), d-1))
            return best
    p = player_to_move(s)
    best_move = None
    best_val = -10**9 if p==MAX else 10**9
    for a in legal_moves(s):
        v = mm(next_state(s,a), depth-1)
        if (p==MAX and v>best_val) or (p==MIN and v<best_val):
            best_val, best_move = v, a
    return best_val, best_move, nodes

def move_order_heuristic(s: str, moves: List[int]) -> List[int]:
    # center > corners > edges
    center = [4]; corners = [0,2,6,8]; edges = [1,3,5,7]
    order = center + corners + edges
    return sorted(moves, key=lambda a: order.index(a) if a in order else 99)

def alphabeta(s: str, depth: int, w=(3,-3,1,1)) -> Tuple[int, Optional[int], int, int]:
    """Return (value, best_move, nodes, prunes)."""
    nodes = prunes = 0
    TT: Dict[Tuple[str,int], int] = {}  # simple transposition: (state,depth)->value

    def ab(state, d, alpha, beta) -> int:
        nonlocal nodes, prunes
        nodes += 1
        key = (state, d)
        if key in TT:
            return TT[key]
        if is_terminal(state) or d == 0:
            val = utility(state) if is_terminal(state) else eval_linear(state, w)
            TT[key] = val
            return val
        p = player_to_move(state)
        moves = move_order_heuristic(state, legal_moves(state))
        if p == MAX:
            val = -10**9
            for a in moves:
                val = max(val, ab(next_state(state,a), d-1, alpha, beta))
                alpha = max(alpha, val)
                if alpha >= beta:
                    prunes += 1
                    break
            TT[key]=val; return val
        else:
            val = 10**9
            for a in moves:
                val = min(val, ab(next_state(state,a), d-1, alpha, beta))
                beta = min(beta, val)
                if alpha >= beta:
                    prunes += 1
                    break
            TT[key]=val; return val

if __name__ == "__main__":
    s = "X.O..O..."
    print(pretty(s))
    print("Player to move:", player_to_move(s))
    v1, a1, n1 = minimax(s, depth=4)
    print("Minimax depth=4:", v1, "move", a1, "nodes", n1)
    v2, a2, n2, p2 = alphabeta(s, depth=6)
    print("AlphaBeta depth=6:", v2, "move", a2, "nodes", n2, "prunes", p2)
