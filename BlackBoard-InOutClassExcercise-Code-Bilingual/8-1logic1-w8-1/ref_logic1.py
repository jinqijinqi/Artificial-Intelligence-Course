from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Iterable, Tuple, List, Optional
import itertools

# ---- AST ----
@dataclass(frozen=True)
class Var:
    name: str

@dataclass(frozen=True)
class Not:
    f: object

@dataclass(frozen=True)
class And:
    a: object; b: object

@dataclass(frozen=True)
class Or:
    a: object; b: object

@dataclass(frozen=True)
class Imp:
    a: object; b: object  # a -> b

@dataclass(frozen=True)
class Iff:
    a: object; b: object  # a <-> b

def atoms_in(f) -> Set[str]:
    if isinstance(f, Var): return {f.name}
    if isinstance(f, Not): return atoms_in(f.f)
    if isinstance(f, (And, Or, Imp, Iff)): return atoms_in(f.a) | atoms_in(f.b)
    raise TypeError(f"Unknown node: {f}")

def eval_formula(f, w: Dict[str, int]) -> int:
    if isinstance(f, Var): return 1 if w.get(f.name, 0) else 0
    if isinstance(f, Not): return 1 - eval_formula(f.f, w)
    if isinstance(f, And): return eval_formula(f.a, w) & eval_formula(f.b, w)
    if isinstance(f, Or):  return max(eval_formula(f.a, w), eval_formula(f.b, w))
    if isinstance(f, Imp): return 1 if (eval_formula(f.a, w)==0 or eval_formula(f.b, w)==1) else 0
    if isinstance(f, Iff): 
        ea, eb = eval_formula(f.a, w), eval_formula(f.b, w)
        return 1 if ea==eb else 0
    raise TypeError(f"Unknown node: {f}")

def models_of_KB(KB: Iterable[object]) -> List[Dict[str,int]]:
    atoms = sorted(set().union(*[atoms_in(f) for f in KB])) if KB else []
    sols = []
    for vals in itertools.product([0,1], repeat=len(atoms)):
        w = dict(zip(atoms, vals))
        if all(eval_formula(f, w)==1 for f in KB):
            sols.append(w)
    return sols

def satisfiable(KB: Iterable[object]) -> Tuple[bool, Optional[Dict[str,int]]]:
    sols = models_of_KB(KB)
    if sols: return True, sols[0]
    return False, None

def entails(KB: Iterable[object], f) -> bool:
    # KB |= f  iff  KB ∪ {¬f} is UNSAT
    sat, _ = satisfiable(list(KB) + [Not(f)])
    return not sat

# ---- Forward chaining with Modus Ponens only ----
def forward_chain_modus_ponens(KB: Iterable[object]) -> Set[object]:
    KB = set(KB)
    changed = True
    while changed:
        changed = False
        # collect (p, (p->q)) pairs
        facts = {f for f in KB if isinstance(f, Var) or (isinstance(f, Not) and isinstance(f.f, Var))}
        imps  = {f for f in KB if isinstance(f, Imp)}
        for imp in list(imps):
            p, q = imp.a, imp.b
            if p in KB and q not in KB:
                KB.add(q); changed = True
    return KB

# ---- Examples used in class ----
Rain, Wet, Slippery, Snow = map(Var, ["Rain","Wet","Slippery","Snow"])

if __name__ == "__main__":
    KB = {Rain, Imp(Rain,Wet), Imp(Wet,Slippery)}
    # Entailment checks
    print("KB entails Wet?", entails(KB, Wet))
    print("KB entails Rain->Slippery?", entails(KB, Imp(Rain,Slippery)))
    print("KB entails not Rain?", entails(KB, Not(Rain)))
    # Forward chaining (MP)
    FC = forward_chain_modus_ponens(KB)
    print("Forward-derived:", FC)
    # Contingency witness for Snow
    print("KB ∪ {Snow} satisfiable?", satisfiable(KB | {Snow}))
    print("KB ∪ {¬Snow} satisfiable?", satisfiable(KB | {Not(Snow)}))
