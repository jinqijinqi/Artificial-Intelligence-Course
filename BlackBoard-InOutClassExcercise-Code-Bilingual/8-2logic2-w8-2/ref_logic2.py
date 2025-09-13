from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Optional, Iterable, Union
import itertools

# ===== Propositional AST =====
@dataclass(frozen=True)  # atoms
class PVar: name: str
@dataclass(frozen=True)  # unary
class PNot: f: object
@dataclass(frozen=True)  # binary
class PAnd: a: object; b: object
@dataclass(frozen=True)
class POr: a: object; b: object
@dataclass(frozen=True)
class PImp: a: object; b: object
@dataclass(frozen=True)
class PIff: a: object; b: object

def eliminate_iff_imp(f):
    if isinstance(f, PIff):
        # (a<->b) == (a->b)&&(b->a)
        return PAnd(eliminate_iff_imp(PImp(f.a,f.b)), eliminate_iff_imp(PImp(f.b,f.a)))
    if isinstance(f, PImp):
        # (a->b) == (!a || b)
        return POr(PNot(eliminate_iff_imp(f.a)), eliminate_iff_imp(f.b))
    if isinstance(f, PNot): return PNot(eliminate_iff_imp(f.f))
    if isinstance(f, PAnd): return PAnd(eliminate_iff_imp(f.a), eliminate_iff_imp(f.b))
    if isinstance(f, POr):  return POr(eliminate_iff_imp(f.a), eliminate_iff_imp(f.b))
    return f

def push_not(f):
    if isinstance(f, PNot):
        g = f.f
        if isinstance(g, PNot): return push_not(g.f)
        if isinstance(g, PAnd): return POr(push_not(PNot(g.a)), push_not(PNot(g.b)))
        if isinstance(g, POr):  return PAnd(push_not(PNot(g.a)), push_not(PNot(g.b)))
        return f
    if isinstance(f, PAnd): return PAnd(push_not(f.a), push_not(f.b))
    if isinstance(f, POr):  return POr(push_not(f.a), push_not(f.b))
    return f

def distribute_or_over_and(f):
    if isinstance(f, POr):
        A, B = distribute_or_over_and(f.a), distribute_or_over_and(f.b)
        if isinstance(A, PAnd):
            return PAnd(distribute_or_over_and(POr(A.a, B)), distribute_or_over_and(POr(A.b, B)))
        if isinstance(B, PAnd):
            return PAnd(distribute_or_over_and(POr(A, B.a)), distribute_or_over_and(POr(A, B.b)))
        return POr(A,B)
    if isinstance(f, PAnd): return PAnd(distribute_or_over_and(f.a), distribute_or_over_and(f.b))
    return f

def to_cnf(f):
    f1 = eliminate_iff_imp(f)
    f2 = push_not(f1)
    f3 = distribute_or_over_and(f2)
    # extract clauses as sets of literals (name, sign)
    clauses = []
    def gather(g):
        if isinstance(g, PAnd):
            gather(g.a); gather(g.b)
        else:
            # a clause
            lits = set()
            def collect(h):
                if isinstance(h, POr):
                    collect(h.a); collect(h.b)
                elif isinstance(h, PNot) and isinstance(h.f, PVar):
                    lits.add((h.f.name, False))
                elif isinstance(h, PVar):
                    lits.add((h.name, True))
                else:
                    # wrap non-literal as a fresh symbol (rare in our use)
                    lits.add((str(h), True))
            collect(g)
            clauses.append(frozenset(lits))
    gather(f3)
    return set(clauses)

def resolution_entails(kb_clauses: Set[frozenset], query_clauses: Set[frozenset]):
    # Refutation: add negation of query as CNF (here query_clauses already CNF)
    clauses = set(kb_clauses) | set(query_clauses)
    new = set()
    parents = {}  # child -> (c1,c2)
    def resolvents(c1, c2):
        res = set()
        for (p, s1) in c1:
            key = (p, not s1)
            if key in c2:
                # resolvent = (c1\{p^s1}) ∪ (c2\{p^¬s1})
                r = (c1 - {(p,s1)}) | (c2 - {key})
                res.add(frozenset(r))
        return res
    while True:
        pairs = [(c1,c2) for i,c1 in enumerate(clauses) for j,c2 in enumerate(clauses) if i<j]
        for (c1,c2) in pairs:
            for r in resolvents(c1,c2):
                if not r:  # empty clause
                    parents[r] = (c1,c2)
                    return True, parents
                if r not in clauses:
                    new.add(r)
                    parents[r] = (c1,c2)
        if new.issubset(clauses):  # no progress
            return False, parents
        clauses |= new
        new.clear()

# ===== Horn Forward Chaining =====
def forward_chain(facts: Set[str], rules: List[Tuple[Set[str], str]]):
    derived = set(facts)
    just = {}  # head -> premises
    changed = True
    while changed:
        changed = False
        for premises, head in rules:
            if premises.issubset(derived) and head not in derived:
                derived.add(head); just[head] = set(premises); changed=True
    return derived, just

# ===== First-Order: terms, atoms, substitution, unification, FO-MP =====
@dataclass(frozen=True)
class Const: name: str
@dataclass(frozen=True)
class Var: name: str
@dataclass(frozen=True)
class Fun:
    name: str
    args: Tuple[object, ...]
@dataclass(frozen=True)
class Pred:
    name: str
    args: Tuple[object, ...]

Term = Union[Const, Var, Fun]

def occurs(v: Var, t: Term) -> bool:
    if isinstance(t, Var): return t==v
    if isinstance(t, Fun): return any(occurs(v,a) for a in t.args)
    return False

def subst(theta: Dict[Var, Term], obj):
    if isinstance(obj, Var): return theta.get(obj, obj)
    if isinstance(obj, Const): return obj
    if isinstance(obj, Fun):  return Fun(obj.name, tuple(subst(theta,a) for a in obj.args))
    if isinstance(obj, Pred): return Pred(obj.name, tuple(subst(theta,a) for a in obj.args))
    if isinstance(obj, (list,tuple)): return type(obj)(subst(theta,x) for x in obj)
    return obj

def unify(a, b, theta=None):
    if theta is None: theta = {}
    a = subst(theta, a); b = subst(theta, b)
    if a==b: return theta
    if isinstance(a, Var):
        if occurs(a,b): raise ValueError("occurs check fails")
        theta = dict(theta); theta[a]=b; return theta
    if isinstance(b, Var):
        if occurs(b,a): raise ValueError("occurs check fails")
        theta = dict(theta); theta[b]=a; return theta
    if isinstance(a, Fun) and isinstance(b, Fun) and a.name==b.name and len(a.args)==len(b.args):
        for x,y in zip(a.args, b.args):
            theta = unify(x,y,theta)
        return theta
    if isinstance(a, Pred) and isinstance(b, Pred) and a.name==b.name and len(a.args)==len(b.args):
        for x,y in zip(a.args, b.args):
            theta = unify(x,y,theta)
        return theta
    raise ValueError("cannot unify")

def fo_modus_ponens(facts: List[Pred], rule_premises: List[Pred], rule_head: Pred):
    # try to unify conjunction of rule_premises with some subset of facts
    # naive: try all matchings of rule premises to facts
    results = []
    for combo in itertools.permutations(facts, r=len(rule_premises)):
        try:
            theta = {}
            ok=True
            for a,b in zip(combo, rule_premises):
                theta = unify(a, b, theta)
            head_inst = subst(theta, rule_head)
            results.append(head_inst)
        except Exception:
            ok=False
        if ok: break
    return results

# ===== Example wiring for class KB =====
def class_kb_demo():
    # Propositional Horn version
    facts = {"Takes_alice_cs221", "Covers_cs221_mdp"}
    rules = [
        ({"Takes_alice_cs221", "Covers_cs221_mdp"}, "Knows_alice_mdp")
    ]
    derived, just = forward_chain(facts, rules)

    # Propositional resolution refutation for KB ∧ ¬Knows
    A = PVar("Takes_alice_cs221"); B = PVar("Covers_cs221_mdp"); C = PVar("Knows_alice_mdp")
    rule = PImp(PAnd(A,B), C)
    kb_cnf = to_cnf(rule) | to_cnf(A) | to_cnf(B)
    neg_query = to_cnf(PNot(C))
    entails, proof = resolution_entails(kb_cnf, neg_query)

    # FO-MP
    alice, cs221, mdp = Const("alice"), Const("cs221"), Const("mdp")
    x,y,z = Var("x"), Var("y"), Var("z")
    facts_fo = [Pred("Takes",(alice,cs221)), Pred("Covers",(cs221,mdp))]
    rule_prems = [Pred("Takes",(x,y)), Pred("Covers",(y,z))]
    rule_head = Pred("Knows",(x,z))
    fo_results = fo_modus_ponens(facts_fo, rule_prems, rule_head)
    return derived, just, entails, fo_results

if __name__ == "__main__":
    print(class_kb_demo())
