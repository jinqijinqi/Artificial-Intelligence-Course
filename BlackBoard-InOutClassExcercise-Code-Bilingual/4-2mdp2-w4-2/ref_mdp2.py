from typing import Dict, Tuple, Iterable
State = Tuple[int,int]  # (row, col)
Action = str            # 'U','D','L','R'

class GridWorldMDP:
    def __init__(self, rows=3, cols=4, walls={(2,2)}, goals={(1,4):1.0}, lava={(2,4):-1.0},
                 step_reward=-0.04, slip=0.2):
        self.R = rows; self.C = cols
        self.walls = set(walls)
        self.terminal = dict(goals); self.terminal.update(lava)
        self.step_reward = step_reward; self.slip = slip
        self.actions_list = ['U','D','L','R']
    def states(self):
        for r in range(1,self.R+1):
            for c in range(1,self.C+1):
                if (r,c) not in self.walls: yield (r,c)
    def is_end(self,s): return s in self.terminal
    def actions(self,s): return [] if self.is_end(s) else self.actions_list
    def _move(self,s,a):
        r,c=s; drc={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}[a]
        rr,cc=r+drc[0],c+drc[1]
        if not (1<=rr<=self.R and 1<=cc<=self.C) or (rr,cc) in self.walls: return s
        return (rr,cc)
    def transitions(self,s,a):
        if self.is_end(s): return
        perp={'U':['L','R'],'D':['L','R'],'L':['U','D'],'R':['U','D']}[a]
        outcomes=[(self._move(s,a),1-self.slip),
                  (self._move(s,perp[0]),self.slip/2.0),
                  (self._move(s,perp[1]),self.slip/2.0)]
        probs={}
        for sp,p in outcomes: probs[sp]=probs.get(sp,0.0)+p
        for sp,p in probs.items():
            r=self.terminal.get(sp,self.step_reward)
            yield (sp,p,r)

def value_iteration(mdp, gamma=0.99, eps=1e-6):
    V={s:0.0 for s in mdp.states()}
    iters=0
    while True:
        iters+=1; delta=0.0
        for s in list(mdp.states()):
            if mdp.is_end(s): V[s]=mdp.terminal[s]; continue
            best=float('-inf')
            for a in mdp.actions(s):
                q=0.0
                for sp,p,r in mdp.transitions(s,a):
                    q+=p*(r+gamma*V[sp])
                if q>best: best=q
            delta=max(delta,abs(best-V[s])); V[s]=best
        if delta<=eps: break
    pi={}
    for s in mdp.states():
        if mdp.is_end(s): continue
        best_a,best_q=None,float('-inf')
        for a in mdp.actions(s):
            q=sum(p*(r+gamma*V[sp]) for sp,p,r in mdp.transitions(s,a))
            if q>best_q: best_q,best_a=q,a
        pi[s]=best_a
    return V,pi,iters

def policy_evaluation(mdp, pi, gamma=0.99, eps=1e-8, max_iters=10000):
    V={s:0.0 for s in mdp.states()}
    for _ in range(max_iters):
        delta=0.0; Vprev=V.copy()
        for s in mdp.states():
            if mdp.is_end(s): V[s]=mdp.terminal[s]; continue
            a=pi[s]
            val=sum(p*(r+gamma*Vprev[sp]) for sp,p,r in mdp.transitions(s,a))
            delta=max(delta,abs(val-Vprev[s])); V[s]=val
        if delta<=eps: break
    return V

def policy_improvement(mdp, V, gamma=0.99):
    pi={}
    for s in mdp.states():
        if mdp.is_end(s): continue
        best_a,best_q=None,float('-inf')
        for a in mdp.actions(s):
            q=sum(p*(r+gamma*V[sp]) for sp,p,r in mdp.transitions(s,a))
            if q>best_q: best_q,best_a=q,a
        pi[s]=best_a
    return pi

def policy_iteration(mdp, gamma=0.99, eval_eps=1e-8, max_pe_iters=1000):
    pi={s:'R' for s in mdp.states() if not mdp.is_end(s)}
    iters=0
    while True:
        iters+=1
        V=policy_evaluation(mdp, pi, gamma=gamma, eps=eval_eps, max_iters=max_pe_iters)
        new_pi=policy_improvement(mdp, V, gamma=gamma)
        if new_pi==pi: break
        pi=new_pi
    return V,pi,iters

def q_value_iteration(mdp, gamma=0.99, eps=1e-6):
    Q={(s,a):0.0 for s in mdp.states() for a in mdp.actions(s)}
    def best_next(sp):
        return 0.0 if mdp.is_end(sp) else max(Q[(sp,a)] for a in mdp.actions(sp))
    iters=0
    while True:
        iters+=1; delta=0.0
        for s in mdp.states():
            if mdp.is_end(s):
                for a in ['U','D','L','R']:
                    if (s,a) in Q: Q[(s,a)]=mdp.terminal[s]
                continue
            for a in mdp.actions(s):
                old=Q[(s,a)]
                new=sum(p*(r+gamma*best_next(sp)) for sp,p,r in mdp.transitions(s,a))
                Q[(s,a)]=new
                delta=max(delta,abs(new-old))
        if delta<=eps: break
    pi={s:max(mdp.actions(s), key=lambda a: Q[(s,a)]) for s in mdp.states() if not mdp.is_end(s)}
    return Q,pi,iters

if __name__=='__main__':
    mdp=GridWorldMDP()
    V_vi,pi_vi,it_vi=value_iteration(mdp,eps=1e-5); print('VI sweeps:',it_vi)
    V_pi,pi_pi,it_pi=policy_iteration(mdp); print('PI iters:',it_pi)
    Q,pi_q,it_q=q_value_iteration(mdp,eps=1e-5); print('Q-Iter iters:',it_q)
