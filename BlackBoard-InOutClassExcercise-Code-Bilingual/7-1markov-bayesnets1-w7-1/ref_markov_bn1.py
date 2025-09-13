from typing import Dict, Tuple, List
import itertools, math, random
random.seed(0)

# ----- MRF: object tracking (3 steps) -----
dom = [0,1,2]
obs = {1:0, 2:2, 3:2}  # o=(0,2,2)

def o(i, x):
    return max(0, 2-abs(x-obs[i]))

def t(x, y):
    if x==y: return 2
    if abs(x-y)==1: return 1
    return 0

def weight(assign):  # assign: (x1,x2,x3)
    x1,x2,x3 = assign
    return o(1,x1)*o(2,x2)*o(3,x3)*t(x1,x2)*t(x2,x3)

def enumerate_exact():
    table = []
    Z = 0.0
    for a in itertools.product(dom, repeat=3):
        w = weight(a)
        if w>0:
            table.append((a, w))
            Z += w
    # marginals for X2
    p2 = {v:0.0 for v in dom}
    for (x1,x2,x3), w in table:
        p2[x2] += w/Z
    # max-weight assignment
    max_a, max_w = max(table, key=lambda t: t[1])
    return Z, p2, max_a, max_w

def gibbs(n_iters=5000, burn_in=500):
    # initialize randomly among support
    x = [random.choice(dom) for _ in range(3)]
    # If zero weight, force to support
    def cond_prob(i, x):
        # return distribution over dom for Xi given others
        probs = []
        for v in dom:
            y = x.copy()
            y[i]=v
            # local factors touching i: o_i, t with neighbors
            if i==0:
                w = o(1,v)*t(v,y[1])
            elif i==1:
                w = o(2,v)*t(y[0],v)*t(v,y[2])
            else:
                w = o(3,v)*t(y[1],v)
            probs.append(max(0.0,w))
        s = sum(probs)
        probs = [p/s if s>0 else 1.0/len(dom) for p in probs]
        return probs
    counts = {v:0 for v in dom}
    for it in range(n_iters):
        for i in range(3):
            probs = cond_prob(i, x)
            r = random.random(); c=0.0
            pick = 0
            for idx,p in enumerate(probs):
                c += p
                if r<=c:
                    pick=idx; break
            x[i]=dom[pick]
        if it>=burn_in:
            counts[x[1]] += 1
    total = sum(counts.values())
    p2_hat = {v: counts[v]/total for v in dom}
    return p2_hat

# ----- BN/HMM: H1->H2->H3, emissions E1..E3 -----
def row_norm_row(vs):
    s = sum(vs)
    return [vi/s if s>0 else 1.0/len(vs) for vi in vs]

# transition CPT p(h_{i+1}|h_i) from t
trans = {x: row_norm_row([t(x,y) for y in dom]) for x in dom}
# emission CPT p(e|h) from o
emit = {h: row_norm_row([o(1,h), o(1,h), o(1,h)]) for h in dom}  # same shape for each i; we will index by obs

def forward_backward(evidence):
    # evidence: dict {i: observed value at Ei} for i=1..3
    # prior over H1: uniform
    prior = [1/3]*3
    # forward
    alpha = [{} for _ in range(4)]  # 1..3
    alpha[1] = {h: prior[h]*emit[h][evidence[1]] for h in dom}
    def norm(d):
        s = sum(d.values()); 
        return {k: v/s for k,v in d.items()}
    alpha[1] = norm(alpha[1])
    alpha[2] = {h2: emit[h2][evidence[2]] * sum(alpha[1][h1]*trans[h1][h2] for h1 in dom) for h2 in dom}
    alpha[2] = norm(alpha[2])
    alpha[3] = {h3: emit[h3][evidence[3]] * sum(alpha[2][h2]*trans[h2][h3] for h2 in dom) for h3 in dom}
    alpha[3] = norm(alpha[3])
    # posterior of H2 via one-step smoothing: proportional to alpha2 * backward2
    # backward from the end:
    beta3 = {h:1.0 for h in dom}
    beta2 = {h2: sum(trans[h2][h3]*emit[h3][evidence[3]]*beta3[h3] for h3 in dom) for h2 in dom}
    # combine:
    post2 = {h: alpha[2][h]*beta2[h] for h in dom}
    s = sum(post2.values()); post2 = {k:v/s for k,v in post2.items()}
    return post2

# ----- Alarm BN illustrating explaining away -----
def alarm_probs(eps=0.05):
    # P(B=1|A=1) and P(B=1|A=1,E=1)
    # Using formulas from lecture
    p1 = 1.0/(2.0 - eps)
    p2 = eps
    return p1, p2

if __name__ == "__main__":
    Z, p2, argmax, w = enumerate_exact()
    print("Exact Z:", Z, "P(X2):", p2, "argmax:", argmax, "w:", w)
    print("Gibbs P(X2) ~", gibbs())
    print("BN posterior H2 | E=(0,2,2):", forward_backward({1:0,2:2,3:2}))
    print("Alarm explaining-away:", alarm_probs())
