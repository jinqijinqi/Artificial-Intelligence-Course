from typing import Dict, List, Tuple
import random, math
random.seed(0)

# ----- Probabilistic programs -----
def bernoulli(eps: float) -> int:
    return 1 if random.random() < eps else 0

def sample_alarm(eps=0.05):
    B = bernoulli(eps)
    E = bernoulli(eps)
    A = 1 if (B or E) else 0
    return {"B":B,"E":E,"A":A}

def sample_hmm(T=3, domain=(0,1,2)):
    def trans(h_prev, h):
        if h==h_prev: return 0.5
        if abs(h-h_prev)==1: return 0.25
        return 0.0
    def emit(h, e):
        if e==h: return 0.5
        if abs(e-h)==1: return 0.25
        return 0.0
    H = [random.choice(domain)]
    E = [random.choice(domain)]
    # redraw E[0] conditioned on H[0]
    E[0] = random.choices(domain, [emit(H[0],e) for e in domain])[0]
    for i in range(1,T):
        H.append(random.choices(domain, [trans(H[i-1],h) for h in domain])[0])
        E.append(random.choices(domain, [emit(H[i],e) for e in domain])[0])
    return H, E

# ----- HMM Forward–Backward -----
def forward_backward(evidence: List[int], domain=(0,1,2)):
    n = len(evidence)
    def trans(hp, h):
        if h==hp: return 0.5
        if abs(h-hp)==1: return 0.25
        return 0.0
    def emit(h, e):
        if e==h: return 0.5
        if abs(e-h)==1: return 0.25
        return 0.0
    # prior uniform
    prior = {h:1/len(domain) for h in domain}
    F = [ {h:0.0 for h in domain} for _ in range(n) ]
    B = [ {h:1.0 for h in domain} for _ in range(n) ]
    # forward
    for h in domain:
        F[0][h] = prior[h] * emit(h, evidence[0])
    # normalize
    s = sum(F[0].values());  F[0] = {h: F[0][h]/s for h in domain}
    for i in range(1,n):
        for h in domain:
            F[i][h] = emit(h, evidence[i]) * sum(F[i-1][hp]*trans(hp,h) for hp in domain)
        s = sum(F[i].values());  F[i] = {h: F[i][h]/s for h in domain}
    # backward
    for i in reversed(range(n-1)):
        for h in domain:
            B[i][h] = sum(B[i+1][hn]*trans(h,hn)*emit(hn, evidence[i+1]) for hn in domain)
        s = sum(B[i].values());  B[i] = {h: B[i][h]/s for h in domain}
    # smoothing
    post = []
    for i in range(n):
        S = {h: F[i][h]*B[i][h] for h in domain}
        s = sum(S.values()); S = {h: S[h]/s for h in domain}
        post.append(S)
    return F, B, post

# ----- Particle Filter (filtering) -----
def particle_filter(evidence: List[int], K=200, domain=(0,1,2), seed=0):
    random.seed(seed)
    def trans(hp, h):
        if h==hp: return 0.5
        if abs(h-hp)==1: return 0.25
        return 0.0
    def emit(h, e):
        if e==h: return 0.5
        if abs(e-h)==1: return 0.25
        return 0.0
    # initialize H1 ~ prior uniform but weight by emission
    particles = random.choices(domain, k=K)
    weights = [emit(h, evidence[0]) for h in particles]
    # resample
    def resample(parts, ws):
        s = sum(ws)
        if s==0: ws = [1.0/len(ws)]*len(ws)
        else: ws = [w/s for w in ws]
        # multinomial resampling
        cs = []
        c=0.0
        for w in ws:
            c+=w; cs.append(c)
        new = []
        for _ in parts:
            r = random.random()
            j=0
            while r>cs[j]: j+=1
            new.append(parts[j])
        return new
    particles = resample(particles, weights)
    # iterate
    for i in range(1, len(evidence)):
        # propose
        proposed = []
        for hprev in particles:
            proposed.append(random.choices(domain, [trans(hprev,h) for h in domain])[0])
        # weight by emission
        weights = [emit(h, evidence[i]) for h in proposed]
        particles = resample(proposed, weights)
    # counts for last Hi
    counts = {h:0 for h in domain}
    for h in particles: counts[h]+=1
    total = sum(counts.values())
    approx = {h: counts[h]/total for h in domain}
    return approx, counts

# ----- Gibbs on tiny medical BN: C,A cause H,I; evidence H=1,I=1 -----
def gibbs_CA(num_iters=5000, burn=500, seed=0):
    random.seed(seed)
    # priors p(C=1)=0.1, p(A=1)=0.3; conditionals:
    pC = 0.1; pA=0.3
    # p(H=1|C,A): OR-like
    def pH(c,a): return 0.9 if (c or a) else 0.1
    # p(I=1|A): itchy if allergies
    def pI(a): return 0.8 if a==1 else 0.2
    # evidence H=1, I=1
    c,a = 0,1
    cntC1=0
    for it in range(num_iters):
        # sample C | A,H=1,I=1 ∝ p(C)p(H=1|C,A)
        w0 = (1-pC)*pH(0,a)
        w1 = pC*pH(1,a)
        s = w0+w1
        c = 1 if random.random() < (w1/s) else 0
        # sample A | C,H=1,I=1 ∝ p(A)p(H=1|C,A)p(I=1|A)
        w0 = (1-pA)*pH(c,0)*pI(0)
        w1 = pA*pH(c,1)*pI(1)
        s = w0+w1
        a = 1 if random.random() < (w1/s) else 0
        if it>=burn:
            cntC1 += c
    return cntC1/(num_iters-burn)
