from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import math, random

# Domains
G_vals = ["c","d"]
R_vals = [1,2,3,4,5]

Example = Tuple[Optional[str], int, int]  # (G or None, R1, R2)

def normalize(d: Dict):
    s = sum(d.values())
    if s == 0:
        # uniform fallback
        n = len(d)
        for k in d:
            d[k] = 1.0/n
    else:
        for k in d:
            d[k] /= s
    return d

def fit_mle(supervised: List[Example], lambda_: float=0.0, share_R: bool=True):
    """
    Fully observed MLE with optional Laplace smoothing and parameter sharing for p_R.
    Returns: pG (dict), pR (dict g-> {r: prob})
    """
    # Counts
    countG = defaultdict(float, {g: 0.0 for g in G_vals})
    if share_R:
        countR = {g: defaultdict(float, {r: 0.0 for r in R_vals}) for g in G_vals}
    else:
        countR1 = {g: defaultdict(float, {r: 0.0 for r in R_vals}) for g in G_vals}
        countR2 = {g: defaultdict(float, {r: 0.0 for r in R_vals}) for g in G_vals}
    # Laplace preload
    for g in G_vals:
        countG[g] += lambda_
        if share_R:
            for r in R_vals:
                countR[g][r] += lambda_
        else:
            for r in R_vals:
                countR1[g][r] += lambda_
                countR2[g][r] += lambda_
    # Tally data
    for g, r1, r2 in supervised:
        assert g is not None, "fit_mle expects fully observed data"
        countG[g] += 1
        if share_R:
            countR[g][r1] += 1
            countR[g][r2] += 1
        else:
            countR1[g][r1] += 1
            countR2[g][r2] += 1
    # Normalize
    pG = normalize(dict(countG))
    if share_R:
        pR = {g: normalize(dict(countR[g])) for g in G_vals}
    else:
        pR = {"R1": {g: normalize(dict(countR1[g])) for g in G_vals},
              "R2": {g: normalize(dict(countR2[g])) for g in G_vals}}
    return pG, pR

def log_likelihood(mixed: List[Example], pG, pR, share_R: bool=True):
    ll = 0.0
    for g_obs, r1, r2 in mixed:
        if g_obs is None:
            # sum over g
            s = 0.0
            for g in G_vals:
                if share_R:
                    s += pG[g]*pR[g][r1]*pR[g][r2]
                else:
                    s += pG[g]*pR["R1"][g][r1]*pR["R2"][g][r2]
            ll += math.log(max(s, 1e-12))
        else:
            g = g_obs
            if share_R:
                prob = pG[g]*pR[g][r1]*pR[g][r2]
            else:
                prob = pG[g]*pR["R1"][g][r1]*pR["R2"][g][r2]
            ll += math.log(max(prob, 1e-12))
    return ll

def e_step_posteriors(mixed: List[Example], pG, pR, share_R=True):
    """Return list of posteriors q for each example (dict over g), using current params."""
    qs = []
    for g_obs, r1, r2 in mixed:
        if g_obs is not None:
            q = {g: 1.0 if g==g_obs else 0.0 for g in G_vals}
        else:
            un = {}
            for g in G_vals:
                if share_R:
                    un[g] = pG[g]*pR[g][r1]*pR[g][r2]
                else:
                    un[g] = pG[g]*pR["R1"][g][r1]*pR["R2"][g][r2]
            s = sum(un.values())
            q = {g: (un[g]/s if s>0 else 1.0/len(G_vals)) for g in G_vals}
        qs.append(q)
    return qs

def m_step(mixed: List[Example], qs, lambda_: float=0.0, share_R: bool=True):
    # fractional counts with Laplace preload
    countG = defaultdict(float, {g: lambda_ for g in G_vals})
    if share_R:
        countR = {g: defaultdict(float, {r: lambda_ for r in R_vals}) for g in G_vals}
    else:
        countR1 = {g: defaultdict(float, {r: lambda_ for r in R_vals}) for g in G_vals}
        countR2 = {g: defaultdict(float, {r: lambda_ for r in R_vals}) for g in G_vals}
    for (g_obs, r1, r2), q in zip(mixed, qs):
        for g in G_vals:
            w = q[g]
            countG[g] += w
            if share_R:
                countR[g][r1] += w
                countR[g][r2] += w
            else:
                countR1[g][r1] += w
                countR2[g][r2] += w
    pG = normalize(dict(countG))
    if share_R:
        pR = {g: normalize(dict(countR[g])) for g in G_vals}
    else:
        pR = {"R1": {g: normalize(dict(countR1[g])) for g in G_vals},
              "R2": {g: normalize(dict(countR2[g])) for g in G_vals}}
    return pG, pR

def fit_em(mixed: List[Example], init=None, lambda_: float=0.0, iters: int=5, share_R=True):
    if init is None:
        # uniform init
        if share_R:
            pR = {g: {r: 1.0/len(R_vals) for r in R_vals} for g in G_vals}
        else:
            pR = {"R1": {g: {r: 1.0/len(R_vals) for r in R_vals} for g in G_vals},
                  "R2": {g: {r: 1.0/len(R_vals) for r in R_vals} for g in G_vals}}
        pG = {g: 1.0/len(G_vals) for g in G_vals}
    else:
        pG, pR = init
    history = [log_likelihood(mixed, pG, pR, share_R=share_R)]
    for _ in range(iters):
        qs = e_step_posteriors(mixed, pG, pR, share_R=share_R)
        pG, pR = m_step(mixed, qs, lambda_=lambda_, share_R=share_R)
        history.append(log_likelihood(mixed, pG, pR, share_R=share_R))
    return (pG, pR), history

if __name__ == "__main__":
    supervised = [("d",4,5),("d",4,4),("d",5,3),("c",1,2),("c",5,4)]
    pG, pR = fit_mle(supervised, lambda_=0.0, share_R=True)
    print("MLE pG:", pG); print("MLE pR:", pR)
    pG1, pR1 = fit_mle(supervised, lambda_=1.0, share_R=True)
    print("Laplace(1) pR for d:", pR1["d"])
    mixed = supervised + [(None,2,2),(None,1,2)]
    (pG_em, pR_em), hist = fit_em(mixed, init=(pG1,pR1), lambda_=1.0, iters=3, share_R=True)
    print("EM pG:", pG_em); print("LL hist:", hist)
