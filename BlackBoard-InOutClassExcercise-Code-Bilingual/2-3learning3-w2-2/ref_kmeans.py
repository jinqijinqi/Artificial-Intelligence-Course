import numpy as np
def kmeans_pp_init(X, K, rng):
    n = X.shape[0]; centers = np.empty((K, X.shape[1]))
    i0 = rng.integers(n); centers[0] = X[i0]; d2 = np.full(n, np.inf)
    for k in range(1, K):
        d2 = np.minimum(d2, ((X - centers[k-1])**2).sum(1))
        probs = d2 / d2.sum(); i = rng.choice(n, p=probs); centers[k] = X[i]
    return centers
def kmeans(X, K, init='random', restarts=10, max_iter=100, seed=0):
    rng = np.random.default_rng(seed); best = None
    for _ in range(restarts):
        if init=='k++': centers = kmeans_pp_init(X, K, rng)
        else:
            lo, hi = X.min(0), X.max(0); centers = rng.uniform(lo, hi, size=(K, X.shape[1]))
        for _ in range(max_iter):
            d2 = ((X[:,None,:]-centers[None,:,:])**2).sum(-1)
            assign = d2.argmin(1)
            new = centers.copy()
            for k in range(K):
                idx = np.where(assign==k)[0]
                if len(idx)>0: new[k] = X[idx].mean(0)
            if np.allclose(new, centers): centers = new; break
            centers = new
        loss = ((X - centers[assign])**2).sum()
        if best is None or loss < best[-1]: best = (assign, centers, float(loss))
    return best
