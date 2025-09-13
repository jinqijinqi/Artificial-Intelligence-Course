import numpy as np
def add_bias_1d(x):
    x = np.asarray(x).reshape(-1,1); return np.hstack([np.ones_like(x), x])
def fit_ridge(X, y, lam=0.0, lr=0.1, epochs=300, early_stop=False, patience=20, X_val=None, y_val=None):
    X = np.asarray(X); y = np.asarray(y).reshape(-1)
    n, d = X.shape; w = np.zeros(d); best = (np.inf, w.copy()); wait = 0
    def mse(A,b,w): e=A@w-b; return float((e**2).mean())
    for ep in range(epochs):
        grad = (2.0/n) * (X.T @ (X@w - y)) + lam * w
        w -= lr * grad
        if X_val is not None:
            v = mse(X_val, y_val, w)
            if v < best[0]-1e-10: best = (v, w.copy()); wait=0
            else:
                wait += 1
                if early_stop and wait>=patience: break
    return best[1] if early_stop else w
