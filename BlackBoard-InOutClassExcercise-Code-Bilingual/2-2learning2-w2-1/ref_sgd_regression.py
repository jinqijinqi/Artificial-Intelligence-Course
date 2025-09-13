import time, math
import numpy as np

def add_bias_1d(x):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x])

def poly2_1d(x):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x, x**2])

def loss_mse(X, y, w):
    err = X @ w - y
    return float((err**2).mean())

def grad_mse_full(X, y, w):
    n = X.shape[0]
    return (2.0/n) * (X.T @ (X@w - y))

def fit_linear(X, y, method="sgd", lr=0.1, epochs=10, batch_size=32, lr_schedule="constant", seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X); y = np.asarray(y).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    t_updates = 0
    losses = []

    for ep in range(epochs):
        idx = np.arange(n); rng.shuffle(idx)
        if method == "gd":
            eta = lr / math.sqrt(max(1, t_updates)) if lr_schedule=="sqrt_decay" else lr
            g = grad_mse_full(X, y, w)
            w -= eta * g
            t_updates += 1
            losses.append(loss_mse(X, y, w))
        elif method == "sgd":
            for i in idx:
                xi, yi = X[i], y[i]
                eta = lr / math.sqrt(max(1, t_updates)) if lr_schedule=="sqrt_decay" else lr
                g = 2.0 * (xi @ w - yi) * xi
                w -= eta * g
                t_updates += 1
            losses.append(loss_mse(X, y, w))
        else:  # minibatch
            B = max(1, min(batch_size, n))
            for k in range(0, n, B):
                j = idx[k:k+B]
                Xb, yb = X[j], y[j]
                eta = lr / math.sqrt(max(1, t_updates)) if lr_schedule=="sqrt_decay" else lr
                g = (2.0/len(j)) * (Xb.T @ (Xb@w - yb))
                w -= eta * g
                t_updates += 1
            losses.append(loss_mse(X, y, w))

    return w, np.array(losses)
