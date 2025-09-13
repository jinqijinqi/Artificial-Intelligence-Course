import numpy as np

def add_bias(x):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x])

def fit_linear_gd(X, y, lr=0.1, epochs=200):
    X = np.asarray(X); y = np.asarray(y).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    hist = []
    for _ in range(epochs):
        pred = X @ w
        err = pred - y
        loss = (err**2).mean()
        grad = (2.0/n) * (X.T @ err)
        w -= lr * grad
        hist.append(loss)
    return w, np.array(hist)

if __name__ == "__main__":
    # Toy dataset from slides
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([1.0, 3.0, 3.0])
    X = add_bias(x)
    w, hist = fit_linear_gd(X, y, lr=0.1, epochs=200)
    print("w* =", w)
    print("final loss =", hist[-1])
