import numpy as np

def phi_linear_1d(x):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x])

def phi_poly2_1d(x):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x, x**2])

def phi_bins_1d(x, B=5, lo=0.0, hi=5.0):
    x = np.asarray(x).reshape(-1)
    edges = np.linspace(lo, hi, B+1)
    X = np.zeros((len(x), B))
    for i, xi in enumerate(x):
        b = np.searchsorted(edges, xi, side="right") - 1
        b = min(max(b, 0), B-1)
        X[i, b] = 1.0
    return np.hstack([np.ones((len(x),1)), X])

def phi_periodic_1d(x, omega=3.0):
    x = np.asarray(x).reshape(-1,1)
    return np.hstack([np.ones_like(x), x, x**2, np.cos(omega*x)])
