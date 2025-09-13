import numpy as np
def sigmoid(z): return 1/(1+np.exp(-z))
def forward(phi, V, w, y):
    z1 = V @ phi; h = sigmoid(z1); s = float(w @ h)
    residual = s - float(y); loss = residual**2
    return (phi, V, w, y, z1, h, s, residual, loss)
def backward(cache):
    phi, V, w, y, z1, h, s, residual, loss = cache
    grad_w = 2.0 * residual * h
    g = 2.0 * residual * (w * h * (1-h))
    grad_V = np.outer(g, phi)
    return grad_w, grad_V, loss
