import numpy as np
def relu(z): return np.maximum(z, 0.0)
def d_relu(z): return (z > 0).astype(float)

class TinyTwoLayer:
    def __init__(self, d_in, d_h=4, lr=0.1, epochs=200, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(scale=0.5, size=(d_h, d_in)); self.b1 = np.zeros(d_h)
        self.W2 = rng.normal(scale=0.5, size=(1, d_h));    self.b2 = np.zeros(1)
        self.lr, self.epochs = lr, epochs
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).reshape(-1)  # {0,1}
        for _ in range(self.epochs):
            z1 = X @ self.W1.T + self.b1; h = relu(z1)
            z2 = h @ self.W2.T + self.b2
            yhat = 1/(1+np.exp(-z2)).reshape(-1)
            dz2 = (yhat - y)[:,None]
            gW2 = dz2.T @ h / len(y); gb2 = dz2.mean(0)
            dh  = dz2 @ self.W2; dz1 = dh * d_relu(z1)
            gW1 = dz1.T @ X / len(y); gb1 = dz1.mean(0)
            self.W2 -= self.lr*gW2; self.b2 -= self.lr*gb2
            self.W1 -= self.lr*gW1; self.b1 -= self.lr*gb1
        return self
    def predict(self, X):
        z1 = X @ self.W1.T + self.b1; h = relu(z1)
        z2 = h @ self.W2.T + self.b2
        yhat = 1/(1+np.exp(-z2)).reshape(-1)
        return (yhat >= 0.5).astype(int)
