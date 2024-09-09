import numpy as np

class Model():
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, x, y):
        B, N = x.shape
        
        self.w = np.zeros(N)
        self.b = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(x):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.sum(predictions == y) / len(y)