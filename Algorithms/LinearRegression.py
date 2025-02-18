import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int):
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape(-1, 1)
    
    for _ in range(iterations):
        gradients = (1/m) * X.T @ (X @ theta - y)
        theta -= alpha * gradients

    return theta.flatten()

import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X)
    y = np.array(y)

    theta = np.linalg.pinv(X.T @ X) @ X.T @ y

    return theta.round(3).tolist()
