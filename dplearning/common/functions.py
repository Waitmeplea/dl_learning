import numpy as np


def softmax(x, c=7):
    x_array = (np.array(x) - c).astype(np.float64)
    exp = np.exp(x_array)
    return exp / np.sum(exp)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
    return -np.sum(np.log(y + 1e-7) * t) / y.shape[0]


def cross_entropy_error_1(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
    return -np.sum(np.log(y(np.arange(y.shape[0]), t) + 1e-7)) / y.shape[0]


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        tmp_val = x[i]
        x[i] = tmp_val + h
        f_r = f(x)

        x[i] = tmp_val - h
        f_l = f(x)
        grad[i] = (f_r - f_l) / (2 * h)
        x[i] = tmp_val
    return grad


def gradient_descent(f, int_x, lr=0.01, step_num=100):
    x = int_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x = x - lr * grad
    return x


if __name__ == "__main__":
    print(softmax([1, 1]))
