import numpy as np
from icecream import ic

X = np.random.normal(size=(10, 7))
y = np.array([
    [1],
    [0],
    [0],
    [0],
    [1],
    [0],
    [0],
    [1],
    [0],
    [0],
])

weights = np.random.normal(size=(1, 7))
bias = 0


def loss(yhats, y):
    return np.mean( (yhats - y) ** 2 )


def partial_w(yhats, y, train_x):
    return 2 * np.mean((yhats - y) * train_x, axis=0)


def partial_b(yhats, y):
    return 2 * np.mean(yhats - y)


def logistic(x):
    return 1 / (1 + np.exp(-x))

#
# def softmax(x):
#     x -= np.max(x)
#     sum = np.sum(np.exp(x))
#
#     return np.exp(x) / sum
def softmax(x):
    x -= np.max(x, axis=1).reshape(x.shape[0], 1)
    x_sum = np.sum(np.exp(x), axis=1)
    return np.exp(x) / x_sum.reshape(x.shape[0], 1)

def cross_entropy(yhats, y):
    return - np.mean( y * np.log(yhats))


def train_linear_regression(X, weights, bias, y):
    for i in range(10):
        # yhats = X @ weights.T + bias
        yhats = logistic(X @ weights.T + bias)
        threshold = 0.5
        probs = np.array((yhats > threshold), dtype=np.int)
        ic(probs)
        # loss = cross_entropy(yhats, y)
        # loss_value = loss(yhats, y)
        # ic(loss_value)
        # learning_rate = 1e-3
        # weights += -1 * partial_w(yhats, y, X) * learning_rate
        # bias += -1 * partial_b(yhats, y)
        ic(yhats)
        # ic(loss)


if __name__ == '__main__':
    train_linear_regression(X, weights, bias, y)
