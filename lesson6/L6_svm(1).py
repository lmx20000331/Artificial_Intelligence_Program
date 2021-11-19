import numpy as np
import matplotlib.pyplot as plt

label_a = np.random.normal(6, 2, size=(50, 2))
label_b = np.random.normal(-6, 2, size=(50, 2))

plt.scatter(*zip(*label_a))
plt.scatter(*zip(*label_b))

# plt.show()

k_and_b = []

label_a_x = label_a[:, 0]
label_b_x = label_b[:, 0]


def f(x, k, b):
    return k * x + b


for i in range(100):
    k, b = (np.random.random(size=(1, 2)) * 10 - 5)[0]
    print(k, b)

    if np.max(f(label_a_x, k, b)) <= -1 and np.min(f(label_b_x, k, b)) >= 1:
        print(k, b)
        k_and_b.append((k, b))

x = np.concatenate((label_a_x, label_b_x))

for k, b in k_and_b:
    plt.plot(x, f(x, k, b))

k_star, b_star = sorted(k_and_b, key=lambda t: abs(t[0]))[0]

plt.plot(x, f(x, k_star, b_star), '-o')

plt.show()