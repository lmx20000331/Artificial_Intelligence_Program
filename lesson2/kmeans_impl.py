import matplotlib.pyplot as plt
import numpy as np
import random
from icecream import ic
from collections import defaultdict
from matplotlib.colors import BASE_COLORS


points0 = np.random.normal(size=(100, 2))
points1 = np.random.normal(loc=1, size=(100, 2))
points2 = np.random.normal(loc=2, size=(100, 2))
points3 = np.random.normal(loc=5, size=(100, 2))

points = np.concatenate([points0, points1, points2, points3])

k = 3


def random_centers(k, points):
    # step-01

    for i in range(k):
        yield random.choice(points[:, 0]), random.choice(points[:, 1])


def mean(points):
    all_x, all_y = [x for x, y in points], [y for x, y in points]

    return np.mean(all_x), np.mean(all_y)


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return np.sqrt((x1 - x2) ** 2 + (y1 - y2)**2)


def kmeans(k, points, centers=None):
    colors = list(BASE_COLORS.values())

    if not centers:
        centers = list(random_centers(k=k, points=points))

    ic(centers)

    for i, c in enumerate(centers):
        plt.scatter([c[0]], [c[1]], s=90, marker='*', c=colors[i])

    plt.scatter(*zip(*points), c='black')

    centers_neighbor = defaultdict(set)

    for p in points:
        closet_c = min(centers, key=lambda c: distance(p, c))
        centers_neighbor[closet_c].add(tuple(p))

    for i, c in enumerate(centers):
        _points = centers_neighbor[c]
        all_x, all_y = [x for x, y in _points], [y for x, y in _points]
        plt.scatter(all_x, all_y, c=colors[i])

    plt.show()

    new_centers = []

    for c in centers_neighbor:
        new_c = mean(centers_neighbor[c])
        new_centers.append(new_c)

    threshold = 1
    distances_old_and_new = [distance(c_old, c_new) for c_old, c_new in zip(centers, new_centers)]
    ic(distances_old_and_new)
    if all(c < threshold for c in distances_old_and_new):
        return centers_neighbor
    else:
        kmeans(k, points, new_centers)


kmeans(4, points=points)
