from sklearn.datasets import load_iris
from icecream import ic
from collections import Counter
import numpy as np
import pandas as pd

# iris_x = load_iris()['data']
# iris_y = load_iris()['target']


def knn(x, y, query, k=3, clf=True):
    history = {tuple(x_): y_ for x_, y_ in zip(x, y)}
    neighbors = sorted(history.items(), key=lambda x_y: np.sum((np.array(x_y[0]) - np.array(query)) ** 2))[:k]
    neighbors_y = [y for x, y in neighbors]

    if clf: return Counter(neighbors_y).most_common(1)[0][0]
    else:
        return np.mean(neighbors_y)

# decision tree explain


def get_pros(elements):
    counter = Counter(elements)
    pr = np.array([counter[c] / len(elements) for c in counter])

    return pr


def entropy(elements):
    pr = get_pros(elements)
    return -np.sum(pr * np.log2(pr))


def gini(elements):
    pr = get_pros(elements)
    return 1 - np.sum(pr**2)




def cart_loss(left, right, pure_fn):
    m_left, m_right = len(left), len(right)
    m = m_left + m_right

    return m_left / m * pure_fn(left) + m_right / m * pure_fn(right)

sales = {
    'gender': ['Female', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male'],
    'income': ['H', 'M', 'H', 'M', 'H', 'H', 'L'],
    'family-number': [1, 1, 2, 1, 1, 1, 2],
    'bought': [1, 1, 1, 0, 0, 0, 1]
}

sales_dataset = pd.DataFrame.from_dict(sales)
target = 'bought'


def find_best_split(training_dataset, target):
    dataset = training_dataset
    fields = set(dataset.columns.tolist()) - {target}
    print(fields)

    mini_loss = float('inf')
    best_feature, best_split = None, None

    for x in fields:
        filed_value = dataset[x]
        for v in filed_value:
            split_left = dataset[dataset[x] == v][target].tolist()
            split_right = dataset[dataset[x] != v][target].tolist()

            loss = cart_loss(split_left, split_right, pure_fn=gini)
            ic(x, v, cart_loss(split_left, split_right, pure_fn=gini))
            if loss < mini_loss:
                best_feature, best_split = x, v

    return best_feature, best_split


if __name__ == '__main__':
    # ic(entropy([1, 1]))
    # ic(gini([1, 1]))
    # ic(entropy([0, 0]))
    # ic(gini([0, 0]))
    # ic(entropy([0, 0, 1, 1, 1,1 ,1, 1]))
    # ic(gini([0, 0, 1, 1, 1,1 ,1, 1]))
    # ic(entropy([0, 0, 0, 0, 0, 0, 0, 0]))
    # ic(gini([0, 0, 0, 0, 0, 0, 0, 0]))
    # ic(entropy([1, 2, 3, 4, 56, 7, 8, 1, 19]))
    # ic(gini([1, 2, 3, 4, 56, 7, 8, 1, 19]))
    # ic(entropy([1, 2, 3, 4, 65, 76, 87, 32, 21]))
    # ic(gini([1, 2, 3, 4, 65, 76, 87, 32, 21]))

    ic(find_best_split(sales_dataset, target='bought'))