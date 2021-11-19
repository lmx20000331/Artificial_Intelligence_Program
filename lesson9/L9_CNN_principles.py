# Lecture-09: CNN Principles how computer recognize images.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


def conv(image, filters, stride=1):
    conv_results = [conv_(image, f, stride) for f in filters]

    return np.array(conv_results)


def conv_(image, filter, stride=1):
    height = image.shape[0] - filter.shape[0] + 1
    width = image.shape[1] - filter.shape[1] + 1

    conv_result = np.zeros(shape=(height // stride, width // stride))

    for h in range(0, height, stride):
        for w in range(0, width, stride):
            window = image[h: h + filter.shape[0], w: w + filter.shape[1], :]
            conv_result[h][w] = np.sum(np.multiply(window, filter))
            # np.max(window) => max pooling
            # np.mean(window) => mean pooling

    return conv_result


if __name__ == '__main__':
    dog = Image.open('dog.jpg')

    dog = np.array(dog)

    plt.imshow(dog)

    filter_ = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])

    filter_2 = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])

    ic(dog.shape)

    dog_convs = conv(dog, [filter_, filter_2])

    ic(dog_convs.shape)

    flatten = dog_convs.reshape(1, -1)

    outputs = np.matmul(flatten, np.random.random(size=(flatten.shape[1], 5)))

    print(outputs)


