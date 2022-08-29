# Here we consider a dataset where KNN isn't the "obvious go-to" classifier
import numpy as np
import matplotlib.pyplot as plt

from knn_mnist import KNN


def get_data():
    # an alternating 8x8 grid
    width = 8
    height = 8
    N = width * height
    X = np.zeros((N, 2))
    Y = np.zeros(N)
    n = 0
    start_t = 0
    for i in range(width):
        t = start_t
        for j in range(height):
            X[n] = [i, j]
            Y[n] = t
            n += 1
            t = (t + 1) % 2
        start_t = (start_t + 1) % 2
    return X, Y


if __name__ == "__main__":
    X, Y = get_data()

    plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
    plt.show()
    for k in (1, 2, 3, 4, 5, 6):
        print("k =", k)
        model = KNN(k)
        model.fit(X, Y)
        print("\tTrain accuracy:", model.score(X, Y))
