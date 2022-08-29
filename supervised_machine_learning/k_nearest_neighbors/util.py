import pandas as pd
from numpy import zeros, array, pi, concatenate, cos, sin
from numpy.random import shuffle, random, randn


def get_data(limit=None):
    df = pd.read_csv("../data/mnist-train/train.csv")
    data = df.to_numpy()
    shuffle(data)
    X = data[:, 1:] / 255.0  # normalizing the pixelation
    Y = data[:, 0]  # labels in the first column
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


def get_xor(N=None):
    # the celebrated xor dataset
    N = int(N) if N is not None else 300
    X = zeros((N, 2))
    X[: N // 4] = random((N // 4, 2)) / 2 + 0.5
    X[N // 4 : N // 2] = random((N // 4, 2)) / 2
    X[N // 2 : 3 * N // 4] = random((N // 4, 2)) / 2 + array([[0, 0.5]])
    X[3 * N // 4 :] = random((N // 4, 2)) / 2 + array([[0.5, 0]])
    Y = array([0] * (N // 2) + [1] * (N // 2))
    return X, Y


def get_torus(N):
    # we generate a dataset with toroidal (donut-like) geometry
    N = int(N) if N is not None else 300
    N_2 = N // 2

    R_int = 5
    R_ext = 10

    R1 = randn(N_2) + R_int
    theta = 2 * pi * random(N_2)
    X_int = concatenate([[R1 * cos(theta)], [R1 * sin(theta)]]).T

    R2 = randn(N_2) + R_ext
    theta = 2 * pi * random(N_2)
    X_ext = concatenate([[R2 * cos(theta)], [R2 * sin(theta)]]).T

    X = concatenate([X_int, X_ext])
    Y = array([0] * (N_2) + [1] * (N_2))

    return X, Y
