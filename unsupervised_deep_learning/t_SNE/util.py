import pandas as pd
from numpy import array, pi, concatenate, cos, sin, zeros
from numpy.random import random, randn
from sklearn.utils import shuffle

def get_torus(N):
    # we generate a dataset with toroidal (donut-like) geometry
    N = int(N) if N is not None else 300
    N_2 = N // 2

    R_int = 10
    R_ext = 20

    R1 = randn(N_2) + R_int
    theta = 2 * pi * random(N_2)
    X_int = concatenate([[R1 * cos(theta)], [R1 * sin(theta)]]).T

    R2 = randn(N_2) + R_ext
    theta = 2 * pi * random(N_2)
    X_ext = concatenate([[R2 * cos(theta)], [R2 * sin(theta)]]).T

    X = concatenate([X_int, X_ext])
    Y = array([0] * (N_2) + [1] * (N_2))

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

def get_mnist():
  df = pd.read_csv("../data/mnist-train/train.csv").to_numpy()
  df = shuffle(df)
  # print(df.head())
  Xtrain, Ytrain = df[:-1000,1:]/255, df[:-1000,0]
  Xtest, Ytest = df[-1000:,1:]/255, df[-1000:,0]
  
  return Xtrain, Ytrain, Xtest, Ytest