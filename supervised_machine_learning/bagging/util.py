from numpy import array, pi, concatenate, cos, sin, meshgrid, arange, c_, zeros
from numpy.random import random, randn
import matplotlib.pyplot as plt


def plot_2D_decision_boundary(X, model):
  h = .02  # step size in the mesh
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = meshgrid(arange(x_min, x_max, h),
                       arange(y_min, y_max, h))


  # Plot the decision boundary. For that, we will assign a color to each
  # point in the mesh [x_min, m_max]x[y_min, y_max].
  Z = model.predict(c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


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


def get_xor(N=None):
    # the celebrated xor dataset
    N = int(N) if N is not None else 300
    sep=1
    X = zeros((N, 2))
    X[: N // 4] = random((N // 4, 2)) / 2 + sep
    X[N // 4 : N // 2] = random((N // 4, 2)) / 2
    X[N // 2 : 3 * N // 4] = random((N // 4, 2)) / 2 + array([[0, sep]])
    X[3 * N // 4 :] = random((N // 4, 2)) / 2 + array([[sep, 0]])
    Y = array([0] * (N // 2) + [1] * (N // 2))
    return X, Y
