import numpy as np
from util import get_data, get_xor, get_torus
from time import time


def entropy(y):
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return -((p0 * np.log2(p0)) + (p1 * np.log2(p1)))


class TreeNode:
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X, Y):
        if len(Y) == 1 or len(set(Y)) == 1:
            self.col = None
            self.split = None
            self.left = None
            self.rigth = None
            self.prediction = Y[0]

        else:
            D = X.shape[1]
            cols = range(D)

            max_info_gain = 0
            best_col = None
            best_split = None
            for col in cols:
                info_gain, split = self.find_split(X, Y, col)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_col = col
                    best_split = split

            if max_info_gain == 0:
                self.col = None
                self.split = None
                self.left = None
                self.rigth = None
                self.prediction = np.round(Y.mean())
            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth:
                    self.left = None
                    self.rigth = None
                    self.prediction = [
                        np.round(Y[X[:, best_col] < self.split].mean()),
                        np.round(Y[X[:, best_col] >= self.split].mean()),
                    ]
                else:
                    left_idx = X[:, best_col] < best_split
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = TreeNode(self.depth + 1, self.max_depth)
                    self.left.fit(Xleft, Yleft)

                    right_idx = X[:, best_col] >= best_split
                    Xright = X[right_idx]
                    Yright = Y[right_idx]
                    self.right = TreeNode(self.depth + 1, self.max_depth)
                    self.right.fit(Xright, Yright)

    def find_split(self, X, Y, col):
        x_values = X[:, col]
        sort_index = np.argsort(x_values)
        x_values = x_values[sort_index]
        y_values = Y[sort_index]

        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_info_gain = 0
        for i in boundaries:
            split = (x_values[i] + x_values[i + 1]) / 2
            info_gain = self.info_gain(x_values, y_values, split)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_split = split
        return max_info_gain, best_split

    def info_gain(self, x, y, split):
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(y0len) / N
        p1 = 1 - p0
        return entropy(y) - ((p0 * entropy(y0)) + (p1 * entropy(y1)))

    def predict_one(self, x):
        if (self.col is not None) and (self.split is not None):
            feature = x[self.col]
            if feature < self.split:
                if self.left:
                    p = self.left.predict_one(x)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]
        else:
            p = self.prediction
        return p

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.root = TreeNode(max_depth=self.max_depth)
        self.root.fit(X, Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == "__main__":
    n = None
    # X, Y = get_data(n)
    X, Y = get_xor(n)
    # X, Y = get_torus(n)
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]
    # shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    Ntrain = int(len(Y)*.8)
    print("Ntrain =", Ntrain)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = DecisionTree()
    t0 = time()
    model.fit(Xtrain, Ytrain)
    print("Training time:", time() - t0)

    t0 = time()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to calculate train accuracy:", time() - t0)

    t0 = time()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to calculate Test accuracy:", time() - t0)