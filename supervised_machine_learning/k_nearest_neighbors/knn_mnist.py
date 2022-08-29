# here we use the KNN classifier on MNIST dataset
# exploration of the hyperparameter k is done
# import numpy as np
from numpy import zeros, mean
from sortedcontainers import SortedList
from time import time
from util import get_data


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.X = x
        self.y = y

    def predict(self, X):
        y = zeros(len(X))
        for j, x in enumerate(X):
            sl = SortedList()
            sl._reset(self.k)  # setting load to k
            for n, xt in enumerate(self.X):
                diff = x - xt
                d = diff.dot(diff)  # square distance
                if len(sl) < self.k:
                    sl.add((d, self.y[n]))
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[n]))
            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v, 0) + 1
            max_votes = 0
            max_votes_class = -1

            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[j] = max_votes_class
        return y

    def score(self, X, Y):
        P = self.predict(X)
        return mean(P == Y)


if __name__ == "__main__":
    N = 5000
    X, Y = get_data(N)
    Ntrain = int(0.8 * N)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    for k in (1, 2, 3, 4, 5):
        print("k =", k)
        knn = KNN(k)
        t0 = time()
        knn.fit(Xtrain, Ytrain)
        print("\tTraining time:", time() - t0)
        print("\tTrain size =", len(Ytrain))
        print("\tTest size =", len(Ytest))
        t0 = time()
        print("\tTrain accuracy:", knn.score(Xtrain, Ytrain))
        print("\tTime to calculate train accuracy:", time() - t0)

        t0 = time()
        print("\tTest accuracy:", knn.score(Xtest, Ytest))
        print("\tTime to calculate test accuracy:", time() - t0)
