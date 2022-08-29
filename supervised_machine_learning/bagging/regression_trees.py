import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle



class Bagged_DT_Regressor:
    def __init__(self, B):
        self.B = B

    def fit(self, X, Y):
        N = len(X)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]

            model = DecisionTreeRegressor()
            model.fit(Xb, Yb)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return predictions / self.B

    def score(self, X, Y):
        d1 = Y - self.predict(X)
        d2 = Y - Y.mean()
        return 1 - (d1.dot(d1) / d2.dot(d2))


if __name__ == "__main__":
    T = 1001
    x_axis = np.linspace(0, 2 * np.pi, T)
    y_axis = np.sin(x_axis)

    N = int(T * 0.3)
    idx = np.random.choice(T, size=N, replace=False)
    Xtrain = x_axis[idx].reshape(N, 1)
    Ytrain = y_axis[idx]

    # let's see how does a single decision tree performs

    model = DecisionTreeRegressor()  # a low bias and high variance should be expected!
    model.fit(Xtrain, Ytrain)
    prediction = model.predict(x_axis.reshape(T, 1))
    print("Score - single DT:", model.score(x_axis.reshape(T, 1), y_axis))

    plt.plot(x_axis, prediction, label="1-DT model", c="red")
    plt.scatter(x_axis, y_axis)
    plt.legend()
    plt.show()
    B = 1000
    model = Bagged_DT_Regressor(B)
    model.fit(Xtrain, Ytrain)
    prediction = model.predict(x_axis.reshape(T, 1))
    print("Score - " + str(B) + " DTs:", model.score(x_axis.reshape(T, 1), y_axis))

    plt.plot(x_axis, prediction, label=str(B) + "-DTs model", c="red")
    plt.scatter(x_axis, y_axis)
    plt.legend()
    plt.show()
