import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from util import get_xor, plot_2D_decision_boundary, get_torus



class Bagged_DT_Classifier:
    def __init__(self, B):
        self.B = B

    def fit(self, X, Y):
        N = len(X)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]

            model = (
                DecisionTreeClassifier()
            )  # let each of them overfit its own data! (locally low bias)
            model.fit(Xb, Yb)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return np.round(
            predictions / self.B
        )  # the bagged prediction lowers the variance

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)


if __name__ == "__main__":
    N = 500
    X, Y = get_xor(N)
    # X, Y = get_torus(N)

    # let's see how does a single decision tree perform

    model = DecisionTreeClassifier()  # a low bias and high variance should be expected!
    model.fit(X, Y)
    print("Score - single DT:", model.score(X, Y))

    plt.title("Single DT classification")
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, alpha=0.3)
    plot_2D_decision_boundary(X, model)
    plt.legend()
    plt.show()
    B = 200
    model = Bagged_DT_Classifier(B)
    model.fit(X, Y)
    print("Score - " + str(B) + " DTs:", model.score(X, Y))

    plt.title(str(B) + "-DTs model")
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, alpha=0.3)
    plot_2D_decision_boundary(X, model)
    plt.show()
    plt.close()
