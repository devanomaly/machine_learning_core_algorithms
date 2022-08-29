# here we apply gaussian naive bayes on the MNIST dataset, first without and then with PCA
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from auxiliary import get_mnist
from sklearn.decomposition import PCA


class GaussianNB(object):
    def fit(self, X, Y, smoothing=0.05):
        self.gaussians = {}
        self.priors = {}
        labels = set(Y)
        size_Y = len(Y)
        for c in labels:
            currentX = X[Y == c]
            self.gaussians[c] = {
                "mean": currentX.mean(axis=0),
                "var": currentX.var(axis=0) + smoothing,
            }
            self.priors[c] = len(Y[Y == c]) / size_Y
        return None

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, var = g["mean"], g["var"]
            P[:, c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis=1)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


# load the data
Xtrain, Ytrain, Xtest, Ytest = get_mnist()

# Naive Bayes without PCA
model = GaussianNB()
model.fit(Xtrain, Ytrain)
print(
    "gaussian nb train score:", model.score(Xtrain, Ytrain)
)
print(
    "gaussian nb test score:", model.score(Xtest, Ytest)
)
# Now, we use PCA
n_components = 50
pca = PCA(n_components=n_components)
Ztrain = pca.fit_transform(Xtrain)
Ztest = pca.transform(Xtest)

model2 = GaussianNB()
model2.fit(Ztrain, Ytrain)
print(
    "gaussian nb+pca(" + (n_components) + " components) train score:",
    model2.score(Ztrain, Ytrain),
)
print("gaussian nb+pca(" + (n_components) + " components) test score:", model2.score(Ztest, Ytest))
