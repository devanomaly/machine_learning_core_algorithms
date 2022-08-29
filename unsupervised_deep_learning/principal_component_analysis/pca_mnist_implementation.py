# We implement PCA directly and then use it on the MNIST dataset (please, see the file "pca_mnist_sklearn.py" and compare the performances on your own if you'd like :D)

import numpy as np
import matplotlib.pyplot as plt

from auxiliary import get_mnist

Xtrain, Ytrain, Xtest, Ytest = get_mnist()

# the covariance matrix and its decomposition
covX = np.cov(Xtrain.T)
lambdas, Q = np.linalg.eigh(covX)

# the eigenvalues (variances) are sorted from smallest to largest, and are positive definite (with small negative values occurring due to limited precision)
eigvals_indexes = np.argsort(-lambdas)
lambdas = lambdas[eigvals_indexes] #sorting the eigenvalues
lambdas = np.maximum(lambdas, 0) #only get positive values
Q = Q[:, eigvals_indexes]

# let us visualize the two "largest" dimensions
Z = Xtrain.dot(Q)
plt.scatter(Z[:,0], Z[:,1], s=100, c=Ytrain, alpha=0.3)
plt.show()

# the eigenvalues (unnormalized variances)
plt.plot(lambdas)
plt.title("Components variances")
plt.show()

# cumulative (unormalized) variances
plt.plot(np.cumsum(lambdas))
plt.title("Cumulative variance")
plt.show()