import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

from util import get_mnist

Xtrain, Ytrain, _, _ = get_mnist()

sample_size = 1000
X = Xtrain[:sample_size]
Y = Ytrain[:sample_size]

tsne = TSNE()
transformed = tsne.fit_transform(X)

plt.scatter(transformed[:, 0], transformed[:, 1], c=Y, alpha=0.5, s=100)
plt.show()