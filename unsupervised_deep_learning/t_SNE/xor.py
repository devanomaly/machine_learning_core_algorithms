import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

from util import get_xor

X, Y = get_xor(600)
plt.scatter(X[:,0],X[:,1],c=Y,alpha=0.5, s=100)
plt.show()

tsne = TSNE(perplexity=40)
transformed = tsne.fit_transform(X)

plt.scatter(transformed[:, 0], transformed[:, 1], c=Y, alpha=0.5, s=100)
plt.show()