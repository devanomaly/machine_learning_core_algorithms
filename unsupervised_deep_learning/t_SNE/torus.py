import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

from util import get_torus

X, Y = get_torus(600)
plt.scatter(X[:,0],X[:,1],c=Y,alpha=0.6)
plt.show()

tsne = TSNE(perplexity=40)
transformed = tsne.fit_transform(X)

plt.scatter(transformed[:, 0], transformed[:, 1], c=Y, alpha=0.2)
plt.show()