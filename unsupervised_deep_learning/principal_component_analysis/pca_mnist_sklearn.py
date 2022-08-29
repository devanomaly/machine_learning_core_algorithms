# Here we use PCA on the MNIST dataset, using the sklearn implementation of PCA 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from auxiliary import get_mnist

Xtrain, Ytrain, Xtest, Ytest = get_mnist()

pca = PCA()
reduced = pca.fit_transform(Xtrain)
# let us see how does MNIST data is understood with two dimensions (two features) only
plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Ytrain, alpha=0.5)
plt.show()

pca_explained_variance_ratio = pca.explained_variance_ratio_
plt.plot(pca_explained_variance_ratio)
plt.show()

# we now check the cumulative variance for each number of dimensions k
# a good PCA usually means finding the minimum k such that you can "reconstruct" your data with 95-99% of variance

cumulative = []
last = 0
k=1
for v in pca_explained_variance_ratio:
  cumulative.append(last+v)
  last = cumulative[-1]
  if (0.95 < last < 0.96):
    print("least k is",k," - with variance =",last)
    # break
  k+=1
plt.plot(cumulative)
plt.show()