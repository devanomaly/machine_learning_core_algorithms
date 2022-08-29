from knn_mnist import KNN
from util import get_torus
import matplotlib.pyplot as plt

if __name__ == '__main__':
  n = 1000
  X, Y = get_torus(n)
  plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
  plt.show()
  
  for k in range(1,7):
    print("k =", k)
    model = KNN(k)
    model.fit(X, Y)
    print("\tTrain accuracy:", model.score(X,Y))