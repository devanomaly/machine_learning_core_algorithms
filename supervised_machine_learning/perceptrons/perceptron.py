import numpy as np
import matplotlib.pyplot as plt
from util import get_data, get_xor
from time import time


def linear_data():
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = (np.random.random((300, 2)) * 2) - 1
    Y = np.sign(X.dot(w) + b)
    
    return X, Y
  
class Perceptron:
  def fit(self, X, Y, learning_rate=1.0, epochs=1000):
    D=X.shape[1]
    self.w = np.random.randn(D)
    self.b = 0
    
    N = len(Y)
    costs = []
    
    for epoch in range(epochs):
      Yhat = self.predict(X)
      wrong = np.nonzero(Y != Yhat)[0]
      if len(wrong) == 0:
        break
      
      i = np.random.choice(wrong)
      self.w += learning_rate*Y[i]*X[i]
      self.b += learning_rate*Y[i]
      
      c = len(wrong)/N
      costs.append(c)
    print("w:",self.w,"\nb:",self.b,"\nepochs:",(epoch + 1),"/",epochs)
    plt.plot(costs)
    plt.show()
  
  def predict(self, X):
    return np.sign(X.dot(self.w) + self.b)
  
  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(P == Y) 
      
if __name__=="__main__":
  # Using the perceptron to classify our hand-cooked linear data
  X, Y = linear_data()
  plt.scatter(X[:,0], X[:, 1], c=Y, s=100, alpha=.5)
  plt.show()
  
  Ntrain = int(len(Y)*.5)
  print("Ntrain =", Ntrain)
  Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
  Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

  model = Perceptron()
  t0 = time()
  model.fit(Xtrain, Ytrain)
  print("Training time:", time() - t0)

  t0 = time()
  print("Train accuracy:", model.score(Xtrain, Ytrain))
  print("Time to calculate train accuracy:", time() - t0)

  t0 = time()
  print("Test accuracy:", model.score(Xtest, Ytest))
  print("Time to calculate Test accuracy:", time() - t0)

  
  #Using the perceptron to distinguish zeros and ones in the mnist dataset
  X, Y = get_data()
  idx = np.logical_or(Y==0,Y==1)
  X = X[idx]
  Y = Y[idx]
  Y[Y==0] = -1
  model = Perceptron()
  t0 = time()
  model.fit(X, Y, learning_rate=0.001)
  print("MNIST 0s and 1s train accuracy:", model.score(X, Y))

  #Using the perceptron to distinguish zeros and ones in the XOR dataset
  X, Y = get_xor()
  model = Perceptron()
  t0 = time()
  model.fit(X, Y)
  print("XOR train accuracy:", model.score(X, Y))