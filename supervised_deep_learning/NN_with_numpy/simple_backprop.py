# here we do backprop in a simple NN with single hidden layer
import numpy as np
import matplotlib.pyplot as plt

sigmoid = lambda X, W, b: 1/(1+np.exp(-X.dot(W) - b))

def forward(X, W1, b1, W2, b2):
  Z = sigmoid(X,W1,b1)
  A = Z.dot(W2) + b2
  expA = np.exp(A)
  Y = expA/(expA.sum(axis=1,keepdims=True))
  return Y, Z

cost = lambda Targets,Y: (Targets*np.log(Y)).sum()

deriv_W2 = lambda Z, Targets, Y: Z.T.dot(Targets-Y)
deriv_W1 = lambda X, Z, Targets, Y, W2: X.T.dot((Targets-Y).dot(W2.T)*Z*(1-Z))

deriv_b2 = lambda Targets, Y: (Targets-Y).sum(axis=0)
deriv_b1 = lambda Targets, Y, W2, Z: ((Targets-Y).dot(W2.T)*Z*(1-Z)).sum(axis=0)

def classif_rate(Y,P):
  correct = 0
  total = 0
  len_Y = len(Y)
  for i in range(len_Y):
    total += 1.0
    if Y[i]==P[i]: 
      correct+=1.0
  return correct/total

def main():

  # Let us generate some mock data
  Nclass = 500
  D = 2
  M = 3
  K = 3

  X1 = np.random.randn(Nclass, D) + np.array([2,-2])
  X2 = np.random.randn(Nclass, D) + np.array([-2,2])
  X3 = np.random.randn(Nclass, D) + np.array([2,2])
  # X1 = np.random.randn(Nclass, D) + np.array([0, -2])
  # X2 = np.random.randn(Nclass, D) + np.array([2, 2])
  # X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
  X = np.vstack([X1, X2, X3])
  Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

  # peeking at the data
  plt.scatter(X[:,0], X[:,1], c=Y)
  plt.show()
  N = len(Y)

  Targets = np.zeros((N,K))
  for i in range(N):
    Targets[i,Y[i]] = 1 # one-hot encoding the targets

  # random initial weights
  W1 = np.random.randn(D,M)
  b1 = np.random.randn(M)
  W2 = np.random.randn(M,K)
  b2 = np.random.randn(K)

  eta = 0.00001 # learning rate
  costs=[]
  epochs = 100000
  for epoch in range(epochs):
    out, hidden = forward(X,W1,b1,W2,b2)
    if epoch % 100 == 0:
      c = cost(Targets, out)
      preds = np.argmax(out, axis=1)
      r = classif_rate(Y,preds)
      print("cost:",c,"\n classification rate:",r)
      costs.append(c)
    # gradient ascent
    W2 += eta*deriv_W2(hidden, Targets, out)
    b2 += eta*deriv_b2(Targets, out)
    W1 += eta*deriv_W1(X, hidden, Targets, out, W2)
    b1 += eta*deriv_b1(Targets, out, W2, hidden)
  plt.plot(costs)
  plt.show()
    
if __name__ == "__main__":
  main()