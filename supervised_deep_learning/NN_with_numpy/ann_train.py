import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.utils import shuffle

# here you can write a function to load and return the splitted data (train-test); I have applied it to the fer2013 dataset, but help yourself (;
def get_data():
  Y = []
  X = []
  header = True
  for line in open('./fer2013/fer2013.csv'):
    if header: header=False
    else:
      row = line.split(',')
      Y.append(int(row[0]))
      X.append([int(p) for p in row[1].split()])
  X, Y = np.array(X)/255, np.array(Y)
  # if you explore the data, you'll see that class 1 is severely imbalanced (only 547 samples, whereas the other five classes have typically 5k samples); we deal with such an imbalance by repeating the data...
  X_rest, Y_rest = X[Y!=1,:], Y[Y!=1]
  X1 = X[Y==1, :]
  X1 = np.repeat(X1, 9, axis=0)
  X = np.vstack([X_rest,X1])
  Y = np.concatenate((Y_rest, [1]*len(X1)))
  
  X, Y = shuffle(X,Y)
  Ntrain = int(0.8*len(X))
  Xtrain, Ytrain = X[:Ntrain, :], Y[:Ntrain]
  Xtest, Ytest = X[Ntrain:, :], Y[Ntrain:]
  
  return Xtrain, Ytrain, Xtest, Ytest

def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1 #one-hot encoding the targets/indicators
    return ind

Xtrain, Ytrain, Xtest, Ytest = get_data()
D = Xtrain.shape[1]
K = len(set(Ytrain) | set(Ytest))
M = 5 # num hidden units (hyperparam!)

# convert to indicator
Ytrain_ind = y2indicator(Ytrain, K)
Ytest_ind = y2indicator(Ytest, K)

# randomly initialize weights
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

# the prediction-related functions
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z

predict = lambda P_Y_given_X: np.argmax(P_Y_given_X, axis=1)
classif_rate = lambda Y, P: np.mean(Y == P)
cost = lambda T, pY: -np.mean(T*np.log(pY)) # the cross-entropy

# train loop
train_costs = []
test_costs = []
eta = 0.001 # the learning rate
for i in range(10000):
    pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
    pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)

    ctrain = cost(Ytrain_ind, pYtrain)
    ctest = cost(Ytest_ind, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # gradient descent
    pYminusT = pYtrain - Ytrain_ind # train predictions minus training targets 
    W2 -= eta*Ztrain.T.dot(pYminusT)
    b2 -= eta*(pYminusT).sum(axis=0)
    dZ = (pYminusT).dot(W2.T) * (1 - Ztrain*Ztrain)
    W1 -= eta*Xtrain.T.dot(dZ)
    b1 -= eta*dZ.sum(axis=0)

    print("i:",i, " - train cost:",ctrain," - test cost:", ctest)
    
# Of course, in the real world we need good "convergence criteria"... after which we should assign the final weights and biases (W2, b2, W1 and b1) to our model (the "fitted model"); that is easily done via a class implementation of our ANN...

print("Final train classif_rate:", classif_rate(Ytrain, predict(pYtrain)))
print("Final test classif_rate:", classif_rate(Ytest, predict(pYtest)))

plt.plot(train_costs, label='train cost')
plt.plot(test_costs, label='test cost')
plt.legend()
plt.show()