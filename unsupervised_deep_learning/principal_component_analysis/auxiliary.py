import numpy as np
import pandas as pd

from sklearn.utils import shuffle

def relu(x):
  return x*(x>0)

def error_rate(p,t):
  return np.mean(p != t)

def get_mnist():
  df = pd.read_csv("../data/mnist-train/train.csv").to_numpy()
  df = shuffle(df)
  # print(df.head())
  Xtrain, Ytrain = df[:-1000,1:]/255, df[:-1000,0]
  Xtest, Ytest = df[-1000:,1:]/255, df[-1000:,0]
  
  return Xtrain, Ytrain, Xtest, Ytest

def init_weights(shape):
  return np.random.randn(*shape) / np.sqrt(sum(shape))