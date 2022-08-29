# here we build a 'quasi' random-forest, in the sense that sampling is done on data columns/features only once for a given model, not on features at every node (which's what a true random forest does); we benchmark our implementation against a true RF (scikitlearn's one) and against our Bagged_DT_Classifier (simple bootstrap with Decision Trees; see "classification_trees.py") with the Mushroom Classification dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from classification_trees import Bagged_DT_Classifier

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

# We'll use the poisonous mushroom dataset
catCols = np.arange(22) + 1
numCols = () # the dataset has no numerical columns
class DataTransformer:
  def fit(self, df):
    self.labelEncoders = {}
    self.scalers = {}
    for col in numCols:
      scaler = StandardScaler()
      scaler.fit(df[col].reshape(-1,1))
      self.scalers[col] = scaler
    
    for col in catCols:
      encoder = LabelEncoder()
      values = df[col].tolist()
      values.append('missing')
      encoder.fit(values)
      self.labelEncoders[col] = encoder
      
    self.D = len(numCols)
    for col, encoder in self.labelEncoders.items():
      self.D += len(encoder.classes_)
    print("dimensionality =", self.D)

  
  def transform(self,df):
    N, _ = df.shape
    X = np.zeros((N, self.D))
    i = 0
    for col, scaler in self.scalers.items():
      X[:,i] = scaler.transform(df[col].as_matrix().reshape(-1,1)).flatten()
      i += 1
      
    for col, encoder in self.labelEncoders.items():
      k = len(encoder.classes_)
      X[np.arange(N), encoder.transform(df[col]) + i] = 1
      i += k
    return X
  
  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)
  
def replace_missing(df):
  # standard method of replacement for numerical columns is median
  for col in numCols:
    if np.any(df[col].isnull()):
      med = np.median(df[ col ][ df[col].notnull() ])
      df.loc[ df[col].isnull(), col ] = med

  # set a special value = 'missing'
  for col in catCols:
    if np.any(df[col].isnull()):
      print(col)
      df.loc[ df[col].isnull(), col ] = 'missing'

def get_data():
  df = pd.read_csv('../data/poisonous-mushrooms/agaricus-lepiota.data', header=None)

  # replace label column: e/p --> 0/1
  # e = edible = 0, p = poisonous = 1
  df[0] = df.apply(lambda row: 0 if row[0] == 'e' else 1, axis=1)

  # check if there is missing data
  replace_missing(df)

  # transform the data
  transformer = DataTransformer()

  X = transformer.fit_transform(df)
  Y = df[0].values
  return X, Y

X, Y = get_data()
Ntrain = int(0.8*len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

class QuasiRandomForest:
  def __init__(self, n_estimators):
    self.B = n_estimators

  def fit(self, X, Y, M=None):
    N, D = X.shape
    if M is None:
      M = int(np.sqrt(D))

    self.models = []
    self.features = []
    for b in range(self.B):
      tree = DecisionTreeClassifier()

      # sample features (WITHOUT replacement!)
      features = np.random.choice(D, size=M, replace=False)

      # sample training samples (WITH replacement, for we're bootstraping here)
      idx = np.random.choice(N, size=N, replace=True)
      Xb = X[idx]
      Yb = Y[idx]

      tree.fit(Xb[:, features], Yb)
      self.features.append(features)
      self.models.append(tree)

  def predict(self, X):
    N = len(X)
    P = np.zeros(N)
    for features, tree in zip(self.features, self.models):
      P += tree.predict(X[:, features])
    return np.round(P / self.B)

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(P == Y)


T = 1000
test_accuracy_qrf = np.empty(T)
test_accuracy_rf = np.empty(T)
test_accuracy_bag = np.empty(T)
for trees in range(T):
  if trees == 0:
    test_accuracy_qrf[trees] = None
    test_accuracy_rf[trees] = None
    test_accuracy_bag[trees] = None
  else:
    rf = RandomForestClassifier(n_estimators=trees)
    rf.fit(Xtrain, Ytrain)
    test_accuracy_rf[trees] = rf.score(Xtest, Ytest)

    bag = Bagged_DT_Classifier(trees)
    bag.fit(Xtrain, Ytrain)
    test_accuracy_bag[trees] = bag.score(Xtest, Ytest)

    qrf = QuasiRandomForest(n_estimators=trees)
    qrf.fit(Xtrain, Ytrain)
    test_accuracy_qrf[trees] = qrf.score(Xtest, Ytest)

  if trees % 10 == 0:
    print("trees:", trees)

plt.plot(test_accuracy_rf, label='rf')
plt.plot(test_accuracy_qrf, label='quasi rf')
plt.plot(test_accuracy_bag, label='bag')
plt.legend()
plt.show()