import numpy as np
from collections import Counter

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X, k=1):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      # distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1)) # Using L2-Norm instead
      # Get the indices of the k smallest distances
      k_indices = np.argsort(distances)[:k]
      k_nearest_labels = self.ytr[k_indices]

      # Majority vote
      most_common = Counter(k_nearest_labels).most_common(1)
      Ypred[i] = most_common[0][0]

    return Ypred
