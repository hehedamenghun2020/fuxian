import numpy as np
import scipy.spatial
from sklearn import preprocessing

"******************************* CENTROID *************************************"
class CentroidBasedOneClassClassifier:
    def __init__(self, threshold = 0.5, metric="euclidean", scale = "standard"):
        
        self.threshold = threshold
        """only CEN used StandardScaler because the centroid of training set need
        to be move to origin"""
        self.scaler = preprocessing.StandardScaler()                
        self.metric = metric

    def fit(self, X):
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        # because we are using StandardScaler, the centroid is a
        # vector of zeros, but we save it in shape (1, n) to allow
        # cdist to work happily later.
        self.centroid = np.zeros((1, X.shape[1]))
        # no need to scale again
        dists = self.get_density(X, scale=False) 
        # transform relative threshold (eg 95%) to absolute
        self.abs_threshold = np.percentile(dists, 100 * self.threshold)
        
        
    #It is actually the mean of the distances from each points in training set
    #to the centroid zero.
    def get_density(self, X, scale=True):
        if scale:
            X = self.scaler.transform(X)
        dists = scipy.spatial.distance.cdist(X, self.centroid, metric=self.metric)
        dists = np.mean(dists, axis=1)
        return dists
   
    def predict(self, X):
        dists = self.get_density(X)
        return dists > self.abs_threshold