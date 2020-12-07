from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from scipy.spatial import ConvexHull
import numpy as np


class DTL_Classifier():
    def __init__(self):
        self.dtl = None
        self.nnl = None
        self.hull = None
        self.max1d = None
        self.min1d = None


    def fit(self, X, Y):
        dim = np.size(X, axis=1)
        if dim == 1:
            self.dtl = interp1d(X[:, 0], Y, 'linear')
            self.max1d = np.max(X[:, 0])
            self.min1d = np.min(X[:, 0])

        else:

            self.dtl = LinearNDInterpolator(X, Y)
            self.hull = ConvexHull(X)
            X_vertices = X[self.hull.vertices, :]
            Y_vertices = Y[self.hull.vertices]
            self.nnl = NearestNDInterpolator(X_vertices, Y_vertices)


    def predict_prob(self, X):
        dim = np.size(X, axis=1)
        if dim == 1:
            idx_smaller = X[:, 0] < self.min1d
            idx_larger = X[:, 0] > self.max1d
            X[idx_smaller, 0] = self.min1d
            X[idx_larger, 0] = self.max1d
            dtl_predict = self.dtl(X[:, 0])
            return dtl_predict

        else:
            dtl_predict = self.dtl.__call__(X)
            dtl_isnan = np.isnan(dtl_predict)
            nnl_predict = self.nnl.__call__(X)

            return np.nan_to_num(dtl_predict*(1-dtl_isnan)) + dtl_isnan*nnl_predict


    def predict(self, X):
        dim = np.size(X, axis=1)
        if dim == 1:
            idx_smaller = X[:, 0] < self.min1d
            idx_larger = X[:, 0] > self.max1d
            X[idx_smaller, 0] = self.min1d
            X[idx_larger, 0] = self.max1d
            dtl_predict = self.dtl(X[:, 0])
            return dtl_predict

        else:
            dtl_predict = self.dtl.__call__(X)
            dtl_isnan = np.isnan(dtl_predict)
            nnl_predict = self.nnl.__call__(X)

            return (np.nan_to_num(dtl_predict * (1 - dtl_isnan)) + dtl_isnan * nnl_predict) > 0.5


    def mcr(self, X, Y):
        return np.average(self.predict(X) != Y)


if __name__ == '__main__':
    from DataGenerator import *

    X1 = np.random.uniform(size=[20000, 1])
    X2 = np.random.uniform(size=[20000, 1])
    X = np.concatenate([X1, X2], axis=1)
    Y = (X1 > 0.5).astype('int')

    X_train = X[:10000, :]
    X_test = X[10000:, :]
    Y_train = Y[:10000]
    Y_test = Y[10000:]

    n_train = 10000  # sample size
    n_test = 10000

    dtl = DTL_Classifier()
    dtl.fit(X_train, Y_train)
    print (dtl.predict(X_test))

