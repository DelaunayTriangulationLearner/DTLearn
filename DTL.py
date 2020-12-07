from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from scipy.spatial import ConvexHull
import numpy as np


class DTL():
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

            return np.nan_to_num(dtl_predict*(1-dtl_isnan)) + dtl_isnan*nnl_predict


    def mse(self, X, Y):
        try:
            return np.var(self.predict(X) - Y)
        except:
            return np.inf

if __name__ == '__main__':
    from DataGenerator import *
    from sklearn.ensemble import GradientBoostingRegressor
    n_train = 10000
    n_test = 10000
    p = 2

    X_train, Y_train = data_generator(f, n_train, p)  # generate training data from f(X)
    X_test, Y_test = data_generator(f, n_test, p)  # generate testing data from f(X)

    bdtlr = DTL()
    bdtlr.n_thread = 24

    bdtlr.n_estimators = 1
    bdtlr.n_bootstrap = 0.8
    bdtlr.max_dim = 2
    bdtlr.learning_rate = 1
    bdtlr.fit(X_train, Y_train)

    print (bdtlr.mse(X_train, Y_train))
    print (bdtlr.mse(X_test, Y_test))


    gbt = GradientBoostingRegressor()
    gbt.fit(X_train, Y_train)
    print (np.var(Y_train - gbt.predict(X_train)))
    print (np.var(Y_test - gbt.predict(X_test)))


