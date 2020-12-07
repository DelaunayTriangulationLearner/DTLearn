__author__ = 'CUPL'
import numpy as np


# true model
def f(X):

    P = 1./(1.+np.exp(-np.sum(X[:, :2], axis=1)))
    return np.array([np.random.binomial(1, m) for m in P])

# Generate training data
def data_generator(f, n, p):  #sample size = n and dimension = p
    X_train = np.random.normal(scale=1, size=[n, p])
    Y_train = f(X_train)
    return X_train, Y_train


#circle data
def make_circle(n, p):
    v = np.random.normal(size=[n, p])
    x = v/np.linalg.norm(v, ord=2)
    Y = np.random.binomial(n=1, size=n, p=0.5)
    r = (Y == 1) + 1
    X = np.zeros([n, p])
    for k in range(n):
        X[k, :] = x[k, :]*r[k]
    return X, Y

def make_gaussian(n, p):
    Y = np.random.binomial(n=1, size=n, p=0.5)
    X1 = (Y == 1) + np.random.normal(size=n, scale=0.5)
    X2 = (Y == 0) + np.random.normal(size=n, scale=0.5)
    X = np.random.normal(size=[n, p], scale=1)
    X[:, 0] = X1
    X[:, 1] = X2
    return X, Y



if __name__ == '__main__':
    from matplotlib import pyplot as plt

    n_train = 10  # sample size
    n_test = 10
    p = 5  # dimension of features

    X_train, Y_train = data_generator(f, n_train, p)  # generate training data from f(X)
    X_test, Y_test = data_generator(f, n_test, p)  # generate testing data from f(X)
    #print Y_train

    X, Y = make_circle(100, p)





