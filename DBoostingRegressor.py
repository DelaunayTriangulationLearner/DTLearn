from F_D_Lambda import *
import itertools
from scipy.stats import linregress
import statsmodels.api as sm

class DBoostingRegression:
    def __init__(self, n_estimator=None, max_dim=None, n_bootstrap=None,
                 Lambda=None, learning_rate=None):

        self.n_estimator = n_estimator
        self.max_dim = max_dim
        self.n_bootstrap = n_bootstrap
        self.Lambda = Lambda
        self.learning_rate = learning_rate

        self.List_f_d_lambda = []
        self.List_is_in = []

        if self.n_estimator is None:
            self.n_estimator = 100

        if self.Lambda is None:
            self.Lambda = 0

        if self.learning_rate is None:
            self.learning_rate = 0.1

    def fit(self, X_train, Y_train):

        if self.max_dim is None:
            self.max_dim = np.size(X_train, axis=1)

        self.List_subspace = []

        n = len(Y_train)
        if self.n_bootstrap is None:
            self.n_bootstrap = 0.9

        def Bootstrap(X, Y):
            sample_size = len(X)
            bootstrap_idx = np.random.choice(range(sample_size), size=int(n*self.n_bootstrap), replace=False)
            oob_idx = list(set(range(len(Y)))-set(bootstrap_idx))

            Xb = X[bootstrap_idx, :]
            Yb = Y[bootstrap_idx]
            Xoob = X[oob_idx, :]
            Yoob = Y[oob_idx]
            return Xb, Yb, Xoob, Yoob

        self.List_f_d_lambda = []
        self.List_parameters = []

        list_dims = []
        for dim_subspace in range(1, self.max_dim + 1):
            list_dims.extend(np.random.permutation(list(itertools.combinations(range(np.size(X_train, axis=1)),
                                                                               dim_subspace))))
        for k in range(self.n_estimator):
            print(k)
            Xb, Yb, Xoob, Yoob = Bootstrap(X_train, Y_train)  # bootstrap data
            if k >= 1:
                Matrix_predicts_b = np.zeros([len(self.List_parameters), len(Yb)]) # current predicts by boosted model
                for j in range(len(self.List_parameters)):
                    Matrix_predicts_b[j, :] = self.List_f_d_lambda[j].predict(Xb[:, self.List_subspace[j]])
                Matrix_predicts_oob = np.zeros([len(self.List_parameters), len(Yoob)])  # current predicts by boosted model
                for j in range(len(self.List_parameters)):
                    Matrix_predicts_oob[j, :] = self.List_f_d_lambda[j].predict(Xoob[:, self.List_subspace[j]])

                Rb = Yb - self.learning_rate*np.dot(self.List_parameters, Matrix_predicts_b)
                Roob = Yoob - self.learning_rate*np.dot(self.List_parameters, Matrix_predicts_oob)
            else:
                Rb = Yb
                Roob = Yoob

            # iterate all possible subspaces and greedy find the optimal one.
            min_mse = np.inf  # training r square
            opt_f_d_lambda = F_D_Lambda()

            for dims in list_dims:
                # d=1
                if len(dims) == 1:
                    try:
                        f_d_lambda = F_D_Lambda(Lambda=self.Lambda)

                        f_d_lambda.quick_fit1(Xb[:, dims[0]], Rb)
                        mse = f_d_lambda.mse(Xoob[:, dims], Roob)
                        if mse < min_mse:
                            min_mse = mse
                            opt_f_d_lambda = f_d_lambda
                            opt_subspace = list(dims)
                    except:
                        pass
                else:
                    # d>=2
                    try:

                        f_d_lambda = F_D_Lambda(Lambda=self.Lambda)
                        f_d_lambda.quick_fit(Xb[:, dims], Rb)
                        mse = f_d_lambda.mse(Xoob[:, dims], Roob)

                        # filter the optimal subspace and base learner
                        if mse < min_mse:
                            min_mse = mse
                            opt_f_d_lambda = f_d_lambda
                            opt_subspace = list(dims)
                    except:
                        #print 'Collinearity'
                        pass

            # optimal coefficient

            model = sm.OLS(Rb, opt_f_d_lambda.predict(Xb[:, opt_subspace]))
            results = model.fit()
            theta = results.params[0]

            self.List_subspace.append(opt_subspace)
            self.List_f_d_lambda.append(opt_f_d_lambda)
            self.List_parameters.append(theta)

        return self.List_f_d_lambda, self.List_subspace

    def predict(self, X_predict):
        Matrix_predicts = np.zeros([len(self.List_parameters), np.size(X_predict, axis=0)])
        for j in range(self.n_estimator):
            Matrix_predicts[j, :] = self.List_f_d_lambda[j].predict(X_predict[:, self.List_subspace[j]])

        return self.learning_rate*np.dot(self.List_parameters, Matrix_predicts)

    def mse(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        MSE = np.var(Y_predict-Y_test)
        return MSE


if __name__ == '__main__':
    from DataGenerator import *
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    n_train = 100  # sample size
    n_test = 100
    eps = 0.01  # precision
    alpha = 1  # if the lambda is large, smaller alpha should be used.

    p = 5  # dimension of features
    list_db = []
    list_gbt = []
    for k in range(100):
        X_train, Y_train = data_generator(f, n_train, p)  # generate training data from f(X)
        X_test, Y_test = data_generator(f, n_test, p)


        db = DBoostingRegression(n_estimator=30, n_bootstrap=0.9, max_dim=2)
        db.fit(X_train, Y_train)
        print (np.var(Y_test - db.predict(X_test)))
        list_db.append(np.var(Y_test - db.predict(X_test)))


        gbt = GradientBoostingRegressor(max_depth=2,n_estimators=100)
        gbt.fit(X_train, Y_train)
        print(np.var(Y_test - gbt.predict(X_test)))
        list_gbt.append(np.var(Y_test - gbt.predict(X_test)))

    print (np.average(list_db), np.average(list_gbt))