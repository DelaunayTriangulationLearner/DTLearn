from F_D_Lambda import *
from scipy.optimize import minimize
import itertools

class DBoosting:
    def __init__(self, n_estimator=None, dim_subspace=None, learning_rate=None, n_bootstrap=None,
                 Lambda=None, alpha=None, eps=None, h=None, List_f_d_lambda=None, gamma_list=None, list_base_learner=None,
                 List_subspace=None, initial=None):
        self.n_estimator = n_estimator
        self.dim_subspace = dim_subspace
        self.learning_rate = learning_rate
        self.n_bootstrap = n_bootstrap
        self.List_f_d_lambda = List_f_d_lambda
        self.Lambda = Lambda
        self.alpha = alpha
        self.eps = eps
        self.h = h
        self.gamma_list = gamma_list
        self.list_base_learner = list_base_learner
        self.List_subspace = List_subspace
        self.initial = initial

        if self.gamma_list is None:
            self.gamma_list = np.zeros(self.n_estimator)

        if self.initial is None:
            self.initial = 'Y'

        if self.List_f_d_lambda is None:
            self.List_f_d_lambda = []

        if self.n_estimator is None:
            self.n_estimator = 100

        if self.dim_subspace is None:
            self.dim_subspace = 2

        if self.learning_rate is None:
            self.learning_rate = 0.1


        if self.list_base_learner is None:
            self.list_base_learner = []

    def fit(self, X_train, Y_train):
        n = len(Y_train)
        if self.n_bootstrap is None:
            self.n_bootstrap = n

        if self.gamma_list is None:
            self.gamma_list = np.zeros(self.n_estimator)

        self.List_subspace = np.zeros([self.n_estimator, self.dim_subspace])

        def Bootstrap(X, Y):
            sample_size = len(X)
            bootstrap_idx = np.random.choice(range(sample_size), size=self.n_bootstrap, replace=False)
            Xb = X[bootstrap_idx, :]
            Yb = Y[bootstrap_idx]
            return Xb, Yb

        def Residual(gamma):
            return np.var(Yb - gamma * estimate_new_base - self.learning_rate*np.dot(list_estimate_X_train, self.gamma_list))

        list_estimate_X_train = np.zeros([self.n_bootstrap, self.n_estimator])
        k = 0
        while k < self.n_estimator:
            Xb, Yb = Bootstrap(X_train, Y_train)
            Xoob, Yoob = Bootstrap(X_train, Y_train)  # rebootstrap data
            for dims in itertools.combinations(range(np.size(X_train, axis=1)), self.dim_subspace):
                max_r2 = -np.inf  # training r square
                opt_f_d_lambda = F_D_Lambda(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps, h=self.h)
                try:
                    f_d_lambda = F_D_Lambda(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps, h=self.h)
                    f_d_lambda.quick_fit(Xb[:, dims], Yb - self.learning_rate*np.dot(list_estimate_X_train, self.gamma_list))
                    r2 = f_d_lambda.score(Xoob[:, dims], Yoob)
                    # filter the optimal subspace and base learner
                    if r2 > max_r2:
                        max_r2 = r2
                        opt_f_d_lambda = f_d_lambda
                        self.List_subspace[k, :] = list(dims)

                except:
                    print 'Collinearity'
            self.List_f_d_lambda.append(opt_f_d_lambda)
            estimate_new_base = opt_f_d_lambda.predict(Xb[:, self.List_subspace[k, :].astype(int)])
            list_estimate_X_train[:, k] = estimate_new_base
            self.list_base_learner.append(opt_f_d_lambda)

            gamma_opt = minimize(fun=Residual, x0=np.ones([1])).x[0]
            self.gamma_list[k] = gamma_opt
            k += 1
            print k


    def predict(self, X_predict):
        List_Y_predict = np.zeros([self.n_estimator, len(X_predict)])
        for k in range(self.n_estimator):
            subspace = self.List_subspace[k, :]
            f_d_lambda = self.List_f_d_lambda[k]
            Y_predict = f_d_lambda.predict(X_predict[:, subspace.astype(int)])
            List_Y_predict[k, :] = Y_predict
        return self.learning_rate*np.dot(List_Y_predict.transpose(), self.gamma_list)

    def simulation_predict(self, X_predict):
        List_Y_predict = np.zeros([self.n_estimator, len(X_predict)])
        for k in range(self.n_estimator):
            subspace = self.List_subspace[k, :]
            f_d_lambda = self.List_f_d_lambda[k]
            Y_predict = f_d_lambda.predict(X_predict[:, subspace.astype(int)])
            List_Y_predict[k, :] = Y_predict
        return List_Y_predict.transpose(), self.gamma_list

    def score(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        R2 = 1 - np.var(Y_predict-Y_test)/np.var(Y_test)
        return R2

if __name__ == '__main__':
    from DataGenerator import *
    n_train = 100  # sample size
    n_test = 20
    eps = 0.01  # precision
    alpha = 2  # if the lambda is large, smaller alpha should be used.

    p = 5  # dimension of features
    X_train, Y_train = data_generator(f, n_train, p)  # generate training data from f(X)
    X_test, Y_test = data_generator(f, n_test, p)

    db = DBoosting(n_estimator=20)
    db.fit(X_train, Y_train)

    print db.predict(X_test)