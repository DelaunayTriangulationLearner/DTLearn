from DTL_Classifier import *
import itertools
import statsmodels.api as sm

class Boosting_DTL_Classiffier:
    def __init__(self):
        self.n_estimators = 10
        self.max_dim = 2
        self.list_dtl = None
        self.list_subspaces = None
        self.list_dtl_subspace = None
        self.n_bootstrap = 0.8
        self.learning_rate = 1

        self.List_dtl = []
        self.List_is_in = []


    def fit(self, X, Y):

        self.List_subspace = []
        n = len(Y)

        def Bootstrap(X, Y):
            sample_size = len(X)
            bootstrap_idx = np.random.choice(range(sample_size), size=int(n*self.n_bootstrap), replace=False)
            oob_idx = list(set(range(len(Y)))-set(bootstrap_idx))

            Xb = X[bootstrap_idx, :]
            Yb = Y[bootstrap_idx]
            Xoob = X[oob_idx, :]
            Yoob = Y[oob_idx]
            return Xb, Yb, Xoob, Yoob

        self.List_dtl = []
        self.List_parameters = []

        list_dims = []
        for dim_subspace in range(1, self.max_dim + 1):
            list_dims.extend(list(itertools.combinations(range(np.size(X, axis=1)), dim_subspace)))
        for k in range(self.n_estimators):
            print(k)
            Xb, Yb, Xoob, Yoob = Bootstrap(X, Y)  # bootstrap data
            if k >= 1:
                Matrix_predicts_b = np.zeros([len(self.List_parameters), len(Yb)]) # current predicts by boosted model
                for j in range(len(self.List_parameters)):
                    Matrix_predicts_b[j, :] = self.List_dtl[j].predict(Xb[:, self.List_subspace[j]])
                Matrix_predicts_oob = np.zeros([len(self.List_parameters), len(Yoob)])  # current predicts by boosted model
                for j in range(len(self.List_parameters)):
                    Matrix_predicts_oob[j, :] = self.List_dtl[j].predict(Xoob[:, self.List_subspace[j]])

                Rb = Yb - self.learning_rate*np.dot(self.List_parameters, Matrix_predicts_b)
                Roob = Yoob - self.learning_rate*np.dot(self.List_parameters, Matrix_predicts_oob)
            else:
                Rb = Yb
                Roob = Yoob

            # iterate all possible subspaces and greedy find the optimal one.
            min_mcr = np.inf  # training r square
            opt_dtl = DTL_Classifier()

            for dims in list_dims:
                dtl = DTL_Classifier()
                dtl.fit(Xb[:, dims], Rb)
                mcr = dtl.mcr(Xoob[:, dims], Roob)

                # filter the optimal subspace and base learner
                if mcr < min_mcr:
                    min_mcr = mcr
                    opt_dtl = dtl
                    opt_subspace = list(dims)

            # optimal coefficient

            model = sm.OLS(Rb, opt_dtl.predict(Xb[:, opt_subspace]))
            results = model.fit()
            theta = results.params[0]

            self.List_subspace.append(opt_subspace)
            self.List_dtl.append(opt_dtl)
            self.List_parameters.append(theta)

        return self.List_dtl, self.List_subspace

    def predict(self, X_predict):
        Matrix_predicts = np.zeros([len(self.List_parameters), np.size(X_predict, axis=0)])
        for j in range(self.n_estimators):
            Matrix_predicts[j, :] = self.List_dtl[j].predict(X_predict[:, self.List_subspace[j]])

        return self.learning_rate*np.dot(self.List_parameters, Matrix_predicts)

    def mcr(self, X, Y):
        Y_predict = self.predict(X)
        return np.average(Y_predict != Y)



if __name__ == '__main__':
    from DataGenerator import *

    n_train = 100  # sample size
    n_test = 100

    n = 100
    X1 = np.random.uniform(size=[20000, 1])
    X2 = np.random.uniform(size=[20000, 1])
    X = np.concatenate([X1, X2], axis=1)
    Y = (X[:, 0] > 0.5).astype(int)

    X_train = X[:10000, :]
    X_test = X[10000:, :]
    Y_train = Y[:10000]
    Y_test = Y[10000:]
    bdtl = Boosting_DTL_Classiffier()
    bdtl.learning_rate = 1
    bdtl.n_estimators = 10
    bdtl.max_dim = 2
    bdtl.fit(X_train, Y_train)

    print(bdtl.predict(X_test))





