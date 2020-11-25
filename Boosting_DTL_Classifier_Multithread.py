from DTL_Classifier import *
import itertools
import statsmodels.api as sm
from pathos.multiprocessing import Pool
from contextlib import closing

class Boosting_DTL_Classifier_Multithread:
    def __init__(self):
        self.n_estimators = 1
        self.min_dim = 2
        self.max_dim = 2
        self.list_dtl = None
        self.list_subspaces = None
        self.list_dtl_subspace = None
        self.n_bootstrap = 0.8
        self.learning_rate = 0.1
        self.n_thread = 20

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
        for dim_subspace in range(self.min_dim, self.max_dim + 1):
            list_dims.extend(list(itertools.combinations(range(np.size(X, axis=1)),
                                                                               dim_subspace)))
        for k in range(self.n_estimators):
            print(k)
            Xb, Yb, Xoob, Yoob = Bootstrap(X, Y)  # bootstrap data
            if k >= 1:

                def predict_b_oob((dtl, subspace)):

                    return dtl.predict(Xb[:, subspace]), \
                           dtl.predict(Xoob[:, subspace])


                with closing(Pool(self.n_thread)) as pool:
                    list_predicts_b_oob = np.array(pool.map(predict_b_oob, [(self.List_dtl[j], self.List_subspace[j]) for j in reversed(range(len(self.List_dtl)))]))[::-1]

                Matrix_predicts_b = np.array(list_predicts_b_oob[:, 0])
                Matrix_predicts_oob = np.array(list_predicts_b_oob[:, 1])

                Rb = Yb - self.learning_rate*np.dot(self.List_parameters, Matrix_predicts_b)
                Roob = Yoob - self.learning_rate*np.dot(self.List_parameters, Matrix_predicts_oob)
            else:
                Rb = Yb
                Roob = Yoob

            print np.var(Rb), np.var(Roob)

            # iterate all possible subspaces and greedy find the optimal one.

            def evaluate_mcr(dims):
                #print dims
                dtl = DTL_Classifier()
                dtl.fit(Xb[:, dims], Rb)
                mcr = dtl.mcr(Xoob[:, dims], Roob)
                return mcr

            def fit_dtl(dims):
                dtl = DTL_Classifier()
                dtl.fit(Xb[:, dims], Rb)
                return dtl

            with closing(Pool(self.n_thread)) as pool:
                list_dtls_mcr = pool.map(evaluate_mcr, list_dims[::-1])[::-1] # heavy load work do first

            opt_dtl_idx = np.argmin(list_dtls_mcr)
            opt_dtl = fit_dtl(list_dims[opt_dtl_idx])
            opt_subspace = list_dims[opt_dtl_idx]


            # optimal coefficient

            model = sm.OLS(Rb, opt_dtl.predict(Xb[:, opt_subspace]))
            results = model.fit()
            theta = results.params[0]

            self.List_subspace.append(opt_subspace)
            self.List_dtl.append(opt_dtl)
            self.List_parameters.append(theta)

        return self.List_dtl, self.List_subspace

    def predict(self, X_predict):

        def dtl_predict(j):
            return self.List_dtl[j].predict(X_predict[:, self.List_subspace[j]])

        with closing(Pool(self.n_thread)) as pool:
        #pool = Pool(self.n_thread)
            Matrix_predicts = np.array(pool.map(dtl_predict, range(self.n_estimators)))

        return self.learning_rate*np.dot(self.List_parameters, Matrix_predicts)>0

    def predict_prob(self, X_predict):

        def dtl_predict(j):
            return self.List_dtl[j].predict_prob(X_predict[:, self.List_subspace[j]])

        with closing(Pool(self.n_thread)) as pool:
        #pool = Pool(self.n_thread)
            Matrix_predicts = np.array(pool.map(dtl_predict, range(self.n_estimators)))

        return self.learning_rate*np.dot(self.List_parameters, Matrix_predicts)

    def mcr(self, X, Y):
        Y_predict = self.predict(X)
        return np.average(Y_predict!=Y)



if __name__ == '__main__':
    from DataGenerator import *

    n_train = 10000  # sample size
    n_test = 10000
    p = 2  # dimension of features

    X_train, Y_train = data_generator(f, n_train, p)  # generate training data from f(X)
    X_test, Y_test = data_generator(f, n_test, p)  # generate testing data from f(X)

    bdtl = Boosting_DTL_Classifier_Multithread()
    bdtl.learning_rate = 1
    bdtl.n_estimators = 1
    bdtl.max_dim = 2
    bdtl.fit(X_train, Y_train)

    print bdtl.mcr(X_train, Y_train)
    print bdtl.mcr(X_test, Y_test)
