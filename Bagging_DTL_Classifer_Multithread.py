from DTL_Classifier import *
import itertools
from pathos.multiprocessing import Pool
from contextlib import closing

class Bagging_DTL_Classifier_Multithread():
    def __init__(self, n_estimator=None, max_depth=None):
        self.n_estimators = 10
        self.max_depth = None
        self.list_dtl = None
        self.list_subspaces = None
        self.list_dtl_subspace = None
        self.n_bootstrap = 0.8
        self.n_thread = 4

    def fit(self, X, Y):
        d = np.size(X, axis=1)
        n = np.size(X, axis=0)
        if self.max_depth is None:
            self.max_depth = d

        self.list_subspaces = [j for k in range(1, int(self.max_depth)+1) for j in itertools.combinations(range(d), k)]

        list_idx_bootstrap = [np.random.choice(range(n), size=int(self.n_bootstrap*n), replace=False) for i in range(self.n_estimators)]

        list_Xb = [X[list_idx_bootstrap[i], :] for i in range(self.n_estimators)]
        list_Yb = [Y[list_idx_bootstrap[i]] for i in range(self.n_estimators)]
        list_Xoob = [X[list(set(range(n))-set(list_idx_bootstrap[i])), :] for i in range(self.n_estimators)]
        list_Yoob = [Y[list(set(range(n))-set(list_idx_bootstrap[i]))] for i in range(self.n_estimators)]

        # fit subspaces and evaluate OOB error
        def mse_evaluate(Xb, Yb, Xoob, Yoob, subspace):
            dtl = DTL_Classifier()
            dtl.fit(Xb[:, subspace], Yb)
            mse = np.var(Yoob - dtl.predict(Xoob[:, subspace]))
            return mse

        def dtl_fit(Xb, Yb, subspace):
            dtl = DTL_Classifier()
            dtl.fit(Xb[:, subspace], Yb)
            return dtl

        self.list_dtl = []
        self.list_dtl_subspace = []

        def fit_base_dtl(i):
            print i
            Xb = list_Xb[i]
            Yb = list_Yb[i]
            Xoob = list_Xoob[i]
            Yoob = list_Yoob[i]

            list_mse = np.zeros(len(self.list_subspaces))
            for j in range(len(self.list_subspaces)):
                mse = mse_evaluate(Xb, Yb, Xoob, Yoob, self.list_subspaces[j])
                list_mse[j] = mse

            idx_opt_dtl = np.argmin(list_mse)
            subspace_opt = self.list_subspaces[idx_opt_dtl]
            opt_dtl = dtl_fit(Xb, Yb, subspace_opt)

            return opt_dtl, subspace_opt

        with closing(Pool(self.n_thread)) as pool:
            array_opt_dtl_subspace = np.array(pool.map(fit_base_dtl, range(self.n_estimators)))
        self.list_dtl = array_opt_dtl_subspace[:, 0]
        self.list_dtl_subspace = array_opt_dtl_subspace[:, 1]


    def predict(self, X):
        def dtl_predict(i):
            dtl = self.list_dtl[i]
            return dtl.predict(X[:, self.list_dtl_subspace[i]])


        with closing(Pool(self.n_thread)) as pool:
            matrix_predict = pool.map(dtl_predict, range(self.n_estimators))

        return np.average(matrix_predict, axis=0)

    def predict_proba(self, X):
        def dtl_predict(i):
            dtl = self.list_dtl[i]
            return dtl.predict_proba(X[:, self.list_dtl_subspace[i]])


        with closing(Pool(self.n_thread)) as pool:
            matrix_predict = pool.map(dtl_predict, range(self.n_estimators))

        return np.average(matrix_predict, axis=0)

if __name__ == '__main__':
    n = 10000
    p = 10
    X_train = np.random.uniform(size=[n, p])
    Y_train = np.sum(X_train ** 2, axis=1)
    X_test = np.random.normal(size=[n, p])
    Y_test = np.sum(X_test ** 2, axis=1)

    bdtl = Bagging_DTL_Classifier_Multithread()
    bdtl.max_depth = 2
    bdtl.fit(X_train, Y_train)

    print bdtl.predict(X_train)
    #print Y_train
