from DTL_Classifier import *
import itertools

class Bagging_DTL_Classifier():
    def __init__(self, n_estimators=None, max_depth=None):
        if n_estimators is None:
            self.n_estimators = 100
        else:
            self.n_estimators = n_estimators

        if max_depth is not None:
            self.max_depth = max_depth

        self.list_dtl = None
        self.list_subspaces = None
        self.list_dtl_subspace = None
        self.n_bootstrap = 0.8



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
        def mcr_evaluate(Xb, Yb, Xoob, Yoob, subspace):
            dtl = DTL_Classifier()
            Yb = Yb

            dtl.fit(Xb[:, subspace], Yb)
            mcr = np.average(Yoob != dtl.predict(Xoob[:, subspace]))
            return mcr

        def dtl_fit(Xb, Yb, subspace):
            dtl = DTL_Classifier()
            dtl.fit(Xb[:, subspace], Yb)
            return dtl

        self.list_dtl = []
        self.list_dtl_subspace = []

        for i in range(self.n_estimators):
            #print i
            Xb = list_Xb[i]
            Yb = list_Yb[i]
            Xoob = list_Xoob[i]
            Yoob = list_Yoob[i]

            list_mcr = np.zeros(len(self.list_subspaces))
            for j in range(len(self.list_subspaces)):
                mcr = mcr_evaluate(Xb, Yb, Xoob, Yoob, self.list_subspaces[j])
                list_mcr[j] = mcr

            idx_opt_dtl = np.argmin(list_mcr)
            subspace_opt = self.list_subspaces[idx_opt_dtl]
            opt_dtl = dtl_fit(Xb, Yb, subspace_opt)

            self.list_dtl.append(opt_dtl)
            self.list_dtl_subspace.append(subspace_opt)


    def predict_prob(self, X):
        matrix_predict = np.zeros([self.n_estimators, np.size(X, axis=0)])
        for i in range(self.n_estimators):
            dtl = self.list_dtl[i]
            matrix_predict[i, :] = dtl.predict_prob(X[:, self.list_dtl_subspace[i]])

        return np.average(matrix_predict, axis=0)

    def predict(self, X):
        matrix_predict = np.zeros([self.n_estimators, np.size(X, axis=0)])
        for i in range(self.n_estimators):
            dtl = self.list_dtl[i]
            matrix_predict[i, :] = dtl.predict(X[:, self.list_dtl_subspace[i]])

        return np.average(matrix_predict, axis=0) > 0.5

    def mcr(self, X_test, Y_test):
        return np.average(self.predict(X) != Y)

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    n = 100
    X1 = np.random.uniform(size=[20000, 1])
    X2 = np.random.uniform(size=[20000, 1])
    X = np.concatenate([X1, X2], axis=1)
    Y = (X[:, 0] > 0.5)

    X_train = X[:10000, :]
    X_test = X[10000:, :]
    Y_train = Y[:10000]
    Y_test = Y[10000:]

    bdtl = Bagging_DTL_Classifier(n_estimators=10)
    bdtl.max_depth = 2
    bdtl.fit(X_train, Y_train)
    print(bdtl.predict(X_test))



