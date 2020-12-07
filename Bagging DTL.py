from DTL import *
import itertools

class Bagging_DTL():
    def __init__(self):
        self.n_estimators = 100
        self.max_depth = None
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
        def mse_evaluate(Xb, Yb, Xoob, Yoob, subspace):
            dtl = DTL()

            dtl.fit(Xb[:, subspace], Yb)
            mse = np.var(Yoob - dtl.predict(Xoob[:, subspace]))
            return mse

        def dtl_fit(Xb, Yb, subspace):
            dtl = DTL()
            dtl.fit(Xb[:, subspace], Yb)
            return dtl

        self.list_dtl = []
        self.list_dtl_subspace = []

        for i in range(self.n_estimators):
            print (i)
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

            self.list_dtl.append(opt_dtl)
            self.list_dtl_subspace.append(subspace_opt)


    def predict(self, X):
        matrix_predict = np.zeros([self.n_estimators, np.size(X, axis=0)])
        for i in range(self.n_estimators):
            dtl = self.list_dtl[i]
            matrix_predict[i, :] = dtl.predict(X[:, self.list_dtl_subspace[i]])

        return np.average(matrix_predict, axis=0)



if __name__ == '__main__':
    n = 100
    p = 10
    X_train = np.random.uniform(size=[n, p])
    Y_train = np.sum(X_train ** 1, axis=1)
    X_test = np.random.normal(size=[n, p])
    Y_test = np.sum(X_test ** 1, axis=1)

    bdtl = Bagging_DTL()
    bdtl.max_depth = 2
    bdtl.fit(X_train, Y_train)

