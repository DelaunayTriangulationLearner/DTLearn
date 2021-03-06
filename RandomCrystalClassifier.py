from F_D_Lambda_Classifier import *
import itertools
import warnings
warnings.filterwarnings("ignore")


class RandomCrystalClassifier:
    def __init__(self, n_estimator=None, max_dim=None, n_bootstrap=None, p_bootstrap=None,
                 Lambda=None, alpha=None, eps=None, h=None,
                 List_f_d_lambda=None, List_is_in=None, List_subspace=None, initial=None,
                 weight_inside=None, max_features=None, mode=None):

        self.mode = mode
        self.max_features = max_features
        self.n_estimator = n_estimator
        self.max_dim = max_dim
        self.n_bootstrap = n_bootstrap
        self.p_bootstrap = p_bootstrap
        self.Lambda = Lambda
        self.alpha = alpha
        self.eps = eps
        self.h = h
        self.List_f_d_lambda = List_f_d_lambda
        self.List_is_in = List_is_in
        self.List_subspace = List_subspace
        self.initial = initial
        self.weight_inside = weight_inside

        if self.mode is None:
            self.mode = 'linear'
        else:
            self.mode = 'tree'

        if self.weight_inside is None:
            self.weight_inside = 0.99

        if self.p_bootstrap is None:
            self.p_bootstrap = 1

        if self.List_f_d_lambda is None:
            self.List_f_d_lambda = []

        if self.List_is_in is None:
            self.List_is_in = []

        if self.n_estimator is None:
            self.n_estimator = 100

        if self.initial is None:
            self.initial = 'Y'
        if self.Lambda is None:
            self.Lambda = 0

    def fit(self, X_train, Y_train):
        if self.mode is None:
            self.mode = 'linear'
        else:
            self.mode = 'tree'

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
        list_dims = []
        for dim_subspace in range(1, self.max_dim + 1):
            list_dims.extend(np.random.permutation(list(itertools.combinations(range(np.size(X_train, axis=1)),
                                                                               dim_subspace))))
        for k in range(self.n_estimator):
            print (k)
            Xb, Yb, Xoob, Yoob = Bootstrap(X_train, Y_train)  # bootstrap data
            # iterate all possible subspaces and greedy find the optimal one.
            min_mcr = np.inf  # training r square
            opt_f_d_lambda = F_D_Lambda_classifier()

            list_dims_loop = list_dims
            list_dims_loop = np.random.choice(list_dims_loop,  int(self.p_bootstrap*len(list_dims_loop)), replace=False)

            for dims in list_dims_loop:
                # d=1
                if len(dims) == 1:
                    try:
                        f_d_lambda = F_D_Lambda_classifier(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps, h=self.h)

                        f_d_lambda.quick_fit1(Xb[:, dims[0]], Yb)
                        mcr = f_d_lambda.mcr(Xoob[:, dims], Yoob)
                        if mcr < min_mcr:
                            min_mcr = mcr
                            opt_f_d_lambda = f_d_lambda
                            opt_subspace = list(dims)
                    except:
                        pass
                else:
                    # d>=2
                    try:

                        f_d_lambda = F_D_Lambda_classifier(Lambda=self.Lambda, alpha=self.alpha, eps=self.eps,
                                                           h=self.h,
                                                           )
                        f_d_lambda.quick_fit(Xb[:, dims], Yb)
                        mcr = f_d_lambda.mcr(Xoob[:, dims], Yoob)

                        # filter the optimal subspace and base learner
                        if mcr < min_mcr:
                            min_mcr = mcr
                            opt_f_d_lambda = f_d_lambda
                            opt_subspace = list(dims)
                    except:
                        # print 'Collinearity'
                        pass

            self.List_subspace.append(opt_subspace)
            self.List_f_d_lambda.append(opt_f_d_lambda)

        return self.List_f_d_lambda, self.List_subspace

    def predict(self, X_predict):
        List_Y_predict = np.zeros([self.n_estimator, len(X_predict)])
        List_is_in = np.zeros([self.n_estimator, len(X_predict)])
        for k in range(self.n_estimator):
            subspace = np.array(self.List_subspace[k])
            f_d_lambda = self.List_f_d_lambda[k]
            f_d_lambda.is_inside(X_predict[:, subspace.astype(int)])
            List_is_in[k, :] = f_d_lambda.is_inside(X_predict[:, subspace.astype(int)])
            Y_predict = f_d_lambda.predict_prob(X_predict[:, subspace.astype(int)])
            List_Y_predict[k, :] = Y_predict

        Para_List_is_in = (1-List_is_in)*self.weight_inside + List_is_in*(1-self.weight_inside)
        Para_List_is_in = Para_List_is_in/np.sum(Para_List_is_in, axis=0)
        List_final_predict = np.zeros(len(X_predict))
        for i in range(np.size(X_predict, 0)):
            List_final_predict[i] =\
                np.dot(List_Y_predict[:, i].transpose(), Para_List_is_in[:, i])
        return List_final_predict > 0.5

    def mcr(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        MCR = np.sum(Y_predict != Y_test)/float(len(Y_test))
        return MCR

if __name__ == '__main__':
    from DataGenerator_Classification import *

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

    rc = RandomCrystalClassifier(n_estimator=10)
    rc.fit(X_train, Y_train)
    print (rc.predict(X_test))

