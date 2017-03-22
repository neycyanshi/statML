import numpy as np
import scipy.sparse as sparse
from scipy.special import expit
from sklearn.externals.joblib import Memory, dump, load
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt

mem = Memory("./mycache")


@mem.cache  # decorator: get_data = mem.cache(get_data)
def get_data(data_file_name, n_features=123, dtype=np.float32):
    data = load_svmlight_file(data_file_name, n_features, dtype)
    return data[0], data[1]


def check_train_test_consistent(X_train, X_test):
    discrepancy = X_train.shape[-1] - X_test.shape[-1]
    if discrepancy > 0:
        # X_test.indptr = np.hstack((X_test.indptr, np.repeat(X_test.indptr[-1], discrepency))) # pad zeros to bottom row
        X_test._shape = (X_test._shape[0], X_test.shape[1]+discrepancy) # pad zeros to rightest column
        print "padding %d columns of zero feature to X_test..." % discrepancy
    else:
        print "X_train and X_test are consistent."


def feature_augment(X_train, X_test):
    # augmented vector: input features [x, 1], model weights [w, b]
    print "augmenting input feature vector to [x, 1] ..."
    bias_train = sparse.csr_matrix(np.ones((X_train.shape[0], 1), dtype=np.float32))
    bias_test = sparse.csr_matrix(np.ones((X_test.shape[0], 1), dtype=np.float32))
    X_train = sparse.hstack((X_train, bias_train), format="csr")
    X_test = sparse.hstack((X_test, bias_test), format="csr")

    return X_train, X_test


def label_prep(y_train, y_test):
    # make label y in {0.0, 1.0}
    print "changing label -1 to 0.0 ..."
    y_train[y_train != 1.0] = 0.
    y_test[y_test != 1.0] = 0.


def LR_read_dense(data_file_name):
    """
    LR_read_problem(data_file_name) -> [y, x]
    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    y = []
    x_ind = []
    N = 0
    D = 0
    for line in open(data_file_name):
        N += 1
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        y.append(float(label))
        xi = []
        for e in features.split():
            indstr, _ = e.split(":")
            ind = int(indstr) - 1  # start from 0
            xi.append(ind)
            if ind > D: D = ind
        x_ind.append(xi)

    y = np.array(y)
    x = np.ndarray((D + 1, N))
    for i, xi in enumerate(x_ind): x[xi, i] = 1.0
    return y, x


def LR_read_sparse(data_file_name):
    """
	LR_read_sparse(data_file_name) -> [y, x]
	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""


def sparse_dot(a, b, dense_output=False):
    # type: (object, object, object) -> object
    """Dot product that handle the sparse matrix case

    Uses BLAS GEMM as replacement for numpy.dot where possible to avoid unnecessary copies
    """
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        # if np_version < (1, 7, 2): # np_version = _parse_version(np.__version__)
        #     warnings.warn("Falling back to slow dot product in older Numpy.")
        return np.dot(a, b)


def sparse_split(mat, row_divs=[]):
    """
    mat is a sparse matrix
    row_divs is a list of divisions between rows.  N row_divs will produce N+1 rows of sparse matrices
    return a list of sparse matrices
    """
    if sparse.issparse(mat):
        row_divs = [None] + row_divs + [None]
        list_of_mats = []
        for (rs, re) in zip(row_divs[:-1], row_divs[1:]):
            list_of_mats += [mat[rs:re]]
    else:
        list_of_mats = np.array_split(mat, len(row_divs)+1, axis=0)

    return list_of_mats


def sparse_block_split(mat, row_divs=[], col_divs=[]):
    """
    mat is a sparse matrix
    row_divs is a list of divisions between rows.  N row_divs will produce N+1 rows of sparse matrices
    col_divs is a list of divisions between cols.  N col_divs will produce N+1 cols of sparse matrices
    return a 2-D array of sparse matrices
    """
    row_divs = [None] + row_divs + [None]
    col_divs = [None] + col_divs + [None]

    mat_of_mats = np.empty((len(row_divs) - 1, len(col_divs) - 1), dtype=type(mat))
    for i, (rs, re) in enumerate(zip(row_divs[:-1], row_divs[1:])):
        for j, (cs, ce) in enumerate(zip(col_divs[:-1], col_divs[1:])):
            mat_of_mats[i, j] = mat[rs:re, cs:ce]

    return mat_of_mats


def my_expit(x):
    # 1 / (1 + exp(-x)) = (1 + tanh(x / 2)) / 2
    # This way of computing the logistic is both fast and stable. learn from scikit-learn/fixes.py
    x *= 0.5
    np.tanh(x, x)
    x += 1
    x *= 0.5

    return x


def my_exp_zero(x):
    # if x<0, return exp(x)/(1+exp(x))
    # this way of computing the logistic is stable.
    return np.where(x>0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


def _intercept_dot(w, X):
    """Computes np.dot(X, w)

    It takes into consideration if the intercept should be fit or not."""
    c = 0.
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    score = sparse_dot(X, w) + c
    return w, c, score


class LR_IRLS():
    """Logistic Regression classifier using IRLS algorithm to train.

    Parameters
    ----------
    reg : float, default: 1.0
        Regularization strength; must be a positive float, bigger values specify stronger regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

    max_iter : int, default: 100
        Maximum number of iterations taken for the solvers to converge.

    random_state : int seed, RandomState instance, default: None
        The seed of the pseudo random number generator to use when shuffling the data.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    verbose : int, default: 0
        Set verbose to any positive number for verbosity.

    Attributes
    ----------
    coef_ : array, shape (n_classes, n_features)
        Coefficient of the features in the decision function.

    intercept_ : array, shape (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.

    n_iter_ : array, shape (n_classes,) or (1, )
        Actual number of iterations, returns only 1 element.
    """
    def __init__(self, reg=0.5, tol=1e-4, fit_intercept=True, random_state=None, max_iter=100, verbose=0, save=True):
        self.reg = reg
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        self.save = save

    def fit(self, X, y):
        """Fit the model according to the given tarining data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(self.reg, float) or self.reg < 0:
            raise ValueError("Penalty term must be positive; got (reg=%r)" % self.reg)
        if self.max_iter < 0:
            raise ValueError("Maximum numebr of iteration must be positive; got (max_iter=%r)" % self.max_iter)
        if self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be positive; got (tol=%r)" % self.tol)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("Needs samples of at least 2 classes in the data, but the data contains only one class: %r"
                             % classes_[0])
        if n_classes == 2:
            n_classes = 1
            classes_ = classes_[1:]

        # Consider try different regs in parallel??
        if self.verbose > 0:
            print "training my LR_IRLS model with regularization constant %.1f..." % self.reg
        w_new = np.zeros(n_features)
        # np.random.seed(self.random_state)
        # w_new = np.random.rand(n_features)
        eps = 1e-6
        reach_max_iter = True
        loss_history = []
        for it in xrange(self.max_iter):
            w_old = w_new
            mu = expit(sparse_dot(X, w_new, dense_output=True))
            R = mu * (1 - mu)
            Rinv = 1.0 / np.maximum(R, eps)
            z = sparse_dot(X, w_new) - Rinv * (mu - y)
            b = sparse_dot(X.T, R * z)
            A = sparse_dot(X.T * sparse.dia_matrix((R,0),shape=(R.shape[0],R.shape[0]),copy=False), X) + self.reg * np.eye(n_features)

            w_new = np.dot(np.linalg.pinv(A), b).A1 # Equivalent to np.asarray(x).ravel()

            criterion = np.sum(abs(w_new - w_old))
            loss_history.append(criterion)
            if self.verbose > 1: print "iter: %r criterion = %r" % (it, criterion)
            if criterion < self.tol:
                print "stop at iter %d." % it
                reach_max_iter = False
                break

        if reach_max_iter:
            print "reach max_iter: %d" % self.max_iter
        self.coef_ = w_new[:-1]
        self.intercept_ = w_new[-1]
        if self.save:
            print "saving my Logistic Regression model..."
            dump(self, "LR_IRLS.model")
        return loss_history

    def predict_proba(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("Call fit before prediction")

        if X.shape[1] == self.coef_.shape[0] + 1 and hasattr(self, "intercept_"):
            w = np.hstack((self.coef_, self.intercept_))
        elif X.shape[1] == self.coef_.shape[0] and self.fit_intercept == False:
            w = self.coef_
        else:
            raise ValueError("Mismatch X and w")

        if self.verbose > 1:
            print "L2-norm(w) = %r" % np.sum(w*w)
        score = sparse_dot(X, w, dense_output=True)  # (N,)
        prob = expit(score)
        return prob

    def predict(self, X, y=None):
        prob = self.predict_proba(X)
        y_predict = np.round(prob)

        if y is not None:
            accuracy = 100 * np.mean(y_predict == y)
            print "accuracy of my IRLS: %.2f\n" % accuracy

        return y_predict

    def cross_val_score(self, X, y, num_folds=10):
        self.save = False
        print "reg = %f" % self.reg
        num_samples = X.shape[0]
        orders = np.random.permutation(num_samples)
        Neach_fold, extras =divmod(num_samples, num_folds)
        fold_sizes = (extras * [Neach_fold+1] + (num_folds-extras) * [Neach_fold])
        div_points = np.array(fold_sizes)[:-1].cumsum().tolist()
        X_folds = sparse_split(X[orders], div_points)
        y_folds = np.array_split(y[orders], div_points)

        train_acc = np.empty(num_folds)
        val_acc = np.empty(num_folds)
        for i in range(num_folds):
            print "fold: %d" % (i+1)
            fold_ix = range(num_folds); fold_ix.remove(i)
            X_val = X_folds[i]
            y_val = y_folds[i]
            if num_folds == 1:
                X_train = X_folds[0]
                y_train = y_folds[0]
            else:
                X_train = sparse.vstack([X_folds[j] for j in fold_ix])
                y_train = np.concatenate([y_folds[j] for j in fold_ix])

            self.fit(X_train, y_train)
            y_predict = self.predict(X_val, y_val)
            val_acc[i] = np.mean(y_predict == y_val) # redundant calc acc
            y_predict = self.predict(X_train, y_train)
            train_acc[i] = np.mean(y_predict == y_train)

        val_acc_mean = val_acc.mean()
        train_acc_mean = train_acc.mean()
        print "reg: %f train_acc_mean: %.2f val_acc_mean: %.2f\n" % (self.reg, 100*train_acc_mean, 100*val_acc_mean)
        return val_acc_mean, train_acc_mean



def LR_relu(X, y, maxiter=100, w_init=1, d=0.0001, tol=0.1):
    from numpy import array, diag, dot, maximum, empty, repeat, ones, sum
    from numpy.linalg import inv, pinv

    D, N = X.shape
    X = np.vstack((X, np.ones(N)))
    D += 1
    X = X.T
    y = y.reshape((N, 1))
    n, p = X.shape
    delta = array(repeat(d, n)).reshape(1, n)
    w = repeat(1, n)
    W = diag(w)
    # print "rank of H = %d" % np.linalg.matrix_rank(X.T.dot(W).dot(X))
    B = dot(pinv(X.T.dot(W).dot(X)), (X.T.dot(W).dot(y)))
    for it in range(maxiter):
        print "IRLS2 iter: %d" % it
        _B = B
        _w = abs(y - X.dot(B)).T
        w = float(1) / maximum(delta, _w)
        W = diag(w[0])
        B = dot(pinv(X.T.dot(W).dot(X)), (X.T.dot(W).dot(y)))
        werr = sum(abs(B - _B))
        print "iter %d: werr = %f" % (it, werr)
        if werr < tol:
            return B
    return B


def LR_newton(X, y, reg=0, max_iter=200, w_init=0.01, tol=0.001, seed=1996):
    D, N = X.shape
    X = np.vstack((X, np.ones(N)))
    D += 1
    # w_old = w_init * np.ones(D)
    np.random.seed(seed)
    w_new = w_init * np.random.rand(D)
    for it in xrange(max_iter):
        w_old = w_new
        mu = np.empty(N)
        expit(w_old.dot(X), mu) # expit(x): 1/(1+exp(-x))
        Rflat = mu * (1 - mu)
        grad = X.dot(y - mu)
        H = -np.dot(X * Rflat, X.T)
        w_new = w_old - np.linalg.pinv(H).dot(grad)
        # criterion = np.dot((mu-y).T, (1/R)*(mu-y)) / 2 # Newton criterion
        criterion = np.sum(abs(w_new - w_old))
        # criterion = np.max(np.abs(np.dot(X, (y - mu)))) # grad should be small
        if criterion < tol:
            print "stop at iteration: %d" % it
            return w_new

    print "reach max_iter: %d" % max_iter
    return w_new


def IRLS_dense(X, y, reg=0.5, max_iter=100, w_init=0., tol=1e-4, eps=1e-6, seed=1996, save=True):
    D, N = X.shape
    X = np.vstack((X, np.ones(N)))
    D += 1
    np.random.seed(seed)
    w_new = np.zeros(D, dtype=np.float32)
    # w_new = w_init * np.random.rand(D)
    for it in xrange(max_iter):
        w_old = w_new
        mu = np.empty(N)
        expit(w_new.dot(X), mu)
        R = mu * (1 - mu)
        Rinv = 1.0 / np.maximum(R, eps)
        z = X.T.dot(w_old) - Rinv * (mu - y)
        A = np.dot(X * R, X.T) + reg * np.eye(D)
        b = np.dot(X, R * z)

        w_new = np.dot(np.linalg.pinv(A), b)  # try CG: from scipy.sparse.linalg import cg
        # fun = lambda w: 0.5*np.sum(A.dot(w)**2) - np.sum(b*(A.dot(w))) + 0.5*np.sum(b**2)
        # gfun = lambda w: A.T.dot(A.dot(w)) - b
        # fun = lambda w: w.dot(A.dot(w)) - w.dot(b)
        # gfun = lambda w: A.dot(w) - b
        # w_new, Y, _ = frcg(fun, gfun, w_old)
        #
        # w_new, info = cg(A, b, tol=0.1) # scipy.sparse.linalg.cg

        criterion = np.sum(abs(w_new - w_old))
        # criterion = np.max(np.abs(np.dot(X, (y-mu))))
        if criterion < tol:
            print "stop at iteration: %d" % it
            return w_new

    print "reach max_iter: %d" % max_iter
    if save:
        dump(lr, "lr.model")
    return w_new


def frcg(fun, gfun, x0):
    """Conjugate Gradient Method to solve linear equations.

    Parameters
    ----------
    fun : objective function

    gfun : gradient/first derivative of objective function.

    x0 : initial guess of optimal point.
    """
    maxk = 5000
    rho = 0.6
    sigma = 0.4
    k = 0
    epsilon = 1e-5
    n = np.shape(x0)[0]
    itern = 0
    while k < maxk:
        gk = gfun(x0)
        itern += 1
        itern %= n
        if itern == 1:
            dk = -gk
        else:
            beta = 1.0 * np.dot(gk, gk) / np.dot(g0, g0)
            dk = -gk + beta * d0
            gd = np.dot(gk, dk)
            if gd >= 0.0:
                dk = -gk
        if np.linalg.norm(gk) < epsilon:
            break
        m = 0
        mk = 0
        while m < 20:
            if fun(x0 + rho ** m * dk) < fun(x0) + sigma * rho ** m * np.dot(gk, dk):
                mk = m
                break
            m += 1
        x0 += rho ** mk * dk
        g0 = gk
        d0 = dk
        k += 1

    return x0, fun(x0), k


def sklearn_LR(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression

    print "training model from sklearn.linear_model.LogisticRegression."
    lr = LogisticRegression(random_state=1996)
    lr.fit(X_train, y_train)

    # check gradient == 0
    w0 = lr.coef_.ravel()
    b0 = lr.intercept_
    mu = np.empty(X_train.shape[0])
    expit(X_train * w0 + b0, mu)
    grad = X_train.T * (y_train - mu)
    print "l1-norm of optim state log-likelihood func's gradient: %r" % np.sum(np.abs(grad))

    # predict label of X_test
    y_predict = lr.predict(X_test) # y in {-1.0, +1.0}
    accuracy =100 * np.mean(y_predict == y_test)
    print "accuracy of sklearn.linear_model.LogisticRegression: %.2f\n" % accuracy



if __name__ == "__main__":
    train_name = 'a9a/a9a'
    test_name = 'a9a/a9a.t'
    X_train, y_train = get_data(train_name)  # (32561, 123), (32561,)
    X_test, y_test = get_data(test_name)  # (16281, 122), (16281,)

    # compare with LogisticRegression from sklearn.linear_model
    sklearn_LR(X_train, y_train, X_test, y_test)

    # Logistic Regression using IRLS
    X_train, X_test = feature_augment(X_train, X_test)
    label_prep(y_train, y_test)
    check_train_test_consistent(X_train, X_test)
    manual_seed = 1996
    # test fit and predict
    my_lr = LR_IRLS(reg=1.0, random_state=manual_seed, verbose=2)
    my_lr.fit(X_train, y_train)
    label = my_lr.predict(X_test, y_test)

    # tune regularization strength
    regs = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0]
    results = {}
    best_acc = -1
    best_lr = None
    for reg in regs:
        my_lr = LR_IRLS(reg=reg, random_state=manual_seed, verbose=2)
        val_acc, train_acc = my_lr.cross_val_score(X_train, y_train, num_folds=10)
        results[reg] = (train_acc, val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_lr = my_lr
    print "best validation accuracy achieved during cross-validation: %.2f" % (100*best_acc)


    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")
    plt.show()


    """
    # try preprocess feature by scaling to unit variance before regression, however get the same acc, futile...
    # with regularization, it's recommended to standardize data in preprocessing??
    # feature distribution not like gaussian, do not make sense??
    from sklearn import preprocessing
    # with_mean = False to keep sparsity
    X_train_scaled = preprocessing.scale(X_train, axis=0, with_mean=False, with_std=True, copy=True)
    X_test_scaled = preprocessing.scale(X_test, axis=0, with_mean=False, with_std=True, copy=True)
    manual_seed = 1996
    my_lr = LR_IRLS(random_state=manual_seed, verbose=1)
    my_lr.fit(X_train_scaled, y_train)
    label = my_lr.predict(X_test_scaled, y_test)
    """


    """
    # reuse trained LR_IRLS.model
    old_model = load("LR_IRLS.model")
    label = old_model.predict(X_test, y_test)
    """

    print "done!"
