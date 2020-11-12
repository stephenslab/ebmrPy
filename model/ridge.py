"""
Ridge regression
"""

import numpy as np
from scipy import linalg as sc_linalg

from inference.ebmr import EBMR
from inference import penalized_em
from utils.logs import MyLogger

logger = MyLogger(__name__)

def ridge_regression(X, y,
                     s2_init, sb2_init,
                     max_iter,
                     tol=1e-3,
                     solver='ebmr',
                     variant='full',
                     fast_k = None
                    ):

    n_samples, n_features = X.shape

    logger.debug("Using {:s} solver".format(solver))

    if solver == 'em':
        s2, sb2, b_postmean, b_postvar, loglik, n_iter = penalized_em.ridge(X, y, s2_init, sb2_init, max_iter)
        updates = {'loglik': loglik}


    elif solver == 'em_svd':
        U, D, Vh = sc_linalg.svd(X, full_matrices=False)
        Xtilde = np.dot(np.diag(D), Vh)
        ytilde = np.dot(U.T, y)
        d2 = np.square(D)
        s2, sb2, l2, loglik, n_iter = \
            penalized_em.ridge_svd(ytilde, d2,
                                  s2_init = s2_init, sb2_init = sb2_init, 
                                  tol=tol,
                                  max_iter=max_iter)
        updates = {'loglik': loglik}
        XTXtilde = np.dot(Xtilde.T, Xtilde)
        XTy = np.dot(X.T, y)
        b_postvar = np.linalg.inv((np.eye(n_features) / sb2) + (XTXtilde / s2))
        b_postmean = np.dot(b_postvar, XTy) / s2
        print(n_iter)
                                                                               

    elif solver == 'ebmr':
        #if variant == 'full':
        #    model = 'full'
        #elif variant == 'woodbury_full':
        #    model = 'woodbury_full'
        #elif variant == 'woodbury_svd_full':
        #    model = 'woodbury_svd_full'
        ebmr_ridge = EBMR(X, y, 
                          prior = 'ridge',
                          model = variant,
                          k = fast_k,
                          s2_init = s2_init, sb2_init = sb2_init,
                          max_iter = max_iter, tol = tol)
        ebmr_ridge.update()
        b_postmean = ebmr_ridge.mu
        b_postvar  = ebmr_ridge.s2 * ebmr_ridge.sigma
        s2 = ebmr_ridge.s2
        sb2 = ebmr_ridge.sb2
        updates = {'loglik': ebmr_ridge.loglik_path,
                   'elbo': ebmr_ridge.elbo_path}
        n_iter = ebmr_ridge.n_iter

    return b_postmean, b_postvar, s2, sb2, updates, n_iter


class Ridge:

    def __init__(self,
                 normalize=False,
                 max_iter=1000, tol=1e-4, solver='auto',
                 variant='full',
                 fast_k=None,
                 s2_init=1.0, sb2_init=1.0,
                 random_state=None):
        # Initial values
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        if self.solver == 'auto': self.solver = 'ebmr'
        self.variant = variant
        self.fast_k = fast_k
        self.s2_init = s2_init
        self.sb2_init = sb2_init
        self.random_state = random_state

        # Return values
        self.coef_ = None
        self.bmean_ = None
        self.bvar_ = None
        self.s2_ = None
        self.updates_ = dict()
        self.n_iter_ = 0

        # Logging
        self.logger = MyLogger(__name__)


    def fit(self, X, y):
        self.bmean_, self.bvar_, self.s2_, _, \
        self.updates_, self.n_iter_ = ridge_regression(X, y,
                                                    self.s2_init, self.sb2_init,
                                                    max_iter=self.max_iter, 
                                                    tol=self.tol, 
                                                    solver=self.solver,
                                                    variant=self.variant,
                                                    fast_k=self.fast_k
                                                    )
        self.coef_ = self.bmean_.copy()
        return self
