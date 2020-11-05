"""
Ridge regression
"""

import numpy as np
from utils.logs import MyLogger

from inference.ebmr import EBMR
from inference import penalized_em

def ridge_regression(X, y,
                     s2_init, sb2_init,
                     max_iter,
                     tol=1e-3,
                    ):

    n_samples, n_features = X.shape

    if solver == 'em':
        s2, sb2, b_postmean, b_postvar, loglik, n_iter = penalized_em.ridge(X, y, s2_init, sb2_init, max_iter)
        updates = {'loglik': loglik}

    elif solver == 'ebmr':
        ebmr_ridge = EBMR(X, y, 
                          prior = 'ridge',
                          s2_init = s2_init, sb2_init = sb2_init,
                          max_iter = max_iter, tol = tol)
        ebmr_ridge.update()
        b_postmean = ebmr_ridge.mu
        b_postvar  = ebmr_ridge.sigma
        s2 = ebmr_ridge.s2
        sb2 = ebmr_ridge.sb2
        updates = {'loglik': ebmr_ridge.loglik_path,
                   'elbo': ebmr_ridge.elbo_path}
        n_iter = ebmr_ridge.n_iter

    return b_postmean, b_postvar, s2, sb2, updates, n_iter


class Ridge:

    def __init__(self,
                 normalize=False,
                 max_iter=None, tol=1e-3, solver="auto",
                 s2_init=1.0, sb2_init=1.0,
                 random_state=None):
        # Initial values
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
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

    def fit(self, X, y):
        self.bmean_, self.bvar_, self.s2_, _, \
        self.updates_, self.n_iter_ = ridge_regression(X, y,
                                                    self.s2_init, self.sb2_init,
                                                    max_iter=self.max_iter, 
                                                    tol=self.tol, 
                                                    solver=self.solver,
                                                    )
        self.coef_ = self.bmean_.copy()
        return self
