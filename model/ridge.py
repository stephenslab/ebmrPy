"""
Ridge regression
"""

import numpy as np
from scipy import linalg as sc_linalg
from inference.ebmr import EBMR
from inference import penalized_em
from inference import f_sigma
from utils.logs import MyLogger

class Ridge:

    def __init__(self,
                 normalize=False,
                 s2_init=1.0, sb2_init=1.0,
                 max_iter=1000, tol=1e-4,
                 solver='auto',
                 ebmr_args=['None', 'full', 'direct'],
                 fast_k=None):
        # Initial values
        self.normalize = normalize
        self.s2_init   = s2_init
        self.sb2_init  = sb2_init
        self.max_iter  = max_iter
        self.tol       = tol
        self.solver    = solver
        self.ebmr_args = ebmr_args
        self.fast_k    = fast_k
        if self.solver == 'auto': self.solver = 'ebmr'
        self.logger = MyLogger(__name__)

        # Return values
        self.coef_    = None
        self.bmean_   = None
        self.bvar_    = None
        self.s2_      = None
        self.sb2_     = None
        self.updates_ = dict()
        self.n_iter_  = 0

        # Logging
        self.logger = MyLogger(__name__)


    def fit(self, X, y):

        n_samples, n_features = X.shape
    
        self.logger.debug("Using {:s} solver".format(self.solver))

        if self.solver == 'em':
            # Obtain scaled parameters and convert to model parameters
            s2, _sb2, bmean, bvar, mll, n_iter = \
                penalized_em.ridge(X, y, 
                                   self.s2_init, self.sb2_init, self.max_iter)
            sb2 = _sb2 / s2
            updates = {'loglik': mll}

        elif self.solver == 'em_svd':
            U, D, Vh = sc_linalg.svd(X, full_matrices=False)
            ytilde = np.dot(U.T, y)
            # Obtain scaled parameters
            s2, _sb2, _l2, mll, n_iter = \
                penalized_em.ridge_svd(ytilde, np.square(D),
                                      s2_init = self.s2_init, 
                                      sb2_init = self.sb2_init, 
                                      tol = self.tol,
                                      max_iter = self.max_iter)
            # Convert to model parameters
            sb2 = _l2 * _sb2 / s2

            updates = {'loglik': mll}
            Xtilde = np.dot(np.diag(D), Vh)
            XTXtilde = np.dot(Xtilde.T, Xtilde)
            #XTX = np.dot(X.T, X)
            XTy = np.dot(X.T, y)
            #sigma = f_sigma.direct(XTX, np.eye(X.shape[1]), sb2)
            #bvar = s2 * sigma
            #bmean = np.dot(sigma, XTy) 
            bvar = np.linalg.inv((np.eye(n_features) / _sb2) + (XTXtilde / s2))
            bmean = np.dot(bvar, XTy) / s2
 
        elif self.solver == 'ebmr':
            ebmr_ridge = EBMR(X, y, 
                              prior = 'ridge',
                              grr = self.ebmr_args[0],
                              sigma = self.ebmr_args[1],
                              inverse = self.ebmr_args[2],
                              k = self.fast_k,
                              s2_init = self.s2_init, sb2_init = self.sb2_init,
                              max_iter = self.max_iter, tol = self.tol,
                              mll_calc = True)
            ebmr_ridge.update()
            bmean = ebmr_ridge.mu
            bvar  = ebmr_ridge.s2 * ebmr_ridge.sigma
            s2    = ebmr_ridge.s2
            sb2   = ebmr_ridge.sb2
            updates = {'loglik': ebmr_ridge.mll_path,
                       'elbo': ebmr_ridge.elbo_path}
            n_iter = ebmr_ridge.n_iter

        # Return values
        self.coef_    = bmean
        self.bmean_   = bmean
        self.bvar_    = bvar
        self.s2_      = s2
        self.sb2_     = sb2
        self.updates_ = updates
        self.n_iter_  = n_iter
 
        return self
