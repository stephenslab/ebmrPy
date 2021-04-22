import numpy as np
from ..utils import log_density

class IRidge:

    def __init__(self, X, y, 
                 s2_init = 1.0, sb2_init = 1.0, sw2_init = 1.0,
                 max_iter = 1000, tol = 1e-8):
        self.X           = X
        self.y           = y
        self._mll_path   = np.zeros(max_iter+1)
        self._max_iter   = max_iter
        self._tol        = tol

        n, p             = X.shape
        self._mub        = np.ones(p)
        self._muw        = np.ones(p)
        self._s2         = s2_init
        self._sb2        = sb2_init
        self._sw2        = sw2_init
        self._n_iter     = 0
        return


    @property
    def s2(self):
        return self._s2


    @property
    def sb2(self):
        return self._sb2


    @property
    def sw2(self):
        return self._sw2


    @property
    def mub(self):
        return self._mub


    @property
    def muw(self):
        return self._muw


    @property
    def beta(self):
        return np.multiply(self._muw, self._mub)


    @property
    def n_iter(self):
        return self._n_iter


    @property
    def mll_path(self):
        return self._mll_path[:self._n_iter+1]


    def update(self):
        ''' 
        Iteration 0
        '''
        self._mll_path[0] = self.ridge_mll()
        ''' 
        Iterations
        '''
        for itn in range(1, self._max_iter + 1):
            '''
            Iterate between B and W updates
            '''
            self.update_b()
            self.update_w()
            ''' 
            Convergence
            '''
            self._n_iter += 1
            self._mll_path[itn] = self.ridge_mll()
            if self._mll_path[itn] - self._mll_path[itn - 1] < self._tol: break
        return


    def update_b(self):
        # Update B (scale X with W)
        Xscale = np.dot(self.X, np.diag(self._muw))
        self._s2, self._sb2, self._mub = self.ridge_em_step(Xscale, self.y, self._s2, self._sb2)
        return


    def update_w(self):
        # Update W (scale X with B)
        Xscale = np.dot(self.X, np.diag(self._mub))
        self._s2, self._sw2, self._muw = self.ridge_em_step(Xscale, self.y, self._s2, self._sw2)
        return


    # A single step for the EM ridge regression
    def ridge_em_step(self, X, y, s2, sb2):
        n, p = X.shape
        XTX = np.dot(X.T, X)                # scaled X', different for each iteration (X' = XW)
        XTy = np.dot(X.T, y)                #
        # E-step. Posterior
        sigmainv = (XTX + np.eye(p) * (s2 / sb2)) / s2
        sigma    = np.linalg.inv(sigmainv) # posterior variance of b
        mu       = np.dot(sigma, XTy) / s2  # posterior mean of b
        # M-step. Prior and residual
        mmT      = np.einsum('i,j->ij', mu, mu)
        Xmu      = np.dot(X, mu)
        s2       = (np.sum(np.square(y - Xmu)) + np.trace(np.dot(XTX, sigma))) / n
        sb2      = np.trace(mmT + sigma) / p
        return s2, sb2, mu
      

    def ridge_mll(self):
        n, p   = self.X.shape
        Xscale = np.dot(self.X, np.diag(self._muw))
        XWWtXt = np.dot(Xscale, Xscale.T)
        sigmay = self._s2 * (np.eye(n) + self._sb2 * XWWtXt)
        muy    = np.zeros((n, 1))
        mLL    = log_density.mgauss(self.y.reshape(-1,1), muy, sigmay)
        return mLL
