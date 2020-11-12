import numpy as np
from scipy import linalg as sc_linalg

from inference import f_sigma
from inference import f_elbo
from inference import penalized_em
from utils import log_density
from utils.logs import MyLogger

class EBMR:

    def __init__(self,
                X, y, 
                prior='ridge',
                grr='None', # None | em | em_svd
                sigma='full', # full | diagonal
                inverse='direct', # direct | woodbury | woodbury_svd | woodbury_svd_fast
                k = None, # used for woodbury_svd_fast
                s2_init=1.0, sb2_init=1.0, 
                max_iter=100, tol=1e-4,
                mll_calc=True):

        self.X           = X
        self.y           = y
        self.prior       = prior
        self.grr_model   = grr
        self.sigma_model = sigma
        self.inv_model   = inverse
        self.max_iter    = max_iter
        self.tol         = tol
        self.k           = k
        self.mll_calc    = mll_calc
        self.n_samples, self.n_features = X.shape

        # Initialize other internal variables
        self._s2         = s2_init
        self._sb2        = sb2_init
        self._sigma      = np.zeros((self.n_features, self.n_features))
        self._sigma_diag = np.zeros(self.n_features)
        self._mu         = np.zeros(self.n_features)
        self._Wbar       = np.ones(self.n_features) 
        self._mll_path   = np.zeros(max_iter)
        self._elbo_path  = np.zeros(max_iter)
        self._n_iter     = 0

        # Other variables which will be initialized after the 'update' call
        # self._svd
        # self._XTy
        # self._XTX
        # self._Dinit

        # Logging
        self.logger = MyLogger(__name__)
        self.logger.debug(f'EBMR using {self.prior} prior, {self.grr_model} grr,' 
                        + f' {self.sigma_model} b posterior variance, {self.inv_model} inversion')


    @property
    def s2(self):
        return self._s2


    @property
    def sb2(self):
        return self._sb2


    @property
    def sigma(self):
        if self.sigma_model == 'full':
            return self._sigma
        elif self.sigma_model == 'diagonal':
            return self._sigma_diag


    @property
    def mu(self):
        return self._mu


    @property
    def Wbar(self):
        return self._Wbar


    @property
    def elbo(self):
        return self._elbo

    @property
    def mll_path(self):
        return self._mll_path[:self._n_iter]


    @property
    def elbo_path(self):
        return self._elbo_path[:self._n_iter]


    @property
    def n_iter(self):
        return self._n_iter


    def update(self):

        # Precalculate SVD, X'X, X'y
        self._svd   = sc_linalg.svd(self.X, full_matrices=False)
        self._XTX   = self.svd2XTX(self._svd)
        self._XTy   = np.dot(self.X.T, self.y)
        self._Dinit = np.diag(self._XTX)

        # EBMR iteration
        itn = 0
        self._n_iter = 0
        self._elbo = -np.inf
        self._elbo_path[itn] = -np.inf
        if self.mll_calc: self._mll_path[itn] = -np.inf
        for itn in range(1, self.max_iter+1):

            # GRR Step
            if self.grr_model == 'None':
                self.grr()
            elif self.grr_model == 'em':
                self.grr_em()
            elif self.grr_model == 'em_svd':
                self.grr_em_svd()

            # EBNV Step
            self.update_ebnv()
            self.update_elbo()

            # Bookkeeping
            self._elbo_path[itn] = self._elbo
            self._n_iter += 1
            if self.mll_calc: self._mll_path[itn] = self.grr_logmarglik()

            # Check convergence
            if self._elbo_path[itn] - self._elbo_path[itn-1] < self.tol: break

        return


    '''
    Calculate GRR analytically, using sb2=1.0
    '''
    def grr(self):
        self.update_sigma()
        self.update_mu()
        A = np.sum(np.square(self.y - np.dot(self.X, self._mu)))
        self._s2 = (A + np.sum(np.square(self._mu) / self._Wbar)) / self.n_samples
        return


    '''
    Calculate GRR using EM-Ridge
    '''
    def grr_em(self):
        Xtilde = np.dot(self.X, np.diag(np.sqrt(self._Wbar)))
        _s2, _sb2, _bmu, _bsigma, _logmarglik, _grr_iter = \
            penalized_em.ridge(Xtilde, self.y, self._s2, self._sb2 * self._s2, 100)
        self._sb2 = _sb2 / _s2
        self._s2  = _s2
        self.update_sigma()
        self.update_mu()
        return


    '''
    Calculate GRR using EM-Ridge-SVD
    '''
    def grr_em_svd(self):
        Xtilde = np.dot(self.X, np.diag(np.sqrt(self._Wbar)))
        U, D, Vh = sc_linalg.svd(Xtilde)
        ytilde = np.dot(U.T, self.y)
        l2_init = 1.0
        _s2, _sb2, _l2, _logmarglik, _grr_iter = \
            penalized_em.ridge_svd(ytilde, np.square(D),
                                   s2_init=self._s2, 
                                   sb2_init=self._sb2 * self._s2 / l2_init, 
                                   l2_init=l2_init, 
                                   tol=1e-4, max_iter=100)
        self._sb2 = _sb2 * _l2 / _s2
        self._s2 = _s2
        self.update_sigma()
        self.update_mu()
        return


    '''
    Updating sigma requires inverting a pxp matrix directly (X'X + W.inv / sb2)
    Here, we implement different versions of updating sigma.
    Options are:
        sigma model = full | diagonal
        inv_model   = direct | woodbury | woodbury_svd | woodbury_svd_fast 
    '''
    def update_sigma(self):

        if self.sigma_model == 'full':
            if self.inv_model == 'direct':
                self._sigma = f_sigma.direct(self._XTX, self._Wbar, self._sb2)
            elif self.inv_model == 'woodbury':
                self._sigma = f_sigma.woodbury(self.X, self._Wbar, self._sb2)
            elif self.inv_model == 'woodbury_svd':
                self._sigma = f_sigma.woodbury_svd(self._svd, self._Dinit, self._Wbar, self._sb2)
            elif self.inv_model == 'woodbury_svd_fast':
                self._sigma = f_sigma.woodbury_svd(self._svd, self._Dinit, self._Wbar, self._sb2, k = self.k)

        elif self.sigma_model == 'diagonal':
            if self.inv_model == 'direct':
                self._sigma_diag = f_sigma.direct_diag(self._XTX, self._Wbar, self._sb2)
            elif self.inv_model == 'woodbury':
                self._sigma_diag = f_sigma.woodbury_diag(self.X, self._Wbar, self._sb2)
            elif self.inv_model == 'woodbury_svd':
                self._sigma_diag = f_sigma.woodbury_svd_diag(self._svd, self._Dinit, self._Wbar, self._sb2)
            elif self.inv_model == 'woodbury_svd_fast':
                self._sigma_diag = f_sigma.woodbury_svd_diag(self._svd, self._Dinit, self._Wbar, self._sb2, k = self.k)

        return


    def update_mu(self):
        if self.sigma_model == 'full':
            self._mu = np.dot(self._sigma, self._XTy)
        elif self.sigma_model == 'diagonal':
            self._mu = np.dot(np.diag(self._sigma_diag), self._XTy)
        return


    def update_ebnv(self):
        if self.sigma_model == 'full':
            bj2 = np.square(self._mu) + np.diag(self._sigma) * self._s2
        elif self.sigma_model == 'diagonal':
            bj2 = np.square(self._mu) + self._sigma_diag * self._s2
        if self.prior == 'ridge':
            W_point_estimate = np.sum(bj2) / self._s2 / self.n_features
            self._Wbar = np.repeat(W_point_estimate, self.n_features)
        return


    def update_elbo(self):
        ElogW = np.log(self._Wbar)
        KLW = 0
        if self.sigma_model == 'full':
            b_postvar = self._sigma
        elif self.sigma_model == 'diagonal':
            b_postvar = np.diag(self._sigma_diag)
        self._elbo = f_elbo.parametric(self.X, self.y, 
                                      self._s2,
                                      self._sb2,
                                      self._mu,
                                      b_postvar,
                                      self._Wbar, 
                                      self._XTX, 
                                      ElogW, KLW)
        return


    def grr_logmarglik(self):
        sigma_y = self._s2 * (np.eye(self.n_samples) + self._sb2 * np.dot(self.X, np.dot(np.diag(self._Wbar), self.X.T)))
        loglik  = log_density.mgauss(self.y.reshape(-1, 1), np.zeros((self.n_samples, 1)), sigma_y)
        return loglik


    # calcultae X'X from the SVD of X
    def svd2XTX(self, svd):
        U = svd[0]
        S = svd[1]
        Vh = svd[2]
        nmax = max(S.shape[0], Vh.shape[0])
        S2diag = np.zeros((nmax, nmax))
        S2diag[np.diag_indices(S.shape[0])] = np.square(S)
        return np.dot(Vh.T, np.dot(S2diag, Vh))
