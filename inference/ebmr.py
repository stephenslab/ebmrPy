import numpy as np
from scipy import linalg as sc_linalg

from inference import f_elbo
from utils import log_density
from utils.logs import MyLogger

class EBMR:

    def __init__(self,
                X, y, 
                prior='ridge',
                model='full',
                s2_init=1.0, sb2_init=1.0, 
                max_iter=100, tol=1e-4):
        self.X = X
        self.y = y
        self.prior = prior
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.n_samples, self.n_features = X.shape

        # Initialize other internal variables
        self._s2 = s2_init
        self._sb2 = sb2_init
        self._sigma = np.zeros((self.n_features, self.n_features))
        self._mu = np.zeros(self.n_features)
        self._Wbar = np.ones(self.n_features) 
        self._elbo = -np.inf
        self._loglik_path = np.zeros(max_iter)
        self._elbo_path = np.zeros(max_iter)
        self._n_iter = 0

        # Other variables which will be initialized after the 'update' call
        # self._svd
        # self._XTy
        # self._XTX
        # self._Dinit

        # Logging
        self.logger = MyLogger(__name__)
        self.logger.debug('EBMR using {:s}'.format(self.model))


    @property
    def s2(self):
        return self._s2


    @property
    def sb2(self):
        return self._sb2


    @property
    def sigma(self):
        return self._sigma


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
    def loglik_path(self):
        return self._loglik_path[:self._n_iter]


    @property
    def elbo_path(self):
        return self._elbo_path[:self._n_iter]


    @property
    def n_iter(self):
        return self._n_iter


    def update(self):

        #ipdb.set_trace()

        # Precalculate SVD, X'X, X'y
        self._svd = sc_linalg.svd(self.X, full_matrices=False)
        self._XTX = self.svd2XTX(self._svd)
        self._XTy = np.dot(self.X.T, self.y)
        self._Dinit = np.diag(self._XTX)


        # EBMR iteration
        elbo = self._elbo
        self._n_iter = 0
        for itn in range(self.max_iter):

            # GRR Step
            if self.model == 'full':
                self.update_sigma_direct()
                self.update_mu_direct()

            elif self.model == 'woodbury_full':
                self.update_sigma_woodbury()
                self.update_mu_direct()

            elif self.model == 'woodbury_svd_full':
                self.update_sigma_woodbury_svd()
                self.update_mu_direct()

            # EBNV Step
            #if self.model == 'full':
            self.update_s2()
            self.update_ebnv()
            self.update_elbo()

            self._elbo_path[itn] = self._elbo
            self._loglik_path[itn] = self.grr_loglik()
            self._n_iter += 1
            if self._elbo - elbo < self.tol: break
            elbo = self._elbo

        return


    def update_sigma_direct(self):
        self._sigma = self.cho_inverse(self._XTX + np.diag(1 / self._Wbar))
        return


    def update_sigma_woodbury(self):
        Hinv = self.cho_inverse(np.eye(self.n_samples) + np.linalg.multi_dot([self.X, np.diag(self._Wbar), self.X.T]))
        self._sigma = np.diag(self._Wbar) \
                      - np.linalg.multi_dot([np.diag(self._Wbar), self.X.T, Hinv, self.X, np.diag(self._Wbar)])
        return


    def update_sigma_woodbury_svd(self, k = None):
        U, S, Vh = self._svd
        if k is None:
            k = max(S.shape[0], Vh.shape[0])
        L = np.dot(np.diag(S[:k]), Vh[:k, :])
        D = self._Dinit - np.sum(np.square(L), axis = 0)
        A = D + (1 / self._Wbar)
        Ainv = 1 / A
        Hinv = self.cho_inverse(np.eye(self.n_samples) + np.linalg.multi_dot([L, np.diag(Ainv), L.T]))
        self._sigma = np.diag(Ainv) - np.linalg.multi_dot([np.diag(Ainv), L.T, Hinv, L, np.diag(Ainv)])
        return


    def update_mu_direct(self):
        self._mu = np.dot(self._sigma, self._XTy)
        return


    def update_s2(self):
        A = np.sum(np.square(self.y - np.dot(self.X, self._mu)))
        self._s2 = (A + np.sum(np.square(self._mu) / self._Wbar)) / self.n_samples
        return


    def update_ebnv(self):
        bj2 = np.square(self._mu) + np.diag(self._sigma) * self._s2
        if self.prior == 'ridge':
            W_point_estimate = np.sum(bj2) / self._s2 / self.n_features
            self._Wbar = np.repeat(W_point_estimate, self.n_features)
        return


    def update_elbo(self):
        ElogW = np.log(self._Wbar)
        KLW = 0
        self._elbo = f_elbo.parametric(self.X, self.y, 
                                      self._s2,
                                      self._sb2, 
                                      self._mu, 
                                      self._sigma, 
                                      self._Wbar, 
                                      self._XTX, 
                                      ElogW, KLW)
        return


    def grr_loglik(self):
        sigma_y = self._s2 * (np.eye(self.n_samples) + self._sb2 * np.dot(self.X, np.dot(np.diag(self._Wbar), self.X.T)))
        loglik  = log_density.mgauss(self.y.reshape(-1, 1), np.zeros((self.n_samples, 1)), sigma_y)
        return loglik


    # invert a square matrix using Cholesky
    def cho_inverse(self, A):
        return sc_linalg.cho_solve(sc_linalg.cho_factor(A, lower=True), np.eye(A.shape[0]))


    # calcultae X'X from the SVD of X
    def svd2XTX(self, svd):
        U = svd[0]
        S = svd[1]
        Vh = svd[2]
        nmax = max(S.shape[0], Vh.shape[0])
        S2diag = np.zeros((nmax, nmax))
        S2diag[np.diag_indices(S.shape[0])] = np.square(S)
        return np.dot(Vh.T, np.dot(S2diag, Vh))
