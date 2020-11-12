# https://stackoverflow.com/questions/1896918

import unittest
import numpy as np

from model.ridge import Ridge
from utils.logs import MyLogger

mlogger = MyLogger(__name__)

class TestEBMRRidge(unittest.TestCase):

    def _standardize(self, X):
        Xnorm = (X - np.mean(X, axis = 0)) 
        Xstd = Xnorm / np.sqrt((Xnorm * Xnorm).sum(axis = 0)) 
        return Xstd
    
    
    def _ridge_data(self, n=50, p=100, sd=1.0, seed=200):
        np.random.seed(seed)
        X = np.random.normal(0, 1, n * p).reshape(n, p)
        X = self._standardize(X)
        btrue = np.random.normal(0, 1, p)
        y = np.dot(X, btrue) + np.random.normal(0, sd, n)
        y = y - np.mean(y)
        y = y / np.std(y)
        return X, y, btrue

    def test_em_ebmr_full(self):
        mlogger.info("Check ELBO of EBMR matches logML of EM-Ridge")
        X, y, btrue = self._ridge_data()

        m_em   = Ridge(solver='em', max_iter=100)
        m_em.fit(X, y)

        m_ebmr = Ridge(solver='ebmr', tol=1e-4, max_iter=1000)
        m_ebmr.fit(X, y)

        self._check_elbo(m_ebmr.updates_, 
                         msg="ELBO is different from log marginal likelihood for EBMR")


        self.assertAlmostEqual(m_em.updates_['loglik'][-1], 
                               m_ebmr.updates_['loglik'][-1],
                               places=2,
                               msg="Log marginal likelihood is different for EM and EBMR ridge regression")
        return


    def test_em_ebrr(self):
        mlogger.info("Compare log marginal likelihood of EM-Ridge and EM-Ridge-SVD")
        X, y, btrue = self._ridge_data()
        m_em = Ridge(solver='em', max_iter=100)
        m_em.fit(X, y)
        m_eb = Ridge(solver='em_svd', max_iter=100)
        m_eb.fit(X, y)
        self.assertAlmostEqual(m_em.updates_['loglik'][-1],
                               m_eb.updates_['loglik'][-1],
                               places=3,
                               msg="Log marginal likelihood is different for EM-Ridge and EM-Ridge-SVD")
        return


    def _check_sigma_variant(self, variant1, variant2, msg=None):
        X, y, btrue = self._ridge_data()
        m1 = Ridge(solver='ebmr', variant=variant1)
        m1.fit(X, y)
        m2 = Ridge(solver='ebmr', variant=variant2)
        m2.fit(X, y)
        self.assertTrue(np.allclose(m1.bvar_, m2.bvar_), msg)
        return m1, m2


    def _check_elbo(self, resdict, msg=None):
        final_elbo = resdict['elbo'][-1]
        final_logmarglik = resdict['loglik'][-1]
        mlogger.debug(f"ELBO: {final_elbo:g}")
        self.assertAlmostEqual(final_elbo, final_logmarglik, places=2, msg=msg)
        return


    def test_woodbury_full(self):
        mlogger.info("Compare EBMR ridge regression with and without Woodbury")
        m1, m2 = self._check_sigma_variant('full', 'woodbury_full',
                                           msg="Sigma is different from direct and Woodbury")
        self._check_elbo(m2.updates_, 
                         msg="ELBO is different from log marginal likelihood for EBMR ridge regression with Woodbury")
        return


    def test_woodbury_svd_full(self):
        mlogger.info("Compare EBMR ridge regression with and without SVD")
        m1, m2 = self._check_sigma_variant('woodbury_full', 'woodbury_svd_full',
                                           msg="Sigma is different from Woodbury and Woodbury-SVD")
        self._check_elbo(m2.updates_, 
                         msg="ELBO is different from log marginal likelihood for EBMR ridge regression with Woodbury-SVD")
        return

    def test_woodbury_svd_fast(self):
        mlogger.info("Compare EBMR ridge regression with and without reduced dimension (k)")
        X, y, btrue = self._ridge_data(n=50, p=100, sd=1.0) 

        m_ebmr = Ridge(solver='ebmr', variant='full')
        m_ebmr.fit(X, y)

        m_ebmr_fast = Ridge(solver='ebmr', variant='woodbury_svd_fast', fast_k = 49)
        m_ebmr_fast.fit(X, y)

        self.assertAlmostEqual(m_ebmr.updates_['elbo'][-1],
                               m_ebmr_fast.updates_['elbo'][-1],
                               places=2,
                               msg="ELBO is different for EBMR with k=n and k=n-1")

    def test_ebrr_svd(self):
        mlogger.info("Compare EBMR ridge regression with and without EB-Ridge for GRR")
        X, y, btrue = self._ridge_data()

        m_ebmr = Ridge(solver='ebmr', variant='full')
        m_ebmr.fit(X, y)

        m_ebmr_fast = Ridge(solver='ebmr', variant='ebrr_svd')
        m_ebmr_fast.fit(X, y)

        self._check_elbo(m_ebmr_fast.updates_,
                         msg="ELBO is different from log marginal likelihood for EBMR with EM-Ridge-SVD")

        #self.assertAlmostEqual(m_ebmr.updates_['elbo'][-1],
        #                       m_ebmr_fast.updates_['elbo'][-1],
        #                       places=2,
        #                       msg="ELBO is different for EBMR with and without EBRR")


if __name__ == '__main__':
    unittest.main()
