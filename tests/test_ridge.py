# https://stackoverflow.com/questions/1896918

import unittest
import numpy as np

import ebmrPy

from ebmrPy.model.ridge import Ridge
from ebmrPy.utils.logs import MyLogger

mlogger = MyLogger(__name__)

class TestRidge(unittest.TestCase):

    def _standardize(self, X):
        Xnorm = (X - np.mean(X, axis = 0)) 
        Xstd = Xnorm / np.sqrt((Xnorm * Xnorm).sum(axis = 0)) 
        return Xstd
    
    
    def _ridge_data(self, n=50, p=100, s=10.0, sb=5.0, seed=100):
        np.random.seed(seed)
        X = np.random.normal(0, 1, n * p).reshape(n, p)
        X = self._standardize(X)
        btrue = np.random.normal(0, sb, p)
        y = np.dot(X, btrue) + np.random.normal(0, s, n)
        y = y - np.mean(y)
        #y = y / np.std(y)
        return X, y, btrue


    def _run_ebmr_methods(self, args1, args2):
        X, y, btrue = self._ridge_data()
        m1 = Ridge(solver = 'ebmr', ebmr_args = args1)
        m2 = Ridge(solver = 'ebmr', ebmr_args = args2)
        m1.fit(X, y)
        m2.fit(X, y)
        return m1, m2


    def _compare_sigma(self, m1, m2, msg=None):
        self.assertTrue(np.allclose(m1.bvar_, m2.bvar_), msg)
        return


    def _compare_s2(self, m1, m2, msg=None):
        self.assertAlmostEqual(m1.s2_, m2.s2_, places=4, msg=msg)
        return


    def _compare_elbo(self, m1, m2, msg=None):
        self.assertAlmostEqual(m1.updates_['elbo'],
                               m2.updates_['elbo'],
                               places = 2,
                               msg = msg)
        return


    def _check_elbo(self, m1, method=['Unknown'], msg=None):
        elbo = m1.updates_['elbo']
        mll  = m1.updates_['mll_path'][-1]
        mlogger.debug(f"ELBO for {method}: {elbo:g}")
        self.assertAlmostEqual(elbo, mll, places=2, msg=msg)
        return


    def test_em_ebmr(self):
        mlogger.info("Check ELBO of EBMR matches logML of EM-Ridge")
        X, y, btrue = self._ridge_data()
        m1 = Ridge(solver='em')
        m2 = Ridge(solver='ebmr', ebmr_args=['mle', 'full', 'direct'])
        m1.fit(X, y)
        m2.fit(X, y)

        self._check_elbo(m2,
                         method=['mle', 'full', 'direct'],
                         msg="ELBO is different from log marginal likelihood for EBMR")

        self.assertAlmostEqual(m1.updates_['mll_path'][-1], 
                               m2.updates_['mll_path'][-1],
                               places=2,
                               msg="Log marginal likelihood is different for EM and EBMR ridge regression")
        return


    def test_em_emsvd(self):
        mlogger.info("Compare log marginal likelihood of EM-Ridge and EM-Ridge-SVD")
        X, y, btrue = self._ridge_data()
        m1 = Ridge(solver='em')
        m2 = Ridge(solver='em_svd')
        m1.fit(X, y)
        m2.fit(X, y)
        self.assertAlmostEqual(m1.updates_['mll_path'][-1],
                               m2.updates_['mll_path'][-1],
                               places=2,
                               msg="Log marginal likelihood is different for EM-Ridge and EM-Ridge-SVD")
        return


    def test_woodbury_full(self):
        mlogger.info("Compare EBMR ridge regression with and without Woodbury")
        method1 = ['mle', 'full', 'direct']
        method2 = ['mle', 'full', 'woodbury']
        m1, m2 = self._run_ebmr_methods(method1, method2)

        self._compare_sigma(m1, m2,
            msg="Sigma is different from direct and Woodbury")
        self._check_elbo(m2, 
            method=method2,
            msg="ELBO is different from log marginal likelihood for EBMR ridge regression with Woodbury")
        return


    def test_woodbury_svd_full(self):
        mlogger.info("Compare EBMR ridge regression with and without SVD")
        method1 = ['mle', 'full', 'direct']
        method2 = ['mle', 'full', 'woodbury_svd']
        m1, m2 = self._run_ebmr_methods(method1, method2)

        self._compare_sigma(m1, m2,
            msg="Sigma is different from Woodbury and Woodbury-SVD")
        self._check_elbo(m2, 
            method=method2,
            msg="ELBO is different from log marginal likelihood for EBMR ridge regression with Woodbury-SVD")
        return


    def test_woodbury_svd_fast(self):
        mlogger.info("Compare EBMR ridge regression with and without reduced dimension (k)")
        method1 = ['mle', 'full', 'direct']
        method2 = ['mle', 'full', 'woodbury_svd']
        m1, m2 = self._run_ebmr_methods(method1, method2)
        self._compare_elbo(m1, m2, 
            msg="ELBO is different for EBMR with k=n and k=n-1")
        return


    def test_grr(self):
        mlogger.info("Compare the different options of EBMR-EM-Ridge")
        method1 = ['mle', 'full', 'direct']
        X, y, btrue = self._ridge_data()
        m1 = Ridge(solver = 'ebmr', ebmr_args = method1)
        m1.fit(X, y)
        for grr in ['mle', 'em', 'em_svd']:
        #for grr in ['em_svd']:
            for sigma in ['full', 'diagonal']:
                for inverse in ['direct', 'woodbury', 'woodbury_svd']:
                    method2 = [grr, sigma, inverse]
                    m2 = Ridge(solver = 'ebmr', ebmr_args = method2)
                    m2.fit(X, y)
                    self._check_elbo(m2,
                        method=method2,
                        msg = f"ELBO does not match log marginal likelihood for EBMR with [{grr}, {sigma}, {inverse}]")
                    self._compare_elbo(m1, m2,
                        msg = f"ELBO is different for {method1} and {method2}")
        return


if __name__ == '__main__':
    unittest.main()
