# https://stackoverflow.com/questions/1896918

import unittest
import numpy as np

from model.ridge import Ridge


def standardize(X):
    Xnorm = (X - np.mean(X, axis = 0))
    Xstd = Xnorm / np.sqrt((Xnorm * Xnorm).sum(axis = 0))
    return Xstd


def ridge_data(nsample, nvar, errsigma, seed=200):
    np.random.seed(seed)
    X = np.random.normal(0, 1, nsample * nvar).reshape(nsample, nvar)
    X = standardize(X)
    btrue = np.random.normal(0, 1, nvar)
    y = np.dot(X, btrue) + np.random.normal(0, errsigma, nsample)
    y = y - np.mean(y)
    y = y / np.std(y)
    return X, y, btrue


class TestEBMR (unittest.TestCase):

    def test_ridge_em_ebmr_full(self):
        n = 50
        p = 100
        sd = 1.0
        X, y, btrue = ridge_data(n, p, sd)

        m_em   = Ridge(solver='em', max_iter = 100)
        m_em.fit(X, y)

        m_ebmr = Ridge(solver='ebmr', tol=1e-4, max_iter=1000)
        m_ebmr.fit(X, y)

        self.assertAlmostEqual(m_em.updates_['loglik'][-1], 
                               m_ebmr.updates_['loglik'][-1],
                               places=2)


if __name__ == '__main__':
    unittest.main()
