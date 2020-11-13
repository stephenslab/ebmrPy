import numpy as np
from scipy import linalg as sc_linalg
from utils import log_density


'''
Adapted from M. Stephens (https://stephens999.github.io/misc/ridge_em.html)

Solve
y ~ N(Xb, s2)
b ~ N(0, sb2)
using EM algorithm.
'''
def ridge(X, Y, s2, sb2, 
        max_iter=100, tol=1e-4):

    XTX = np.dot(X.T, X)
    XXT = np.dot(X, X.T)
    XTY = np.dot(X.T, Y)
    YTY = np.dot(Y.T, Y)
    n_samples, n_features = X.shape
    loglik = np.zeros(max_iter+1)

    itn = 0
    SigmaY = sb2 * XXT + np.eye(n_samples) * s2
    loglik[itn] = log_density.mgauss(Y.reshape(-1, 1), np.zeros((n_samples, 1)), SigmaY)

    while itn < max_iter:
        V = XTX + np.eye(n_features) * (s2 / sb2)
        Vinv = sc_linalg.cho_solve(sc_linalg.cho_factor(V, lower=True), np.eye(n_features))
        Sigmab = s2 * Vinv  # posterior variance of b
        mub = np.dot(Vinv, XTY) # posterior mean of b
        b2m = np.einsum('i,j->ij', mub, mub) + Sigmab
        s2 = (YTY + np.dot(XTX, b2m).trace() - 2 * np.dot(XTY, mub)) / n_samples
        sb2 = np.sum(np.square(mub) + np.diag(Sigmab)) / n_features

        itn += 1
        SigmaY = sb2 * XXT + np.eye(n_samples) * s2
        loglik[itn] = log_density.mgauss(Y.reshape(-1, 1), np.zeros((n_samples, 1)), SigmaY)

        if loglik[itn] - loglik[itn-1] < tol: break

    return s2, sb2, mub, Sigmab, loglik[:itn+1], itn


'''
Adapted from M. Stephens (https://stephens999.github.io/misc/ridge_em_svd.html)

y = Xb + e
X = UDV'

Scale with y = U'y and sb * theta = U'Xb

Solve
y ~ N(sb * theta, s2)
b ~ N(0, l2 * d2)
using EM algorithm
'''
def ridge_svd(y, d2, 
              s2_init=1.0, sb2_init=1.0, l2_init=1.0,
              tol=1e-4,
              max_iter=1000):

    k = y.shape[0]
    s2 = s2_init
    sb2 = sb2_init
    l2 = l2_init

    logmarglik = np.zeros(max_iter+1)
    itn = 0
    logmarglik[itn] = log_density.mgauss_diagcov(y, np.zeros(k), s2 + sb2 * l2 * d2)
    while itn < max_iter:
        prior_var = l2 * d2
        data_var = s2 / sb2
        post_var = 1 / ((1 / prior_var) + (1 / data_var))
        post_mean = post_var * (1 / data_var) * (y / np.sqrt(sb2))

        theta2 = np.square(post_mean) + post_var
        sb2 = np.square(np.sum(y * post_mean) / np.sum(theta2))
        l2 = np.mean(theta2 / d2)

        r = y - np.sqrt(sb2) * post_mean
        s2 = np.mean(np.square(r) + sb2 * post_var)

        itn += 1
        logmarglik[itn] = log_density.mgauss_diagcov(y, np.zeros(k), s2 + sb2 * l2 * d2)
        if logmarglik[itn] - logmarglik[itn-1] < tol: break

    return s2, sb2, l2, logmarglik[:itn+1], itn
