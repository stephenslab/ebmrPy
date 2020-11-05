import numpy as np
from scipy import linalg as sc_linalg
from utils import logdensity

def ridge(X, Y, s2, sb2, max_iter):
    XTX = np.dot(X.T, X)
    XXT = np.dot(X, X.T
    XTY = np.dot(X.T, Y)
    YTY = np.dot(Y.T, Y)
    n_samples, n_features = X.shape
    loglik = np.zeros(max_iter)
    itn = 0
    while itn < max_iter:
        V = XTX + np.eye(n_features) * (s2 / sb2)
        Vinv = sc_linalg.cho_solve(sc_linalg.cho_factor(V, lower=True), np.eye(n_features))
        SigmaY = sb2 * XXT + np.eye(n_samples) * s2
        loglik[itn] = logdensity.mgauss(Y.reshape(-1, 1), np.zeros((n_samples, 1)), sigmaY)
        Sigmab = s2 * Vinv  # posterior variance of b
        mub = np.dot(Vinv, XTY) # posterior mean of b
        b2m = np.einsum('i,j->ij', mub, mub) + Sigmab
        s2 = (YTY + np.dot(XTX, b2m).trace() - 2 * np.dot(XTY, mub)) / n_samples
        sb2 = np.sum(np.square(mub) + np.diag(Sigmab)) / n_features
        itn += 1
    return s2, sb2, mub, Sigmab, loglik, itn
