import numpy as np

def parametric(X, y, s2, sb2, mu, sigma, Wbar, XTX, ElogW, KLw):
    n, p = X.shape
    elbo = c_func(n, s2, ElogW) \
           + h1_func(X, y, s2, mu, Wbar) \
           + h2_func(XTX, sigma, Wbar) \
           + KLw
    return elbo


def c_func(n, s2, ElogW):
    val  = - 0.5 * n * np.log(2. * np.pi * s2)
    val += - 0.5 * np.sum(ElogW)
    return val

def h1_func(X, y, s2, mu, Wbar):
    val = - (0.5 / s2) * (np.sum(np.square(y - np.dot(X, mu))) \
                          + np.sum(np.square(mu) / Wbar))
    return val

def h2_func(XTX, sigma, Wbar):
    (sign, logdet) = np.linalg.slogdet(sigma)
    val = - 0.5 * np.trace(np.dot(XTX + np.diag(1 / Wbar), sigma)) + 0.5 * logdet
    return val
