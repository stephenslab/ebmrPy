import numpy as np

def parametric(X, y, s2, sb2, mu, sigma, Wbar, XTX, ElogdetW, KLW, 
        h2_term=0, calc_h2=True):
    n, p = X.shape
    if calc_h2:
        h2_term = h2_func(p, sb2, XTX, sigma, Wbar)
    #c_term = c_func(n, p, s2, sb2, ElogdetW)
    #h1_term = h1_func(X, y, s2, sb2, mu, Wbar)
    #print(f"ELBO terms: {c_term:.3f}\t{h1_term:.3f}\t{h2_term:.3f}")
    elbo = c_func(n, p, s2, sb2, ElogdetW) \
           + h1_func(X, y, s2, sb2, mu, Wbar) \
           + h2_term \
           + KLW
    return elbo


def c_func(n, p, s2, sb2, ElogdetW):
    val  = 0.5 * p - 0.5 * p * np.log(sb2)
    val += - 0.5 * n * np.log(2. * np.pi * s2)
    val += - 0.5 * ElogdetW
    #term1  = 0.5 * p - 0.5 * p * np.log(sb2)
    #term2  = - 0.5 * n * np.log(2. * np.pi * s2)
    #term3  = - 0.5 * ElogdetW
    #print(f"c-terms: {term1:.3f} {term2:.3f} {term3:.3f}")
    return val

def h1_func(X, y, s2, sb2, mu, Wbar):
    val = - (0.5 / s2) * (np.sum(np.square(y - np.dot(X, mu))) \
                          + np.sum(np.square(mu) / Wbar / sb2))
    return val

def h2_func(p, sb2, XTX, sigma, Wbar):
    (sign, logdet) = np.linalg.slogdet(sigma)
    val = - 0.5 * np.trace(np.dot(XTX + np.diag(1 / Wbar / sb2), sigma)) + 0.5 * logdet
    #val = - 0.5 * p + 0.5 * logdet
    return val
