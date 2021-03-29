import numpy as np
from scipy import linalg as sc_linalg

# return inverse and determinant of a square matrix using Cholesky
def cho_inverse_logdet(A):
    L = sc_linalg.cho_factor(A, lower=True)
    Ainv = sc_linalg.cho_solve(L, np.eye(A.shape[0]))
    Ainv_logdet = - 2 * np.sum(np.log(np.diag(L[0])))
    return Ainv, Ainv_logdet


# remove this later, kept for backwards compatibility
def cho_inverse(A):
    return cho_inverse_logdet(A)[0]

'''
Invert a pxp matrix directly (X'X + W.inv / sb2)
Require: XTX, Wbar, sb2
'''
def direct(XTX, Wbar, sb2, compute_full=False):
    sigma, logdet = cho_inverse_logdet(XTX + np.diag(1 / Wbar / sb2))
    if compute_full:
        return sigma, logdet
    else:
        return np.diag(sigma), logdet


'''
Invert a nxn matrix, using Woodbury identity of (X'X + W.inv / sb2)
Require: X, Wbar, sb2

This function is just for checking the Woodbury,
and is computationally costlier than the direct approach
(calculates inverse of nxn matrix, and calculates determinant separately)
'''
def woodbury(X, Wbar, sb2, compute_full=False):
    XWXT  = np.linalg.multi_dot([X, np.diag(Wbar), X.T])
    Hinv  = cho_inverse(np.eye(X.shape[0]) + sb2 * XWXT)
    Wfull = np.diag(Wbar)
    sigma = Wfull * sb2 - np.linalg.multi_dot([Wfull, X.T, Hinv, X, Wfull]) * sb2 * sb2
    # !!! WARNING: I assumed the sign is positive 
    __sign, logdet = np.linalg.slogdet(sigma)
    if compute_full:
        return sigma, logdet
    else:
        return np.diag(sigma), logdet


'''
Use the already computed SVD of X to approximate X'X = L'L + D
and then invert the matrix. 
Reduces the computation cost of the diagonal elements of the inverse.
Require: svd(X), Dinit = diag(X'X), Wbar, sb2
'''
def woodbury_svdX(svdX, Dinit, Wbar, sb2, k = None, compute_full=False):
    U, S, Vh = svdX
    p = Wbar.shape[0]
    if k is None:
        k = min(S.shape[0], Vh.shape[0])
    L = np.dot(np.diag(S[:k]), Vh[:k, :])
    D = Dinit - np.sum(np.square(L), axis = 0)
    Wscale = 1 / (D + (1 / Wbar / sb2))
    Wtilde = np.diag(np.sqrt(Wscale))
    Ltilde = np.dot(L, Wtilde)

    H = np.eye(k) + np.dot(Ltilde, Ltilde.T)
    Hinv, Hinv_logdet = cho_inverse_logdet(H)
    H_logdet = - Hinv_logdet

    A_inner = np.eye(p) - np.linalg.multi_dot([Ltilde.T, Hinv, Ltilde])
    if compute_full:
        sigma = np.linalg.multi_dot([Wtilde, A_inner, Wtilde])
        sign, logdet = np.linalg.slogdet(sigma)
        return sigma, logdet
    else:
        sigma_diag = Wscale * np.diag(A_inner)
        logdet = np.sum(np.log(Wscale)) - H_logdet
        return sigma_diag, logdet


'''
Use the already computed SVD of XW^0.5
Reduces the computation cost of the diagonal elements of the inverse.
Require: n_features, svd(XW^0.5), Wbar, sb2
This function does not return logdet(Sigma)
'''
def woodbury_svdXW(p, svdXW, Wbar, sb2, compute_full=False):
    U, D, Vh = svdXW
    d2 = np.square(D)
    dt = sb2 * d2 / (1 + sb2 * d2)
    vdtv = np.eye(p) - np.linalg.multi_dot([Vh.T, np.diag(dt), Vh])
    Wsqrt_m = np.diag(np.sqrt(Wbar)) # convert the Wbar vector to matrix format
    logdet = np.sum(np.log(sb2 * Wbar)) - np.sum(np.log(1 + sb2 * d2))
    if compute_full:
        sigma = sb2 * np.linalg.multi_dot([Wsqrt_m, vdtv, Wsqrt_m])
        return sigma, logdet
    else:
        sigma_diag = sb2 * Wbar * (1 - np.einsum('ij,ij,i->j', Vh, Vh, dt))
        return sigma_diag, logdet
