import numpy as np
from scipy import linalg as sc_linalg

# invert a square matrix using Cholesky
def cho_inverse(A):
    return sc_linalg.cho_solve(sc_linalg.cho_factor(A, lower=True), np.eye(A.shape[0]))

'''
Invert a pxp matrix directly (X'X + W.inv / sb2)
Require: XTX, Wbar, sb2
'''
def direct(XTX, Wbar, sb2):
    sigma = cho_inverse(XTX + np.diag(1 / Wbar / sb2))
    return sigma

def direct_diag(XTX, Wbar, sb2):
    return np.diag(direct(XTX, Wbar, sb2))


'''
Invert a nxn matrix, using Woodbury identity of (X'X + W.inv / sb2)
Require: X, Wbar, sb2
'''
def woodbury(X, Wbar, sb2):
    XWXT = np.linalg.multi_dot([X, np.diag(Wbar), X.T])
    Hinv = cho_inverse(np.eye(X.shape[0]) + sb2 * XWXT)
    sigma = np.diag(Wbar) * sb2 \
            - np.linalg.multi_dot([np.diag(Wbar), X.T, Hinv, X, np.diag(Wbar)]) \
            * np.square(sb2)
    return sigma

def woodbury_diag(X, Wbar, sb2):
    return np.diag(woodbury(X, Wbar, sb2))


'''
Use the already computed SVD of X to approximate X'X = L'L + D
and then invert the matrix. 
Reduces the computation cost of the diagonal elements of the inverse.
Require: svd(X), Dinit = diag(X'X), Wbar, sb2
'''
def woodbury_svd(svdX, Dinit, Wbar, sb2, k = None):
    U, S, Vh = svdX
    if k is None:
        k = max(S.shape[0], Vh.shape[0])
    L = np.dot(np.diag(S[:k]), Vh[:k, :])
    D = Dinit - np.sum(np.square(L), axis = 0)
    A = D + (1 / Wbar / sb2)
    Ainv = np.diag(1 / A)
    Hinv = cho_inverse(np.eye(k) + np.linalg.multi_dot([L, Ainv, L.T]))
    sigma = Ainv - np.linalg.multi_dot([Ainv, L.T, Hinv, L, Ainv])
    return sigma

def woodbury_svd_diag(svdX, Dinit, Wbar, sb2, k = None):
    return np.diag(woodbury_svd(svdX, Dinit, Wbar, sb2, k = k))
