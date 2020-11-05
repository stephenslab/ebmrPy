import numpy as np

def mgauss(x, u, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = - nsample * 0.5 * np.log(2. * np.pi) - 0.5 * np.linalg.slogdet(cov)[1]
    xlm = x - mu
    part2 =  - 0.5 * np.dot(xlm.T, np.dot(np.linalg.inv(cov), xlm))
    return float(part1 + part2)
