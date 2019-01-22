# ambyer
# 2017.09.18

import numpy as np
import theano
import theano.tensor as T

def ortho_weight(n, dim):
    """
    >>
    > n : how many parameters in a var W, U or V
    > dim : dim of the ortho matrix
    < return : the ortho matrix
    """
    W = np.random.randn(n, dim, dim).astype(theano.config.floatX)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

def getWeight(n, idim, odim = None, ortho = True):
    """
    >>
    > n : how many parameters in a var W, U or V
    > idim : input dim
    > odim : output dim
    > ortho : will return ortho matrix or random matrix
    < return : the init param matrix
    """
    if odim == None:
        odim = idim
    if ortho and idim == odim:
        W = ortho_weight(n, idim)
    else:
        W = np.random.randn(n, idim, odim).astype(theano.config.floatX)

    return W.astype('float32')
