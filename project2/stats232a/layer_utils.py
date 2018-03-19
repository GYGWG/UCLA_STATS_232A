pass
from stats232a.layers import *


def fc_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    out, cache = None, None
    
    ###########################################################################
    # TODO: Implement fc-relu forward pass.                                   #
    ###########################################################################
    fc_out, fc_cache = fc_forward(x, w, b)
    out, relu_cache = relu_forward(fc_out)

    cache = [fc_cache, relu_cache]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return out, cache


def fc_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    dx, dw, db = None, None, None
    
    ###########################################################################
    # TODO: Implement the fc-relu backward pass.                              #
    ###########################################################################
    dx, dw, db = fc_backward(relu_backward(dout, cache[1]), cache[0])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return dx, dw, db


def fc_BN_relu_forward(x, w, b, gamma, beta, bn_params):
    """
    Convenience layer that perorms an affine transform followed by a ReLU
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    out, fc_cache = fc_forward(x, w, b)
    out, bn_cache = batchnorm_forward(out, gamma, beta, bn_params)
    out, relu_cache = relu_forward(out)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def fc_BN_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    dx = relu_backward(dout, relu_cache)
    dx, dgamma, dbeta = batchnorm_backward(dx, bn_cache)
    dx, dw, db = fc_backward(dx, fc_cache)
    return dx, dw, db, dgamma, dbeta