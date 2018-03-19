from builtins import range
import numpy as np


def fc_forward(x, w, b):
    out = x.reshape(x.shape[0], w.shape[0]).dot(w) + b
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    x, w, b = cache
    dw = x.reshape(x.shape[0], w.shape[0]).T.dot(dout)
    db = np.sum(dout, axis=0)
    dx = dout.dot(w.T).reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout
    dx[cache < 0] = 0
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    Ho = 1 + (H + 2 * pad - HH) // stride
    Wo = 1 + (H + 2 * pad - WW) // stride
    out = np.zeros((N, F, Ho, Wo))

    x = np.pad(x, [(0,), (0,), (pad,), (pad,)], mode='constant')

    for i1 in range(N):
        for i2 in range(F):
            for i3 in range(Ho):
                for i4 in range(Wo):
                    out[i1, i2, i3, i4] = \
                        np.sum(x[i1, :, i3*stride:i3*stride+HH, i4*stride:i4*stride+WW] * w[i2,:,:,:]) + b[i2]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    Ho = 1 + (H - HH) // stride
    Wo = 1 + (H - WW) // stride

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0,2,3))
    for i1 in range(N):
        for i2 in range(F):
            for i3 in range(Ho):
                for i4 in range(Wo):
                    dx[i1, :, i3*stride:i3*stride+HH, i4*stride:i4*stride+WW] += w[i2,:,:,:] * dout[i1,i2,i3,i4]
                    dw[i2,:,:,:] += x[i1, :, i3*stride:i3*stride+HH, i4*stride:i4*stride+WW] * dout[i1,i2,i3,i4]
    dx = dx[:,:, 1:-1, 1:-1]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    Hp = pool_param['pool_height']
    Wp = pool_param['pool_width']
    stride = pool_param['stride']
    Ho = 1 + (H - Hp) // stride
    Wo = 1 + (W - Wp) // stride
    out = np.zeros( (N, C, Ho, Wo) )

    for i1 in range(N):
        for i2 in range(C):
            for i3 in range(Ho):
                for i4 in range(Wo):
                    out[i1, i2, i3, i4] = np.max(x[i1, i2, i3*stride:i3*stride+Hp, i4*stride:i4*stride+Wp])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    dx, pool_param = cache
    N, C, H, W = dx.shape
    Hp = pool_param['pool_height']
    Wp = pool_param['pool_width']
    stride = pool_param['stride']
    Ho = 1 + (H - Hp) // stride
    Wo = 1 + (W - Wp) // stride

    for i1 in range(N):
        for i2 in range(C):
            for i3 in range(Ho):
                for i4 in range(Wo):
                    patch = np.copy(dx[i1, i2, i3*stride:i3*stride+Hp, i4*stride:i4*stride+Wp])
                    dx[i1, i2, i3*stride:i3*stride+Hp, i4*stride:i4*stride+Wp][patch < np.max(patch)] = 0
                    dx[i1, i2, i3 * stride:i3 * stride + Hp, i4 * stride:i4 * stride + Wp][patch == np.max(patch)] = \
                        dout[i1, i2, i3, i4]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
