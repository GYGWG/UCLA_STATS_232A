from builtins import range
from builtins import object
import numpy as np

from stats232a.layers import *
from stats232a.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be input - fc - relu - fc - softmax.
    
    The outputs of the second fully-connected layer are the scores for each class.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=1*28*28, hidden_dim=100, num_classes=10, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Following  #
        # the instruction on class to initialize weights carefully, and biases     #
        # should be initialized to zero. All weights and biases should be stored   #
        # in the dictionary self.params, with first layer weights and biases using #
        # the keys 'W1' and 'b1' and second layer weights and biases using the     #
        # keys 'W2' and 'b2'.                                                      # 
        ############################################################################
        dim = [[input_dim, hidden_dim], [hidden_dim, num_classes]]
        for i in range(2):
            self.params["W{}".format(i+1)] = np.random.randn(*(dim[i])) / np.sqrt(dim[i][0])
            self.params["b{}".format(i+1)] = np.zeros(dim[i][1])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        y1, y1_cache = fc_relu_forward(X, self.params["W1"], self.params["b1"])
        scores, y2_cache = fc_forward(y1, self.params["W2"], self.params["b2"])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dLdz2 = softmax_loss(scores, y)
        loss += (np.linalg.norm(self.params["W1"]) ** 2 + np.linalg.norm(self.params["W2"]) ** 2) * 0.5 * self.reg

        dLdy1, grads["W2"], grads["b2"] = fc_backward(dLdz2, y2_cache)
        dLdx, grads["W1"], grads["b1"] = fc_relu_backward(dLdy1, y1_cache)

        grads["W1"] += self.reg * self.params["W1"]
        grads["W2"] += self.reg * self.params["W2"]


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {fc - [batch norm] - relu - [dropout]} x (L - 1) - fc - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=1*28*28, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        dim = np.hstack((input_dim, hidden_dims, num_classes))

        if weight_scale:
            for i in range(self.num_layers):
                self.params["W{}".format(i + 1)] = np.random.normal(0, weight_scale, dim[i:i+2])
                self.params["b{}".format(i + 1)] = np.zeros(dim[i+1])
                if use_batchnorm and i < self.num_layers - 1:
                    self.params["gamma{}".format(i + 1)] = np.random.randn(dim[i + 1]) * 1e-3 + 1
                    self.params["beta{}".format(i + 1)] = np.zeros(dim[i + 1])

        else:
            for i in range(self.num_layers):
                self.params["W{}".format(i + 1)] = np.random.randn(*(dim[i:i+2])) * np.sqrt(2/(dim[i]+dim[i+1]))
                self.params["b{}".format(i + 1)] = np.zeros(dim[i+1])
                if use_batchnorm and i < self.num_layers - 1:
                    self.params["gamma{}".format(i + 1)] = np.random.randn(dim[i + 1]) * 1e-3 + 1
                    self.params["beta{}".format(i + 1)] = np.zeros(dim[i + 1])

        # # First layer
        # self.params['W1'] = np.random.normal(0, weight_scale, [input_dim, hidden_dims[0]])
        # self.params['b1'] = np.zeros(hidden_dims[0])
        # if self.use_batchnorm:
        #     self.params['gamma1'] = np.random.normal(1, 1e-3, hidden_dims[0])
        #     self.params['beta1'] = np.zeros(hidden_dims[0])
        #
        # for layer in range(self.num_layers - 2):
        #     Wi = 'W' + str(layer + 2)
        #     bi = 'b' + str(layer + 2)
        #     yi = 'gamma' + str(layer + 2)
        #     betai = 'beta' + str(layer + 2)
        #     self.params[Wi] = np.random.normal(0, weight_scale, [hidden_dims[layer], hidden_dims[layer + 1]])
        #     self.params[bi] = np.zeros(hidden_dims[layer + 1])
        #     if self.use_batchnorm:
        #         self.params[yi] = np.random.normal(1, 1e-3, hidden_dims[layer + 1])
        #         self.params[betai] = np.zeros(hidden_dims[layer + 1])
        #
        # # Last layer
        # Wi = 'W' + str(self.num_layers)
        # bi = 'b' + str(self.num_layers)
        #
        # self.params[Wi] = np.random.normal(0, weight_scale, [hidden_dims[-1], num_classes])
        # self.params[bi] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        cache = []
        yi = X

        if self.use_batchnorm:
            for i in range(self.num_layers - 1):
                # fc_out, fc_cache = fc_forward(yi, self.params["W{}".format(i + 1)], self.params["b{}".format(i + 1)])
                # BN_out, BN_cache = batchnorm_forward(fc_out, self.params["gamma{}".format(i + 1)],
                #                                      self.params["beta{}".format(i + 1)], self.bn_params[i])
                # yi, relu_cache = relu_forward(BN_out)
                # cache.append({'fc_cache': fc_cache, 'BN_cache': BN_cache, 'relu_cache': relu_cache})

                yi, caches = fc_BN_relu_forward(yi, self.params["W{}".format(i + 1)], self.params["b{}".format(i + 1)],
                                                self.params["gamma{}".format(i + 1)], self.params["beta{}".format(i + 1)]
                                                , self.bn_params[i])
                cache.append(caches)

        else:
            for i in range(1, self.num_layers):
                yi, caches = fc_relu_forward(yi, self.params["W{}".format(i)], self.params["b{}".format(i)])
                cache.append({'fc_cache': caches[0], 'relu_cache': caches[1]})

        # Last layer
        scores, fc_cache = fc_forward(yi, self.params["W{}".format(self.num_layers)],
                                      self.params["b{}".format(self.num_layers)])
        cache.append({'fc_cache': fc_cache})

        # TODO: add dropout
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dLdz = softmax_loss(scores, y)
        loss += np.sum([np.linalg.norm(self.params["W{}".format(i+1)]) ** 2 for i in range(self.num_layers)]) \
                * 0.5 * self.reg
        # Last layer
        dLdy, grads["W{}".format(self.num_layers)], grads["b{}".format(self.num_layers)]\
            = fc_backward(dLdz, cache[-1]['fc_cache'])
        grads["W{}".format(self.num_layers)] += self.reg * self.params["W{}".format(self.num_layers)]

        # L-1 layers of [FC]->[BN]->[ReLU]
        if self.use_batchnorm:
            for i in range(self.num_layers - 1, 0, -1):
                # dLdz_norm = relu_backward(dLdy, cache[i-1]['relu_cache'])
                # dLdz, grads["gamma{}".format(i)], grads["beta{}".format(i)] \
                #     = batchnorm_backward(dLdz_norm, cache[i-1]['BN_cache'])
                # dLdy, grads["W{}".format(i)], grads["b{}".format(i)] = fc_backward(dLdz, cache[i-1]['fc_cache'])
                dLdy, grads["W{}".format(i)], grads["b{}".format(i)], \
                grads["gamma{}".format(i)], grads["beta{}".format(i)] = fc_BN_relu_backward(dLdy, cache[i-1])

                grads["W{}".format(i)] += self.reg * self.params["W{}".format(i)]

        else:
            for i in range(self.num_layers - 1, 0, -1):
                dLdy, grads["W{}".format(i)], grads["b{}".format(i)] \
                    = fc_relu_backward(dLdy, [cache[i-1]['fc_cache'], cache[i-1]['relu_cache']])
                grads["W{}".format(i)] += self.reg * self.params["W{}".format(i)]

        grads['dx'] = dLdy
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads


class FullyConnectedNetAns(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        # First layer
        self.params['W1'] = np.random.normal(0, weight_scale, [input_dim, hidden_dims[0]])
        self.params['b1'] = np.zeros(hidden_dims[0])
        if self.use_batchnorm:
            self.params['y1'] = np.random.normal(1, 1e-3, hidden_dims[0])
            self.params['beta1'] = np.zeros(hidden_dims[0])

        for layer in range(self.num_layers - 2):
            Wi = 'W' + str(layer + 2)
            bi = 'b' + str(layer + 2)
            yi = 'y' + str(layer + 2)
            betai = 'beta' + str(layer + 2)
            self.params[Wi] = np.random.normal(0, weight_scale, [hidden_dims[layer], hidden_dims[layer + 1]])
            self.params[bi] = np.zeros(hidden_dims[layer + 1])
            if self.use_batchnorm:
                self.params[yi] = np.random.normal(1, 1e-3, hidden_dims[layer + 1])
                self.params[betai] = np.zeros(hidden_dims[layer + 1])

        # Last layer
        Wi = 'W' + str(self.num_layers)
        bi = 'b' + str(self.num_layers)

        self.params[Wi] = np.random.normal(0, weight_scale, [hidden_dims[-1], num_classes])
        self.params[bi] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        else:
            self.dropout_param = {'mode': 'train', 'p': 0}
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        input_data = X
        output = {}
        for layer in range(self.num_layers - 1):
            Wi = 'W' + str(layer + 1)
            bi = 'b' + str(layer + 1)
            yi = 'y' + str(layer + 1)
            betai = 'beta' + str(layer + 1)
            if self.use_batchnorm:
                out, cache = fc_BN_relu_forward(input_data, self.params[Wi], self.params[bi],
                                                           self.params[yi], self.params[betai], self.bn_params[layer])

            input_data = out
            output[layer] = {'out_data': out, 'cache': cache}
        # Final Layer
        layer += 1
        Wi = 'W' + str(self.num_layers)
        bi = 'b' + str(self.num_layers)
        out, aff_cache = fc_forward(input_data, self.params[Wi], self.params[bi])
        output[self.num_layers - 1] = {'out_data': out, 'cache': aff_cache}

        scores = out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores
        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # Calculating loss
        sftm_loss, sftm_grad = softmax_loss(scores, y)
        loss += sftm_loss

        # Adding regularizatoin loss
        for layer in range(self.num_layers):
            Wi = 'W' + str(layer + 1)
            loss += .5 * self.reg * np.sum(self.params[Wi] * self.params[Wi])

        # Calculating gradient
        Wi = 'W' + str(self.num_layers)
        bi = 'b' + str(self.num_layers)
        dx, grads[Wi], grads[bi] = fc_backward(sftm_grad, aff_cache)
        grads[Wi] += self.reg * self.params[Wi]

        for layer in range(self.num_layers - 1)[::-1]:
            Wi = 'W' + str(layer + 1)
            bi = 'b' + str(layer + 1)
            yi = 'y' + str(layer + 1)
            betai = 'beta' + str(layer + 1)

            # Storing the gradients
            if self.use_batchnorm:
                dx, grads[Wi], grads[bi], grads[yi], grads[betai] = fc_BN_relu_backward(dx, output[layer][
                    'cache'])
            # Loss due to the regularazitation parameter.
            grads[Wi] += self.reg * self.params[Wi]

        grads['dx'] = dx
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
