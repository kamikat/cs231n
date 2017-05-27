import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MyCifar10Model(object):
  """
  A convolutional network with the following architecture:
  
  [conv - BN - relu - 2x2 max pool]x3 - affine - BN - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=(64,32,32), filter_size=(3,3,3),
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    self.bn_param1 = {'mode': 'train'}
    self.bn_param2 = {'mode': 'train'}
    self.bn_param3 = {'mode': 'train'}
    self.bn_param4 = {'mode': 'train'}
    
    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(num_filters[0], C, filter_size[0], filter_size[0])
    self.params['W2'] = weight_scale * np.random.randn(num_filters[1], num_filters[0], filter_size[1], filter_size[1])
    self.params['W3'] = weight_scale * np.random.randn(num_filters[2], num_filters[1], filter_size[2], filter_size[2])
    self.params['W4'] = weight_scale * np.random.randn(num_filters[2]*(H/8)*(W/8), hidden_dim)
    self.params['W5'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b1'] = np.zeros((num_filters[0],))
    self.params['b2'] = np.zeros((num_filters[1],))
    self.params['b3'] = np.zeros((num_filters[2],))
    self.params['b4'] = np.zeros((hidden_dim,))
    self.params['b5'] = np.zeros((num_classes,))
    self.params['gamma1'], self.params['beta1'] = np.ones((num_filters[0],)), np.zeros((num_filters[0],))
    self.params['gamma2'], self.params['beta2'] = np.ones((num_filters[1],)), np.zeros((num_filters[1],))
    self.params['gamma3'], self.params['beta3'] = np.ones((num_filters[2],)), np.zeros((num_filters[2],))
    self.params['gamma4'], self.params['beta4'] = np.ones((hidden_dim,)), np.zeros((hidden_dim,))

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    
    mode = 'test' if y is None else 'train'
    self.bn_param1['mode'], self.bn_param2['mode'] = mode, mode
    self.bn_param3['mode'], self.bn_param4['mode'] = mode, mode
    
    W1, b1, gamma1, beta1 = self.params['W1'], self.params['b1'], self.params['gamma1'], self.params['beta1']
    W2, b2, gamma2, beta2 = self.params['W2'], self.params['b2'], self.params['gamma2'], self.params['beta2']
    W3, b3, gamma3, beta3 = self.params['W3'], self.params['b3'], self.params['gamma3'], self.params['beta3']
    W4, b4, gamma4, beta4 = self.params['W4'], self.params['b4'], self.params['gamma4'], self.params['beta4']
    W5, b5 = self.params['W5'], self.params['b5']
    
    # pass conv_param to the forward pass for the convolutional layer
    conv_param1 = {'stride': 1, 'pad': (W1.shape[2] - 1) / 2}
    conv_param2 = {'stride': 1, 'pad': (W2.shape[2] - 1) / 2}
    conv_param3 = {'stride': 1, 'pad': (W3.shape[2] - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = X
    scores, cache1 = conv_bn_relu_pool_forward(scores, W1, b1, gamma1, beta1, conv_param1, self.bn_param1, pool_param)
    scores, cache2 = conv_bn_relu_pool_forward(scores, W2, b2, gamma2, beta2, conv_param2, self.bn_param2, pool_param)
    scores, cache3 = conv_bn_relu_pool_forward(scores, W3, b3, gamma3, beta3, conv_param3, self.bn_param3, pool_param)
    scores, cache4 = affine_bn_relu_forward(scores, W4, b4, gamma4, beta4, self.bn_param4)
    scores, cache5 = affine_forward(scores, W5, b5)
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    
    loss, ds = softmax_loss(scores, y)
    ds, dW5, db5 = affine_backward(ds, cache5)
    ds, dW4, db4, dgamma4, dbeta4 = affine_bn_relu_backward(ds, cache4)
    ds, dW3, db3, dgamma3, dbeta3 = conv_bn_relu_pool_backward(ds, cache3)
    ds, dW2, db2, dgamma2, dbeta2 = conv_bn_relu_pool_backward(ds, cache2)
    dx, dW1, db1, dgamma1, dbeta1 = conv_bn_relu_pool_backward(ds, cache1)
    loss += 0.5 * self.reg * sum([np.sum(w*w) for w in [W1, W2, W3, W4, W5, b1, b2, b3, b4, b5, gamma1, gamma2, gamma3, gamma4, beta1, beta2, beta3, beta4]])
    grads = {
      'W1': dW1 + self.reg * np.sum(W1),
      'W2': dW2 + self.reg * np.sum(W2),
      'W3': dW3 + self.reg * np.sum(W3),
      'W4': dW4 + self.reg * np.sum(W4),
      'W5': dW5 + self.reg * np.sum(W5),
      'b1': db1 + self.reg * np.sum(b1),
      'b2': db2 + self.reg * np.sum(b2),
      'b3': db3 + self.reg * np.sum(b3),
      'b4': db4 + self.reg * np.sum(b4),
      'b5': db5 + self.reg * np.sum(b5),
      'gamma1': dgamma1 + self.reg * np.sum(gamma1),
      'gamma2': dgamma2 + self.reg * np.sum(gamma2),
      'gamma3': dgamma3 + self.reg * np.sum(gamma3),
      'gamma4': dgamma4 + self.reg * np.sum(gamma4),
      'beta1': beta1 + self.reg * np.sum(beta1),
      'beta2': beta2 + self.reg * np.sum(beta2),
      'beta3': beta3 + self.reg * np.sum(beta3),
      'beta4': beta4 + self.reg * np.sum(beta4)}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
