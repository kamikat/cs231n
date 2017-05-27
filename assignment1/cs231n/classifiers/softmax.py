import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    f = X[i].dot(W)
    f -= np.max(f)
    s = np.sum(np.exp(f))
    loss += - f[y[i]] + np.log(s)
    dW += (np.tile(X[i],(W.shape[1], 1)) * np.exp(f).reshape((-1,1)) / s).T
    dW[:,y[i]] -= X[i]
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)
  dW /= X.shape[0]
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  F = X.dot(W) # (N, C)
  F -= np.max(F, axis=1).reshape((-1,1))
  S = np.sum(np.exp(F), axis=1).reshape((-1,1)) # (N, 1)
  loss = np.average(np.log(S) - F[np.arange(X.shape[0]),y].reshape((-1,1))) # np.average((N,1) - (N,1))
  loss += 0.5 * reg * np.sum(W * W)
  dW += np.dot(X.T, np.exp(F) / S)
  proj_y = np.zeros((y.shape[0], W.shape[1]))
  proj_y[np.arange(y.shape[0]),y] = 1
  dW -= np.dot(X.T, proj_y)
  dW /= X.shape[0]
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

