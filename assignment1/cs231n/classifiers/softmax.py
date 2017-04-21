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
  nums = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(nums):
    cur_sample = X[i]
    score_class = []
    for j in xrange(num_class):
      score_class.append(np.sum(cur_sample*W[:,j]))
    score_class = np.array(score_class)
    max_score = np.max(score_class)
    score_shifted = score_class - max_score
    score_exp = np.exp(score_shifted)
    score_norm = score_exp/np.sum(score_exp)
    loss -= np.log(score_norm[y[i]])
    for j in xrange(num_class):
      if j!=y[i]:
        dW[:,j] += score_norm[j]*cur_sample
      else:
        dW[:,y[i]] += (score_norm[y[i]]-1)*cur_sample
  loss /= nums
  loss += 0.5*reg*np.sum(W*W)
  dW /= nums
  dW += reg*W
  pass
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
  nums = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  max_score = np.max(scores,axis=1).reshape(-1,1)
  score_shifted = scores - max_score
  score_exp = np.exp(score_shifted)
  score_exp_sum = np.sum(score_exp,axis=1).reshape(-1,1)
  score_norm = score_exp/score_exp_sum
  score_class = score_norm[range(nums),y]
  loss = -np.sum(np.log(score_class))
  loss /= nums
  loss += reg*0.5*np.sum(W*W)
  score_norm[range(nums),y] = score_class-1
  dW = X.T.dot(score_norm)
  dW /= nums
  dW += reg*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

