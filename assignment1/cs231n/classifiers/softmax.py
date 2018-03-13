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
	N, D = X.shape
	D, C = W.shape
	
	for i in range(N):
		sum = 0
		f = X[i].dot(W)
		f -= np.max(f)
		f = np.exp(f)
		sum = np.sum(f)
		for j in range(C):
			if y[i] == j:
				dW[:,j] -= X[i]
			dW[:,j] += (f[j]/sum)*X[i]
			pass
		loss += -np.log(f[y[i]]/sum)
	
	loss /= N
	dW /= N
	loss += reg*np.sum(W*W)
	dW += 2*reg*W
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

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using no explicit loops.  #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	N,D = X.shape
	D,C = W.shape
	f = np.dot(X,W)
	f -= np.max(f,axis=1).reshape((N,1))
	sum = np.sum(np.exp(f),axis=1)
	loss = np.sum(-f[np.arange(N),y] + np.log(sum))
	
	f = np.exp(f)
	mask = np.zeros((N,C))
	mask = f/sum[:,None]
	mask[np.arange(N),y] -= 1 
	dW = np.dot(X.T,mask)
	
	loss /= N
	loss += reg*np.sum(W*W)
	dW /= N
	dW += 2*reg*W 
	pass
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	return loss, dW

