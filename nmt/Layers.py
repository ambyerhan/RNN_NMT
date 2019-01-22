# rebuild project
# ambyer
# 2017.06.28 - 2017.06.29

# this file contains single layer class that used in Neural Network

import theano
import theano.tensor as T
import numpy as np
import numpy.random as rnd

from datetime import datetime
from utils import getThreshold
from Initializer import getWeight
from ErrorMsg import CErrorMsg

class CEmbeddingLayer(object):
	""" embedding layer class """
	def __init__(self, edim, vsize, name):
		"""
		>>
		> edim : the embedding size
		> vsize : the vocabulary size
		> name : name of the layer, this especially useful when save or read models
		< return :
		"""
		self.edim = edim
		self.vsize = vsize
		self.name = name
		self.threshold = getThreshold(vsize, edim)

		vocab = self.threshold * rnd.randn(vsize, edim).astype(theano.config.floatX)
		self.voc = theano.shared(value = vocab, name = self.name, borrow = True)
		
		self.params = [self.voc]

	def activate(self, inputs):
		"""
		>>
		> inputs : the inputs
		< return : return a 3D matrix that extended from inputs matrix, shape is <sentlen, minibatch, edim>
		"""
		outputs = self.voc[inputs]
		return outputs

	def resetBatch(self, newbatch):
		"""
		>>
		> newbatch : the new batch
		< return : 
		"""
		self.minibatch = newbatch

	def saveModel(self, dirname):
		"""
		>>
		> e_n : the num of the epoch
		> dirname : the directory name
		< return : 
		"""
		filename = dirname + self.name + ".npz"

		print '    >>[Debug] Saving the model %s.' % filename
		np.savez(filename, voc = self.voc.get_value())
		print '    >>[Debug] Model is saved...'

	def readModel(self, filename):
		"""
		>>
		> filename : the filename that contains the model
		< return :
		"""
		filename = filename + self.name + ".npz"

		print '    >>[Debug] Reading the model from file : %s' % filename
		model = np.load(filename)
		self.voc.set_value(model['voc'])
		print '    >>[Debug] Model is read...'

class CNNLayer(object):
	""" commen neural network layer class """
	def __init__(self, idim, odim, name, act_method = "sigmoid"):
		"""
		>>
		> idim : the inputs's dim
		> odim : the output's dim
		> name : name of the layer, this especially useful when save or read models
		> act_method : the activation act_method
		< return :
		"""
		self.idim = idim
		self.odim = odim
		self.name = name
		self.wName = name + '_W'
		self.bName = name + '_b'
		self.act_method = act_method

		W = getThreshold(idim, odim) * rnd.randn(idim, odim).astype(theano.config.floatX)
		b = getThreshold(odim, 0) * rnd.randn(odim).astype(theano.config.floatX)

		self.W = theano.shared(value = W, name = self.wName, borrow = True)
		self.b = theano.shared(value = b, name = self.bName, borrow = True)

		self.params = [self.W, self.b]

	def activate(self, inputs):
		"""
		>>
		> inputs : the inputs
		< return : return the activated value
		"""
		CErrorMsg.showErrExMsg((self.act_method in ["sigmoid", "tanh", "pureline", "ReLU"]), "The activation method is wrong!")
		if self.act_method == "sigmoid": # this is activate funt (not init func), means that this can be called frequently, so we use if-elif-...-else structure
			return T.nnet.sigmoid(T.dot(inputs, self.W) + self.b)
		elif self.act_method == "ReLU":
			return T.nnet.relu(T.dot(inputs, self.W) + self.b)
		elif self.act_method == "tanh":
			return T.tanh(T.dot(inputs, self.W) + self.b)
		elif self.act_method == "pureline":
			return (T.dot(inputs, self.W) + self.b)
		elif self.act_method == "dot":
			return (T.dot(inputs, self.W) + 0 * self.b) # this just do a dot-operate, and no bias
		else:
			pass

	def resetBatch(self, newbatch):
		"""
		>>
		> newbatch : the new batch
		< return : 
		"""
		self.minibatch = newbatch

	def saveModel(self, dirname):
		"""
		>>
		> e_n : the num of the epoch
		> dirname : the directory name
		< return : 
		"""
		filename = dirname + self.name + ".npz"

		print '    >>[Debug] Saving the model %s.' % filename
		np.savez(filename, W = self.W.get_value(),
							b = self.b.get_value())
		print '    >>[Debug] Model is saved...'

	def readModel(self, filename):
		"""
		>>
		> filename : the filename that contains the model
		< return :
		"""
		filename = filename + self.name + ".npz"

		print '    >>[Debug] Reading the model from file : %s' % filename
		model = np.load(filename)
		self.W.set_value(model['W'])
		self.b.set_value(model['b'])
		print '    >>[Debug] Model is read...'

class CRNNLayer(object):
	""" simple RNN neural network layer class """
	def __init__(self, idim, hdim, minibatch, name, isOrtho, bptt_trunc = -1):
		"""
		>>
		> idim : the input dimension
		> hdim : the hidden layer dimension
		> minibatch : the minibatch
		> name : name of the layer, this especially useful when save or read models
		> isOrtho : is the matrix is ortho matrix
		> bptt_trunc : used in scan
		< return :
		"""
		self.idim = idim
		self.hdim = hdim
		self.minibatch = minibatch
		self.name = name
		self.wName = name + '_W'
		self.uName = name + '_U'
		self.bName = name + '_b'
		self.bptt_trunc = bptt_trunc

		#W = getThreshold(idim, hdim) * rnd.randn(idim, hdim).astype(theano.config.floatX)
		#U = getThreshold(hdim, hdim) * rnd.randn(hdim, hdim).astype(theano.config.floatX)
		#b = getThreshold(0, 0) * rnd.randn(hdim).astype(theano.config.floatX)
		W = getThreshold(idim, hdim) * getWeight(3, idim, hdim, isOrtho)
		U = getThreshold(hdim, hdim) * getWeight(3, hdim, hdim, isOrtho)
		b = getThreshold(0, 0) * rnd.randn(3, hdim).astype(theano.config.floatX)

		self.W = theano.shared(value = W, name = self.wName, borrow = True)
		self.U = theano.shared(value = U, name = self.uName, borrow = True)
		self.b = theano.shared(value = b, name = self.bName, borrow = True)

		self.params = [self.W, self.U, self.b]

	def activate(self, x, h_pre, xdp, hdp):
		"""
		>>
		> inputs : the inputs
		< return : the activated value
		"""
		out = T.tanh(T.dot(x * xdp[0], self.W) + T.dot(h_pre * hdp[0], self.U) + self.b)
		return out

	def resetBatch(self, newbatch):
		"""
		>>
		> newbatch : the new batch
		< return : 
		"""
		self.minibatch = newbatch

	def saveModel(self, dirname):
		"""
		>>
		> e_n : the num of the epoch
		> dirname : the directory name
		< return : 
		"""
		filename = dirname + self.name + ".npz"

		print '    >>[Debug] Saving the model %s.' % filename
		np.savez(filename, W = self.W.get_value(),
							U = self.U.get_value(),
							b = self.b.get_value())
		print '    >>[Debug] Model is saved...'

	def readModel(self, filename):
		"""
		>>
		> filename : the filename that contains the model
		< return :
		"""
		filename = filename + self.name + ".npz"

		print '    >>[Debug] Reading the model from file : %s' % filename
		model = np.load(filename)
		self.W.set_value(model['W'])
		self.U.set_value(model['U'])
		self.b.set_value(model['b'])
		print '    >>[Debug] Model is read...'

class CGRULayer(object):
	""" gated recurrent unit layer class """
	def __init__(self, idim, hdim, minibatch, name, isOrtho = True, bptt_trunc = -1):
		"""
		>>
		> idim : the input dimension
		> hdim : the hidden layer dimension
		> minibatch : the minibatch
		> name : name of the layer, this especially useful when save or read models
		> isOrtho : is the matrix is ortho matrix
		> bptt_trunc : used in scan
		< return :
		"""
		self.idim = idim
		self.hdim = hdim
		self.minibatch = minibatch
		self.name = name
		self.wName = name + '_Ws'
		self.uName = name + '_Us'
		self.bName = name + '_bs'
		self.bptt_trunc = bptt_trunc

		#W = getThreshold(idim, hdim) * rnd.randn(3, idim, hdim).astype(theano.config.floatX) # the dim-0 indicate Z-gate, dim-1 indicate R-gate and dim-2 indicate G-gate
		#U = getThreshold(hdim, hdim) * rnd.randn(3, hdim, hdim).astype(theano.config.floatX)
		#b = getThreshold(0, 0) * rnd.randn(3, hdim).astype(theano.config.floatX)
		W = getThreshold(idim, hdim) * getWeight(3, idim, hdim, isOrtho)
		U = getThreshold(hdim, hdim) * getWeight(3, hdim, hdim, isOrtho)
		b = getThreshold(0, 0) * rnd.randn(3, hdim).astype(theano.config.floatX)

		self.W = theano.shared(value = W, name = self.wName, borrow = True)
		self.U = theano.shared(value = U, name = self.uName, borrow = True)
		self.b = theano.shared(value = b, name = self.bName, borrow = True)

		self.params = [self.W, self.U, self.b]

	def activate(self, x, h_pre, xdp, hdp):
		"""
		>>
		> x : the inputs
		> h_pre : last hidden output
		> xdp : x dropout, shaped <3 gate, batch, dim>
		> hdp : h dropout
		< return : the activated value
		"""
		zg = T.nnet.sigmoid(T.dot(x * xdp[0], self.W[0]) + T.dot(h_pre * hdp[0], self.U[0]) + self.b[0])
		rg = T.nnet.sigmoid(T.dot(x * xdp[1], self.W[1]) + T.dot(h_pre * hdp[1], self.U[1]) + self.b[1])
		gg = T.tanh(T.dot(x * xdp[2], self.W[2]) + T.dot((rg * h_pre * hdp[2]), self.U[2]) + self.b[2])
		h = ((1. - zg) * h_pre) + (zg * gg)
		return h

	def resetBatch(self, newbatch):
		"""
		>>
		> newbatch : the new batch
		< return : 
		"""
		self.minibatch = newbatch

	def saveModel(self, dirname):
		"""
		>>
		> e_n : the num of the epoch
		> dirname : the directory name
		< return : 
		"""
		filename = dirname + self.name + ".npz"

		print '    >>[Debug] Saving the model %s.' % filename
		np.savez(filename, W = self.W.get_value(),
							U = self.U.get_value(),
							b = self.b.get_value())
		print '    >>[Debug] Model is saved...'

	def readModel(self, filename):
		"""
		>>
		> filename : the filename that contains the model
		< return :
		"""
		filename = filename + self.name + ".npz"

		print '    >>[Debug] Reading the model from file : %s' % filename
		model = np.load(filename)
		self.W.set_value(model['W'])
		self.U.set_value(model['U'])
		self.b.set_value(model['b'])
		print '    >>[Debug] Model is read...'

class CGRULayerLNWN(object):
	""" gated recurrent unit layer class this GRU is contain layer normalization and weight normalization """
	def __init__(self, idim, hdim, minibatch, name, ln, wn, isOrtho = True, bptt_trunc = -1):
		"""
		>>
		> idim : the input dimension
		> hdim : the hidden layer dimension
		> minibatch : the minibatch
		> name : name of the layer, this especially useful when save or read models
		> ln : bool type, layer normalization or not
		> wn : bool type, weight normalization or not
		> isOrtho : is the matrix is ortho matrix
		> bptt_trunc : used in scan
		< return :
		"""
		self.idim = idim
		self.hdim = hdim
		self.minibatch = minibatch
		self.name = name
		self.wName = name + '_Ws'
		self.uName = name + '_Us'
		self.bName = name + '_bs'
		self.l_norm = ln
		self.w_norm = wn
		self.bptt_trunc = bptt_trunc

		#W = getThreshold(idim, hdim) * rnd.randn(3, idim, hdim).astype(theano.config.floatX) # the dim-0 indicate Z-gate, dim-1 indicate R-gate and dim-2 indicate G-gate
		#U = getThreshold(hdim, hdim) * rnd.randn(3, hdim, hdim).astype(theano.config.floatX)
		#b = getThreshold(0, 0) * rnd.randn(3, hdim).astype(theano.config.floatX)
		W = getThreshold(idim, hdim) * getWeight(3, idim, hdim, isOrtho)
		U = getThreshold(hdim, hdim) * getWeight(3, hdim, hdim, isOrtho)
		b = getThreshold(0, 0) * rnd.randn(3, hdim).astype(theano.config.floatX)

		self.W = theano.shared(value = W, name = self.wName, borrow = True)
		self.U = theano.shared(value = U, name = self.uName, borrow = True)
		self.b = theano.shared(value = b, name = self.bName, borrow = True)

		# layer normalization
		self.waName = name + '_W_al'
		self.wbName = name + '_W_be'
		self.uaName = name + '_U_al'
		self.ubName = name + '_U_be'
		W_al = np.zeros((2, hdim), dtype = theano.config.floatX) # W_alpha
		W_be = np.ones((2, hdim), dtype =theano.config.floatX) # W_beta
		U_al = np.zeros((2, hdim), dtype = theano.config.floatX)
		U_be = np.ones((2, hdim), dtype = theano.config.floatX)

		self.W_al = theano.shared(value = W_al, name = self.waName, borrow = True)
		self.W_be = theano.shared(value = W_be, name = self.wbName, borrow = True)
		self.U_al = theano.shared(value = U_al, name = self.uaName, borrow = True)
		self.U_be = theano.shared(value = U_be, name = self.ubName, borrow = True)

		# TODO >> weight normalization

		self.params = [self.W, self.U, self.b, self.W_al, self.W_be, self.U_al, self.U_be]

	def layer_norm(self, x, b, s):
		"""
		>>layer normalization
		>>code from https://github.com/ryankiros/layer-norm
		"""
		_eps = np.float32(1e-5)
		if x.ndim == 3:
			a = x.var(2)[:, :, None]
			output = (x - x.mean(2)[:, :, None]) / T.sqrt((x.var(2)[:, :, None] + _eps))
			output = s[None, None, :] * output + b[None, None, :]
		else:
			output = (x - x.mean(1)[:, None]) / T.sqrt((x.var(1)[:, None] + _eps))
			output = s[None, :] * output + b[None, :]
		return output

	def weight_norm(self, W, s):
		"""
	    >>w = g/||v|| * v, v is original weight W, g is gain parameter
        >>Normalize the columns of a matrix
        """
		_eps = np.float32(1e-5)
		W_norms = T.sqrt((W * W).sum(axis=0, keepdims=True) + _eps)
		W_norms_s = W_norms * s  # do this first to ensure proper broadcasting
		return W / W_norms_s

	def LN(self, x, b, s):
		"""
		"""
		if self.l_norm:
			return self.layer_norm(x, b, s)
		else:
			return x

	def WN(self, W, s):
		"""
		"""
		if self.w_norm and s is not None:
			return self.weight_norm(W, s)
		else:
			return W

	def activate(self, x, h_pre, xdp, hdp):
		"""
		>>
		> x : the inputs
		> h_pre : last hidden output
		> xdp : x dropout, shaped <3 gate, batch, dim>
		> hdp : h dropout
		< return : the activated value
		"""
		zx = self.LN(T.dot(x * xdp[0]    , self.WN(self.W[0], None)), self.W_al[0], self.W_be[0])
		zh = self.LN(T.dot(h_pre * hdp[0], self.WN(self.U[0], None)), self.U_al[0], self.U_be[0])
		rx = self.LN(T.dot(x * xdp[1]    , self.WN(self.W[1], None)), self.W_al[0], self.W_be[0])
		rh = self.LN(T.dot(h_pre * hdp[1], self.WN(self.U[1], None)), self.U_al[0], self.U_be[0])

		zg = T.nnet.sigmoid(zx + zh + self.b[0])
		rg = T.nnet.sigmoid(rx + rh + self.b[1])

		gx = self.LN(T.dot(x * xdp[2], self.WN(self.W[2], None)), self.W_al[1], self.W_be[1])
		gh = self.LN(rg * T.dot((h_pre * hdp[2]), self.WN(self.U[2], None)), self.U_al[1], self.U_be[1])

		gg = T.tanh(gx + gh + self.b[2])

		h = ((1. - zg) * h_pre) + (zg * gg)
		return h

	def resetBatch(self, newbatch):
		"""
		>>
		> newbatch : the new batch
		< return :
		"""
		self.minibatch = newbatch

	def saveModel(self, dirname):
		"""
		>>
		> e_n : the num of the epoch
		> dirname : the directory name
		< return :
		"""
		filename = dirname + self.name + ".npz"

		print '    >>[Debug] Saving the model %s.' % filename
		np.savez(filename, W = self.W.get_value(),
							U = self.U.get_value(),
							b = self.b.get_value(),
                            W_al = self.W_al.get_value(),
                            W_be = self.W_be.get_value(),
                            U_al = self.U_al.get_value(),
                            U_be = self.U_be.get_value())
		print '    >>[Debug] Model is saved...'

	def readModel(self, filename):
		"""
		>>
		> filename : the filename that contains the model
		< return :
		"""
		filename = filename + self.name + ".npz"

		print '    >>[Debug] Reading the model from file : %s' % filename
		model = np.load(filename)
		self.W.set_value(model['W'])
		self.U.set_value(model['U'])
		self.b.set_value(model['b'])
		self.W_al.set_value(model['W_al'])
		self.W_be.set_value(model['W_be'])
		self.U_al.set_value(model['U_al'])
		self.U_be.set_value(model['U_be'])
		print '    >>[Debug] Model is read...'

class CLSTMLayer(object):
	""" the Long-short term memory layer class """
	def __init__(self, idim, hdim, minibatch, name, isOrtho = True, bptt_trunc = -1):
		"""
		>>
		> idim : the input dimension
		> hdim : the hidden layer dimension
		> minibatch : the minibatch
		> name : name of the layer, this especially useful when save or read models
		> isOrtho : is the matrix is ortho matrix
		> bptt_trunc : used in scan
		< return :
		"""
		self.idim = idim
		self.hdim = hdim
		self.minibatch = minibatch
		self.name = name
		self.wName = name + '_Ws'
		self.uName = name + '_Us'
		self.bName = name + '_bs'
		self.bptt_trunc = bptt_trunc

		#W = getThreshold(idim, hdim) * rnd.randn(4, idim, hdim).astype(theano.config.floatX)
		#U = getThreshold(hdim, hdim) * rnd.randn(4, hdim, hdim).astype(theano.config.floatX)
		#b = getThreshold(0, 0) * rnd.randn(4, hdim).astype(theano.config.floatX)
		W = getThreshold(idim, hdim) * getWeight(4, idim, hdim, isOrtho)
		U = getThreshold(hdim, hdim) * getWeight(4, hdim, hdim, isOrtho)
		b = getThreshold(0, 0) * rnd.randn(4, hdim).astype(theano.config.floatX)

		self.W = theano.shared(value = W, name = self.wName, borrow = True)
		self.U = theano.shared(value = U, name = self.uName, borrow = True)
		self.b = theano.shared(value = b, name = self.bName, borrow = True)

		self.params = [self.W, self.U, self.b]
	# FIXME : the activate function should not include the scan
	def activate(self, inputs, h_pre, xdp, hdp):
		"""
		>>
		> inputs : the inputs
		< return : the activated value
		"""
		# FIXME : when decoding, how the c_pre matrix transfer to there
		c_pre = T.zeros((self.minibatch, self.hdim))
		self.inputs = inputs
		def recurrence(x, h_pre, c_pre):
			ig = T.nnet.sigmoid(T.dot(x, self.W[0]) + T.dot(h_pre, self.U[0]) + self.b[0])
			fg = T.nnet.sigmoid(T.dot(x, self.W[1]) + T.dot(h_pre, self.U[1]) + self.b[1])
			og = T.nnet.sigmoid(T.dot(x, self.W[2]) + T.dot(h_pre, self.U[2]) + self.b[2])
			gg = T.tanh(T.dot(x, self.W[3]) + T.dot(h_pre, self.U[3]) + self.b[3])
			c = (c_pre * fg) + (gg * ig)
			h = T.tanh(c) * og # there maybe need a bias
			return [h, c]
		h_out, c_out, _ = theano.scan(fn = recurrence,
								sequences = self.inputs,
								outputs_info = [h_pre, c_pre],
								truncate_gradient = self.bptt_trunc)
		return h_out

	def resetBatch(self, newbatch):
		"""
		>>
		> newbatch : the new batch
		< return : 
		"""
		self.minibatch = newbatch

	def saveModel(self, dirname):
		"""
		>>
		> e_n : the num of the epoch
		> dirname : the directory name
		< return : 
		"""
		filename = dirname + self.name + ".npz"

		print '    >>[Debug] Saving the model %s.' % filename
		np.savez(filename, W = self.W.get_value(),
							U = self.U.get_value(),
							b = self.b.get_value())
		print '    >>[Debug] Model is saved...'

	def readModel(self, filename):
		"""
		>>
		> filename : the filename that contains the model
		< return :
		"""
		filename = filename + self.name + ".npz"

		print '    >>[Debug] Reading the model from file : %s' % filename
		model = np.load(filename)
		model = np.load(filename)
		self.W.set_value(model['W'])
		self.U.set_value(model['U'])
		self.b.set_value(model['b'])
		print '    >>[Debug] Model is read...'

class CAttentionLayer(object):
	""" the attention layer class """
	def __init__(self, hdim, vsize, minibatch, name):
		"""
		>>
		> hdim : the hidden dimension
		> vsize : the vocabulary size
		> minibatch : the batch size
		> name : the layer's name
		< return : 
		"""
		self.hdim = hdim
		self.vsize = vsize
		self.minibatch = minibatch
		self.name = name
		self.waName = name + '_Wa'
		self.baName = name + '_ba'
		self.wcName = name + '_Wc'
		self.bcName = name + '_bc'
		self.vaName = name + '_Va'

		Wa = getThreshold(3 * hdim, hdim) * rnd.randn(3 * hdim, hdim).astype(theano.config.floatX)
		ba = getThreshold(hdim, 0) * rnd.randn(hdim).astype(theano.config.floatX)
		Wc = getThreshold(3 * hdim, hdim) * rnd.randn(3 * hdim, hdim).astype(theano.config.floatX)
		bc = getThreshold(hdim, 0) * rnd.randn(hdim).astype(theano.config.floatX)
		Va = getThreshold(hdim, 0) * rnd.randn(hdim).astype(theano.config.floatX)

		self.Wa = theano.shared(value = Wa, name = self.waName, borrow = True)
		self.ba = theano.shared(value = ba, name = self.baName, borrow = True)
		self.Wc = theano.shared(value = Wc, name = self.wcName, borrow = True)
		self.bc = theano.shared(value = bc, name = self.bcName, borrow = True)
		self.Va = theano.shared(value = Va, name = self.vaName, borrow = True)

		self.params = [self.Wa, self.ba, self.Wc, self.bc, self.Va]

	def activate(self, h_inputs, en_inputs, broadcast_mask):
		"""
		>>
		> h_inputs : the output matrix from decoder's last hidden layer
		< return : the last hidden output, that will calc with outputlayer
		"""
		#TODO >> should used att-mask and set the score.T's useless location as -inf
		broadcast  = broadcast_mask * h_inputs # <maxlen, minibatch, hidm>
		at_con = T.concatenate((en_inputs, broadcast), axis = 2) # <maxlen, minibatch, 3 * hdim>
		score = T.dot(T.tanh(T.dot(at_con, self.Wa) + self.ba), self.Va) # <maxlen, minibatch> = dot(<maxlen, minibatch, hdim>, <hdim>)
		at = T.nnet.softmax(score.T) # at's shape : <minibatch, maxlen> | score's shape is <maxlen, minibatch>, so when use score, we should score.T(maxlen is the 1st axis, minibatch is the 2nd axis, and the 3rd axis is squeeze to 1 scalar)
		hj = en_inputs.dimshuffle(2, 1, 0) # hj's shape : <2 * hdim, minibatch, maxlen>
		ct = T.sum((hj * at), axis = 2) # <2 * hdim, minibath>
		ct = ct.dimshuffle(1, 0) # <minibatch, 2 * hdim>
		ht_con = T.concatenate((ct, h_inputs), axis = 1) # <minibatch, 3 * hdim>
		ht = T.tanh(T.dot(ht_con, self.Wc) + self.bc) # <minibatch, hdim>

		return ht#, ct

	def resetBatch(self, newbatch):
		"""
		>>
		> newbatch : the new batch
		< return : 
		"""
		self.minibatch = newbatch

	def saveModel(self, dirname):
		"""
		>>
		> e_n : the num of the epoch
		> dirname : the directory name
		< return : 
		"""
		filename = dirname + self.name + ".npz"

		print '    >>[Debug] Saving the model %s.' % filename
		np.savez(filename, Wa = self.Wa.get_value(),
							ba = self.ba.get_value(),
							Wc = self.Wc.get_value(),
							bc = self.bc.get_value(),
							Va = self.Va.get_value())
		print '    >>[Debug] Model is saved...'

	def readModel(self, filename):
		"""
		>>
		> filename : the filename that contains the model
		< return :
		"""
		filename = filename + self.name + ".npz"

		print '    >>[Debug] Reading the model from file : %s' % filename
		model = np.load(filename)
		self.Wa.set_value(model['Wa'])
		self.ba.set_value(model['ba'])
		self.Wc.set_value(model['Wc'])
		self.bc.set_value(model['bc'])
		self.Va.set_value(model['Va'])
		print '    >>[Debug] Model is read...'

class COutputLayer(object):
	""" the output layer class """
	def __init__(self, idim, vsize, minibatch, name):
		"""
		>>
		> idim : the input dim
		> vsize : the size of the vocabulary
		> minibatch : the batch size
		> name : name of the layer, this especially useful when save or read models
		< return :
		"""
		self.idim = idim
		self.vsize = vsize
		self.minibatch = minibatch
		self.name = name
		self.vName = name + '_V'
		self.bName = name + '_b'

		V = getThreshold(0, 0) * rnd.randn(idim, vsize).astype(theano.config.floatX)
		b = getThreshold(0, 0) * rnd.randn(vsize).astype(theano.config.floatX)

		self.V = theano.shared(value = V, name = self.vName, borrow = True)
		self.b = theano.shared(value = b, name = self.bName, borrow	= True)

		self.params = [self.V, self.b]

	def activate(self, inputs):
		"""
		>>
		> inputs : the inputs
		< return : the activated value
		"""
		tmp_output = T.dot(inputs, self.V) + self.b
		return tmp_output # <minibatch, vsize>

	def resetBatch(self, newbatch):
		"""
		>>
		> newbatch : the new batch
		< return : 
		"""
		self.minibatch = newbatch

	def saveModel(self, dirname):
		"""
		>>
		> e_n : the num of the epoch
		> dirname : the directory name
		< return : 
		"""
		filename = dirname + self.name + ".npz"

		print '    >>[Debug] Saving the model %s.' % filename
		np.savez(filename, V = self.V.get_value(),
							b = self.b.get_value())
		print '    >>[Debug] Model is saved...'

	def readModel(self, filename):
		"""
		>>
		> filename : the filename that contains the model
		< return :
		"""
		filename = filename + self.name + ".npz"

		print '    >>[Debug] Reading the model from file : %s' % filename
		model = np.load(filename)
		self.V.set_value(model['V'])
		self.b.set_value(model['b'])
		print '    >>[Debug] Model is read...'
