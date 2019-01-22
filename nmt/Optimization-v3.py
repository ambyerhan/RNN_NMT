# rebuild project
# ambyer
# 2017.06.28 - 2017.06.28

# this file contains Optimization class

import theano
import theano.tensor as T
import numpy as np
from ErrorMsg import CErrorMsg

class COptimization(object):
	""" the optimization class """ 
	def __init__(self, params, name, clip = 5, method = 0, lrate = 0.1):
		"""
		>>
		> params : the params that need to be updated
		> name  : the name
		> clip : the gradient clip
		> method : the optimization method [0:SGD, 1:Adadelta, 2:Adam]
		> lrate : the learning rate, only useful when method == 0
		< return : 
		"""
		self.params = params
		self.clip = clip
		self.method = method
		self.name = name

		CErrorMsg.showErrExMsg((self.method in [0, 1, 2]), "the method is wrong, please check!")
		if self.method == 0:
			# initialize  constant
			self.lrate = lrate

		if self.method == 1:
			# initialize  constant
			self.rho = 0.95
			self.eps = 1e-6

			# initialize tensor shared variable
			self.Eg2 = [theano.shared(np.zeros(param.get_value(borrow = True).shape, dtype = theano.config.floatX)) for param in self.params]
			self.Ex2 = [theano.shared(np.zeros(param.get_value(borrow = True).shape, dtype = theano.config.floatX)) for param in self.params]

		if self.method == 2:
			# initialize  constant
			self.alpha = 0.001
			self.beta1 = 0.9
			self.beta2 = 0.999
			self.eps = 1e-8
			self.ts = 0.

			# initialize tensor shared variable
			self.mt = [theano.shared(np.zeros(param.get_value(borrow = True).shape, dtype = theano.config.floatX)) for param in self.params]
			self.vt = [theano.shared(np.zeros(param.get_value(borrow = True).shape, dtype = theano.config.floatX)) for param in self.params]
	
	def __sgd(self, cost):
		"""
		>>
		> cost : the cost value
		< return : return the updates
		"""
		# calculate grads
		grads = theano.grad(theano.gradient.grad_clip(cost, -1.0 * self.clip, self.clip), self.params)

		# update params
		pa_update = [(param, param - self.lrate * grad) for param, grad in zip(self.params, grads)]

		return pa_update

	def __adadelta(self, cost):
		"""
		>>
		> cost : the cost value
		< return : return the updates
		"""
		# calculate grads
		grads = theano.grad(theano.gradient.grad_clip(cost, -1.0 * self.clip, self.clip), self.params)

		# update g2 first
		Eg2_update = [(g2, self.rho * g2 + (1.0 - self.rho) * (g ** 2)) for g2, g in zip(self.Eg2, grads)]
		
		# calculate the delta_x by RMS [RMS(x) = sqrt(x + eps)]
		delta_x = [-1.0 * (T.sqrt(x2_last + self.eps) / T.sqrt(g2_now[1] + self.eps)) * g for x2_last, g2_now, g in zip(self.Ex2, Eg2_update, grads)]
		
		# update Ex2 and params
		Ex2_update = [(x2, self.rho * x2 + (1.0 - self.rho) * (x ** 2)) for  x2, x in zip(self.Ex2, delta_x)] # the delta_x's each elem in for cannot be same in two for(or else, there is a error[the length is not known]), here i use name 'x' and 'delta'.
		delta_x_update = [(param, param + delta) for param, delta in zip(self.params, delta_x)]

		return Eg2_update + Ex2_update + delta_x_update

	def __adam(self, cost):
		"""
		>>
		> cost : the cost value
		< return : return the updates
		"""
		# calculate grads
		grads = theano.grad(theano.gradient.grad_clip(cost, -1.0 * self.clip, self.clip), self.params)
		
		# update timestep
		self.ts += 1.
		
		# update m and v
		mt_upd = [(m, self.beta1 * m + (1. - self.beta1) * grad) for m, grad in zip(self.mt, grads)]
		vt_upd = [(v, self.beta2 * v + (1. - self.beta2) * (grad ** 2)) for v, grad in zip(self.vt, grads)]
		
		# calc mt^ and vt^ (here we note mt^ and vt^ as mt_ and vt_)
		mt_ = [m[1] / (1. - (self.beta1 ** self.ts)) for m in mt_upd]
		vt_ = [v[1] / (1. - (self.beta2 ** self.ts)) for v in vt_upd]
		
		# update params
		pa_upd = [(param, param - self.alpha * (m_ / (T.sqrt(v_) + self.eps))) for param, m_, v_ in zip(self.params, mt_, vt_)]
		
		return mt_upd + vt_upd + pa_upd

	def calcUpdate(self, cost):
		"""
		>>
		< return : return the updates
		"""
		if self.method == 0:
			return self.__sgd(cost)
		elif self.method == 1:
			return self.__adadelta(cost)
		else:
			return self.__adam(cost)
