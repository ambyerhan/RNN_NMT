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
		self.lrate = lrate
		#self.ts = 0.

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
			self.alpha = 0.0001
			self.beta1 = 0.9
			self.beta2 = 0.999
			self.eps = 1e-8
			self.t = 0.

			# initialize tensor shared variable
			self.ts = theano.shared(value = np.float32(0.), name = 'adam_t')
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

	def __adam_am(self, cost): # write by am
		"""
		>>
		> cost : the cost value
		< return : return the updates
		"""
		# calculate grads
		grads = theano.grad(theano.gradient.grad_clip(cost, -1.0 * self.clip, self.clip), self.params)
		
		# update timestep
		ts_upd = [(self.ts, self.ts + 1.)]
		
		# update m and v
		mt_upd = [(m, self.beta1 * m + (1. - self.beta1) * grad) for m, grad in zip(self.mt, grads)]
		vt_upd = [(v, self.beta2 * v + (1. - self.beta2) * (grad ** 2)) for v, grad in zip(self.vt, grads)]
		
		# calc mt^ and vt^ (here we note mt^ and vt^ as mt_ and vt_)
		mt_ = [m[1] / (1. - (self.beta1 ** ts_upd[0][1])) for m in mt_upd] # ts_upd = [(self.ts, self.ts + 1.)], so cur_ts = ts_upd[0][1]
		vt_ = [v[1] / (1. - (self.beta2 ** ts_upd[0][1])) for v in vt_upd]
		
		# update params
		pa_upd = [(param, param - self.alpha * (m_ / (T.sqrt(v_) + self.eps))) for param, m_, v_ in zip(self.params, mt_, vt_)]
		
		return mt_upd + vt_upd + pa_upd + ts_upd

	def __adam(self, cost): # Etinburg
		# calculate grads
		grads = theano.grad(theano.gradient.grad_clip(cost, -1.0 * self.clip, self.clip), self.params)
		
		# update timestep
		ts_upd = [(self.ts, self.ts + 1.)]
		nlr = self.alpha * T.sqrt(1. - self.beta2 ** ts_upd[0][1]) / (1. - self.beta1 ** ts_upd[0][1])
		
		# update m and v
		mt_upd = [(m, self.beta1 * m + (1. - self.beta1) * grad) for m, grad in zip(self.mt, grads)]
		vt_upd = [(v, self.beta2 * v + (1. - self.beta2) * (grad ** 2)) for v, grad in zip(self.vt, grads)]
		pa_upd = [(param, param - nlr * mt_[1] / (T.sqrt(vt_[1]) + self.eps)) for param, mt_, vt_ in zip(self.params, mt_upd, vt_upd)]
		
		return mt_upd + vt_upd + pa_upd + ts_upd

	def __adam_fix_t(self, cost):
		# calculate grads
		grads = theano.grad(theano.gradient.grad_clip(cost, -1.0 * self.clip, self.clip), self.params)

		# update timestep
		#ts_upd = [(self.ts, self.ts + 1.)]
		self.t += 1.

		# update m and v
		mt_upd = [(m, self.beta1 * m + (1. - self.beta1) * grad) for m, grad in zip(self.mt, grads)]
		vt_upd = [(v, self.beta2 * v + (1. - self.beta2) * (grad ** 2)) for v, grad in zip(self.vt, grads)]

		# calc mt^ and vt^ (here we note mt^ and vt^ as mt_ and vt_)
		mt_ = [m[1] / (1. - (self.beta1 ** self.t)) for m in mt_upd] # ts_upd = [(self.ts, self.ts + 1.)], so cur_ts = ts_upd[0][1]
		vt_ = [v[1] / (1. - (self.beta2 ** self.t)) for v in vt_upd]

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
			return self.__adam_am(cost)
			#return self.__adam(cost)
            #return self.__adam_fix_t(cost)

	def saveOptModel(self):
		"""
		>>
		"""
		filename = "./model/optimizer_param.npz"

		print '    >>[Debug] Saving the model %s.' % filename
		np.savez(filename, ts = self.ts.get_value(),
                           mt = self.mt,
                           vt = self.vt)
		print '    >>[Debug] Model is saved...'

	def readOptModel(self, filename):
		"""
		>>
		"""
		filename = "./model/optimizer_param.npz"
		print '    >>[Debug] Reading the model from file : %s' % filename
		model = np.load(filename)
		self.ts.set_value(model['ts'])
		self.mt = list(model['mt'])
		self.vt = list(model['vt'])
		print '   >>[Debug] Model is read...'


### update the parameters by adam function, not adam class
def sharedX(value, name = None, borrow = False, dtype = None):
	if dtype is None:
		dtype = theano.config.floatX
	return theano.shared(value = value, name = name, borrow = borrow)

def adam_func(cost, params, clip = 5., lr = 0.0002, b1 = 0.9, b2 = 0.999, eps = 1e-8):
	print '[Tip] Using the adam function'
	updates = []
	#grads = T.grad(cost, params)
	grads = theano.grad(theano.gradient.grad_clip(cost, -1.0 * clip, clip), params)

	t = sharedX(np.float32(0.))
	t_t = t + 1.
	fix_1 = 1. - b1 ** t_t
	fix_2 = 1. - b2 ** t_t
	lr_t = lr * (T.sqrt(fix_2) / fix_1)

	for p, g in zip(params, grads):
		m = sharedX(p.get_value() * 0.)
		v = sharedX(p.get_value() * 0.)
		m_t = ((1. - b1) * g) + (b1 * m)
		v_t = ((1. - b2) * g ** 2) + (b2 * v)
		g_t = m_t / (T.sqrt(v_t) + eps)
		p_t = p - (lr_t * g_t)

		updates.append((m, m_t))
		updates.append((v, v_t))
		updates.append((p, p_t))
	updates.append((t, t_t))

	return updates

def adam_Etin(lr, tparams, cost, beta1=0.9, beta2=0.999, e=1e-8, optimizer_params={}):
	"""
	"""
	grads = theano.grad(theano.gradient.grad_clip(cost, -5., 5.), tparams)
	PREFIX = 'adam_'

	updates = []
	optimizer_tparams = {}

	t_prev_name = PREFIX + 't_prev'
	if t_prev_name in optimizer_params:
		t_prev_init = optimizer_params[t_prev_name]
	else:
		t_prev_init = 0.
	t_prev = theano.shared(np.float32(t_prev_init), t_prev_name)
	optimizer_tparams[t_prev_name] = t_prev

	t = t_prev + 1.
	lr_t = lr * T.sqrt(1. - beta2 ** t) / (1. - beta1 ** t)

	for p, g in zip(tparams, grads):
		# Create/Load variable for first moment
		m_name = PREFIX + p.name + '_mean'
		if m_name in optimizer_params:
			m_init = optimizer_params[m_name]
		else:
			m_init = p.get_value() * 0.
		m = theano.shared(m_init, m_name)
		optimizer_tparams[m_name] = m

		# Create/Load variable for second moment
		v_name = PREFIX + p.name + '_variance'
		if v_name in optimizer_params:
			v_init = optimizer_params[v_name]
		else:
			v_init = p.get_value() * 0.
		v = theano.shared(v_init, v_name)
		optimizer_tparams[v_name] = v

		# Define updates on shared vars
		m_t = beta1 * m + (1. - beta1) * g
		v_t = beta2 * v + (1. - beta2) * g ** 2
		step = lr_t * m_t / (T.sqrt(v_t) + e)
		p_t = p - step
		updates.append((m, m_t))
		updates.append((v, v_t))
		updates.append((p, p_t))
	updates.append((t_prev, t))

	#f_update = theano.function([lr] + inp, cost, updates=updates, on_unused_input='ignore', profile=profile)

	return updates