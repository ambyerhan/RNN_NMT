# rebuild project
# ambyer
# 2017.06.29 - 

# this file contains a NMT model

import time
from datetime import datetime

from Optimization import COptimization
from EncoderDecoder import *
from utils import *
from Optimization import adam_func

class CNMT(object):
	""" the nmt class """
	def __init__(self, edim, hdim, svsize, tvsize, minibatch, name, inverted, dropout, isOrtho = True, en2de = 0, unit = "sRNN", method = 0, lrate = 0.1, ln = True, wn = False, beamsize = 12, maxlen = 50, clip = 5, gc = 3.0):
		"""
		>>
		> edim : size of embedding layer
		> hdim : size of hidden layer
		> svsize : src vocabulary size
		> tvsize : tgt vocabulary size
		> minibatch : batch size
		> name : name of the nmt class
		> doptout : the hidden dropout
		> inverted : is bool type, indicate that whether dropout is inverted dropout or not
		> isOrtho : is the matrix is ortho matrix
		> en2de : the method of encoder's output to decoder's init-state
		> unit : the hidden unit, sRNN or GRU or LSTM
		> method : the Optimum method of the deep learning [0:SGD, 1:Adadelta, 2:Adam]
		> lrate : learning rate
		> ln : bool type, layer normalization or not
		> wn : bool type, weight normalization or not
		> beamsize : the beam size
		> maxlen : the max-length of decoded sequence
		> clip : the gradient clip
		> gc : used in the formular that prevent to gradient exploring
		< return :
		"""
		self.edim = edim
		self.hdim = hdim
		self.svsize = svsize
		self.tvsize = tvsize
		self.minibatch = minibatch
		self.en_embedding_name = name + '.EnEmbedding'
		self.de_embedding_name = name + '.DeEmbedding'
		self.forward_en_name = name + '.forward_encoder'
		self.bacward_en_name = name + '.bacward_encoder'
		self.decoder_name = name + '.decoder'
		self.optimize_name = name + '.optimize'
		self.dp = dropout
		self.inverted = inverted
		self.en2de = en2de
		self.unit = unit
		self.method = method
		self.lrate = lrate
		self.beamsize = beamsize
		self.maxlen = maxlen
		self.clip = clip
		self.gc = gc

		# add names to dict
		self.names = [self.en_embedding_name,
					  self.de_embedding_name,
					  self.forward_en_name,
					  self.bacward_en_name,
					  self.decoder_name,
					  self.optimize_name]
		self.checkNames()

		# embedding layers
		self.en_embedding = CEmbeddingLayer(edim = edim, vsize = svsize, name = self.en_embedding_name)
		self.de_embedding = CEmbeddingLayer(edim = edim, vsize = tvsize, name = self.de_embedding_name)

		# encoders
		self.forward_encoder = CEncoder(edim = edim, hdim = hdim, svsize = svsize, minibatch = minibatch, name = self.forward_en_name, inverted = inverted, dropout = self.dp, isOrtho = isOrtho, unit = self.unit)
		self.bacward_encoder = CEncoder(edim = edim, hdim = hdim, svsize = svsize, minibatch = minibatch, name = self.bacward_en_name, inverted = inverted, dropout = self.dp, isOrtho = isOrtho, unit = self.unit)

		# decoder
		self.decoder = CDecoder(edim = edim, hdim = hdim, tvsize = tvsize, minibatch = minibatch, en2de = self.en2de, name = self.decoder_name, inverted = inverted, dropout = self.dp, isOrtho = isOrtho, unit = self.unit, beamsize = self.beamsize)

		# all params add here
		self.params = []
		
		# embeddings
		self.params.extend(self.en_embedding.params)
		self.params.extend(self.de_embedding.params)

		# forward encoder
		self.params.extend(self.forward_encoder.hidden_1.params)
		self.params.extend(self.forward_encoder.hidden_2.params)

		# bacward encoder
		self.params.extend(self.bacward_encoder.hidden_1.params)
		self.params.extend(self.bacward_encoder.hidden_2.params)

		# decoder
		self.params.extend(self.decoder.params) # params for init state
		self.params.extend(self.decoder.hidden_1.params)
		self.params.extend(self.decoder.hidden_2.params)
		self.params.extend(self.decoder.attention.params)
		self.params.extend(self.decoder.output.params)

		# optimization
		self.optimize = COptimization(params = self.params, name = self.optimize_name, clip = self.clip, method = self.method, lrate = self.lrate)

	def concate(self, h, rh, lastpos):
		"""
		>>
		> h1 : the hidden layer's output, is a 3D matrix
		> h2 : the hidden layer's output, is a 3D matrix, but the order is opposite to h1
		> lastpos : the position that refers to the last encoder's output, nearest to the decoder
		< return : concatenate the h and rh and return the big matrix
		"""
		def reverse(rh, pos):
			return T.concatenate(((rh[:pos + 1])[::-1], rh[pos + 1:]), axis = 0)
		nrh, _ = theano.scan(fn = reverse,
							 sequences = [rh.dimshuffle(1, 0, 2), lastpos],
							 outputs_info = None)
		nrh = nrh.dimshuffle(1, 0, 2)
		return (T.concatenate((h, nrh), axis = 2))

	def buildNetwork(self, isForTrain = True):
		"""
		>>
		> isForTrain : indicate that build for train or predict
		< return : 
		"""
		if isForTrain:
			""" build for training the network """
			x_forward = T.imatrix("x_forward")
			x_bacward = T.imatrix("x_bacward")
			y = T.imatrix('y')
			gold = T.imatrix('gold')
			lp = T.ivector('lp')
			hm = T.tensor3('hm')
			am = T.tensor3('am')
			#hm = T.imatrix('hm')
			om = T.imatrix('om')

			""" encode part """
			em_forward_out = self.en_embedding.activate(x_forward.T)
			em_bacward_out = self.en_embedding.activate(x_bacward.T)

			h1_forward_out, h2_forward_out = self.forward_encoder.encode(em_forward_out, True)
			h1_bacward_out, h2_bacward_out = self.bacward_encoder.encode(em_bacward_out, True)

			en_h_out = self.concate(h2_forward_out, h2_bacward_out, lp) * hm

			h1_init = []
			h2_init = []
			if self.en2de == 0:
				h1_last = h1_bacward_out[lp, T.arange(self.minibatch)]
				h2_last = h2_bacward_out[lp, T.arange(self.minibatch)]
				h1_init = h1_last
				h2_init = h2_last
			else: # set the mean of encoder's output to h1_init and h2_init
				en_h1_out = self.concate(h1_forward_out, h1_bacward_out, lp) * hm
				h1_init = en_h1_out.sum(axis = 0) / hm.sum(axis = 0) # note:if the hm is matrix and shaped<minibatch, maxlen>, then axis=1
				h2_init = en_h_out.sum(axis = 0) / hm.sum(axis = 0) # the shape go from <maxlen, minibatch, 2 * hdim> to <minibatch, 2 * hdim>

			""" decode part """
			em_out = self.de_embedding.activate(y.T)

			de_out = self.decoder.decode_forTrain(en_h_out, em_out, h1_init, h2_init, gold, am, om)

			""" calc grads and updates """
			self.cost = de_out

			param_updates = self.optimize.calcUpdate(self.cost)
			#param_updates = adam_func(self.cost, self.params)

			""" build the theano function """
			self.__train = theano.function(inputs = [x_forward, x_bacward, y, gold, lp, hm, am, om],
										   outputs = self.cost,
										   updates = param_updates,
										   allow_input_downcast = True)

			return

		else: # build for decode
			print '[Error] Not using this anymore!'
			sys.exit(1)
			""" reset the minibatchs """
			self.resetBatch(1)

			""" build for predicting """
			x_forward = T.ivector("x_forward")
			x_bacward = x_forward[::-1]
			ht_init = T.vector("ht_init")
			y = T.iscalar("y")

			""" encode part """
			em_forward_out = self.en_embedding.activate(x_forward.T) # <mxlen, edim>
			em_bacward_out = self.en_embedding.activate(x_bacward.T) # <mxlen, edim>

			h1_forward_out, h2_forward_out = self.forward_encoder.encode(em_forward_out)
			h1_bacward_out, h2_bacward_out = self.bacward_encoder.encode(em_bacward_out)

			en_h_out = T.concatenate((h2_forward_out, h2_bacward_out[::-1]), axis = 2)

			h1_init = []
			h2_init = []
			if self.en2de == 0:
				h1_last = h1_bacward_out[-1]
				h2_last = h2_bacward_out[-1]
				h1_init = h1_last
				h2_init = h2_last
			else:
				en_h1_out = T.concatenate((h1_forward_out, h1_bacward_out[::-1]), axis = 2)
				h1_init = en_h1_out.sum(axis = 0) / x_forward.shape[0] # note:if the hm is matrix and shaped<minibatch, maxlen>, then axis=1
				h2_init = en_h_out.sum(axis = 0) / x_forward.shape[0]


			""" decode part """
			prob, nidx, cost, h1_out, h2_out, ht_out = self.decoder.decode_forPredict(en_h_out, h1_init, h2_init, self.de_embedding)

			""" build the theano function """
			#self.__predict = theano.function(inputs = [x_forward],
			#								outputs = [tgt_idx, cost],
			#								allow_input_downcast = True)
			f_enc = theano.function(inputs = [x_forward],
									outputs = [en_h_out, h1_init, h2_init],
									allow_input_downcast = True)
			f_dec = theano.function(inputs = [y, en_h_out, h1_init, h2_init, ht_init],
									outputs = [prob, nidx, cost, h1_out, h2_out, ht_out],
									allow_input_downcast = True)

			return f_enc, f_dec

	def train(self, src_seq, tgt_seq):
		"""
		>>train one mini-batch one time
		> src_seq : the src lang seq typed int
		> tgt_seq : the tgt lang seq typed int
		< return : 
		"""
		""" encoder side """
		idx_forward = src_seq
		idx_bacward = get_rindex(src_seq)

		smxlen, lastpos = get_maxlen(idx_forward, True)
		set_pad(idx_forward, smxlen, idx_bacward)

		#hMask = get_Hout_2DMask(idx_forward, self.minibatch, smxlen)
		hMask = get_Hout_3DMask(idx_forward, self.minibatch, 2 * self.hdim, smxlen)
		aMask = get_Hout_3DMask(idx_forward, self.minibatch, self.hdim, smxlen)
		#atMask = get_Att_2DMask(hMask.sum(axis = 2) / (2 * self.hdim), lastpos)

		""" decoder side """
		idx_tgt = tgt_seq

		tmxlen, lp = get_maxlen(idx_tgt, True)
		set_pad(idx_tgt, tmxlen, None)

		oMask, idx_gold = get_Oout_2DMask(idx_tgt, self.minibatch, lp)

		final_rslt = self.__train(idx_forward, idx_bacward, idx_tgt, idx_gold.T, lastpos, hMask, aMask, oMask.T)

		return (final_rslt)

#################################################################################################################
	def _beam_predict(self, src_seq):
		"""
		>>train one mini-batch one time
		> src_seq : the src lang seq typed int
		< return : 
		"""
		""" encoder side """
		src_idx = src_seq

		return (self.__predict(src_idx))

	def _beam_gen_sample(self, seq, funcs_en, funcs_de, k, maxlen):
		"""
		>>
		"""
		samples_idxs  = [] # the indices
		samples_cost = [] # after softmax
		samples_prob = [] # the softmax's output, not the cost

		live_k = 1
		dead_k = 0

		""" encoder part """
		n_models = len(funcs_en)
		ctxs = [None] * n_models
		h1_states = [None] * n_models
		h2_states = [None] * n_models
		ht_states = [np.array(np.zeros(self.hdim), dtype = theano.config.floatX)] * n_models

		for i in xrange(n_models):
			en_h_out, h1_init, h2_init = funcs_en[i](seq)

			ctxs[i] = en_h_out
			h1_states[i] = h1_init
			h2_states[i] = h2_init

		""" decoder part """
		hyp_idxs  = [[] for i in xrange(live_k)]
		hyp_costs = np.zeros(live_k).astype(theano.config.floatX)
		hyp_probs = [[] for i in xrange(live_k)]
		hyp_h1_states = []
		hyp_h2_states = []

		y_idx = np.int64(0)
		cur_probs = [None] * n_models
		cur_costs = [0.] * n_models

		for iii in xrange(maxlen): # the decoded sequence length must be smaller than maxlen
			# get each decoder's output
			for i in xrange(n_models):
				ctx = ctxs[i]
				h1_init = h1_states[i]
				h2_init = h2_states[i]
				ht_init = ht_states[i]

				cur_probs[i], \
				_, \
				cur_costs[i], \
				h1_states[i], \
				h2_states[i], \
				ht_states[i] = funcs_de[i](y_idx, ctx, h1_init, h2_init, ht_init)

			costs = hyp_costs[:, None] + sum(cur_costs)
			probs = sum(cur_probs) / n_models

			costs_flat = costs.flatten()
			probs_flat = probs.flatten()
			ranks_flat = costs_flat.argpartition(k - dead_k - 1)[: (k - dead_k)]

			iis = ranks_flat / self.tvsize
			jjs = ranks_flat % self.tvsize
			costs_ = costs[ranks_flat]

			new_hyp_idxs = []
			new_hyp_costs = np.zeros(k - dead_k).astype(theano.config.floatX)
			new_hyp_probs = []
			new_hyp_h1_states = []
			new_hyp_h2_states = []
			new_hyp_ht_states = []

			for t, [ii, jj] in enumerate(zip(iis, jjs)):
				new_hyp_idxs.append(hyp_idxs[ii] + [jj])
				new_hyp_costs[t] = copy.copy(costs_[t])
				new_hyp_probs.append(hyp_probs[ii] + [probs_flat[ranks_flat[t]].tolist()])
				new_hyp_h1_states.append([copy.copy(h1_states[t1][ii]) for t1 in xrange(n_models)])
				new_hyp_h2_states.append([copy.copy(h2_states[t2][ii]) for t2 in xrange(n_models)])
				new_hyp_ht_states.append([copy.copy(ht_states[tt][ii]) for tt in xrange(n_models)])

			new_live_k = 0

			hyp_idxs = []
			hyp_costs = []
			hyp_probs = []
			hyp_h1_states = []
			hyp_h2_states = []
			hyp_ht_states = []

			for i in xrange(len(new_hyp_idxs)):
				if new_hyp_idxs[i][-1] == 0: # meet the eos tag
					samples_idxs.append(copy.copy(new_hyp_idxs[i]))
					samples_cost.append(new_hyp_costs[i])
					samples_prob.append(new_hyp_probs[i])

					dead_k += 1
				else:
					hyp_idxs.append(copy.copy(new_hyp_idxs[i]))
					hyp_costs.append(new_hyp_costs[i])
					hyp_probs.append(new_hyp_probs[i])
					hyp_h1_states.append(copy.copy(new_hyp_h1_states[i]))
					hyp_h2_states.append(copy.copy(new_hyp_h2_states[i]))
					hyp_ht_states.append(copy.copy(new_hyp_ht_states[i]))

					new_live_k += 1

			hyp_costs = np.array(hyp_costs)
			live_k = new_live_k

			if new_live_k < 1 or dead_k >= k:
				break

			y_idx = np.array(w[-1] for w in hyp_idxs)
			h1_states = [np.array(state) for state in zip(*hyp_h1_states)]
			h2_states = [np.array(state) for state in zip(*hyp_h2_states)]
			ht_states = [np.array(state) for state in zip(*hyp_ht_states)]

		if live_k > 0:
			for i in xrange(live_k):
				samples_idxs.append(hyp_idxs[i])
				samples_cost.append(hyp_costs[i])
				samples_prob.append(hyp_probs[i])

		return samples_idxs, samples_cost, samples_prob, None

	def _beam_buildTranslate(self, models, options):
		"""
		>>
		"""
		self.funcs_enc = []
		self.funcs_dec = []

		for model, option in zip(models, options):
			func_enc, func_dec = self.buildNetwork(False)

			self.funcs_enc.append(func_enc)
			self.funcs_dec.append(func_dec)

	def _beam_translate(self, nbest, normal_alpha, queue, rslt_queue, maxlen):
		"""
		>>
		"""
		def _translate(seq):
			sample, score, word_prob, alignment = self.gen_sample(seq, self.funcs_enc, self.funcs_dec, self.beamsize, maxlen)

			# normalize scores according to seq len
			if normal_alpha:
				adjust_len = np.array([len(s) ** normal_alpha for s in sample])
				score = score / adjust_len

			if  nbest:
				return sample, score, word_prob, alignment
			else:
				idx = np.argmin(score)
				return sample[idx], score[idx], word_prob[idx], alignment[idx]

		while True:
			request = queue.get()
			if request is None:
				break

			i, x = request[0], request[1]

			rslt = _translate(x)

			rslt_queue.put((i, rslt))

		return
#################################################################################################################

	def buildTranslate(self):
		"""
		>>
		"""
		""" reset the minibatchs """
		self.resetBatch(1)

		""" build for predicting """
		x_forward = T.ivector("x_forward")
		x_bacward = x_forward[::-1]

		""" encode part """
		em_forward_out = self.en_embedding.activate(x_forward.T) # <mxlen, edim>
		em_bacward_out = self.en_embedding.activate(x_bacward.T) # <mxlen, edim>

		h1_forward_out, h2_forward_out = self.forward_encoder.encode(em_forward_out, False)
		h1_bacward_out, h2_bacward_out = self.bacward_encoder.encode(em_bacward_out, False)

		en_h_out = T.concatenate((h2_forward_out, h2_bacward_out[::-1]), axis = 2)

		h1_init = []
		h2_init = []
		if self.en2de == 0:
			h1_last = h1_bacward_out[-1]
			h2_last = h2_bacward_out[-1]
			h1_init = h1_last
			h2_init = h2_last
		else:
			en_h1_out = T.concatenate((h1_forward_out, h1_bacward_out[::-1]), axis = 2)
			h1_init = en_h1_out.sum(axis = 0) / x_forward.shape[0] # note:if the hm is matrix and shaped<minibatch, maxlen>, then axis=1
			h2_init = en_h_out.sum(axis = 0) / x_forward.shape[0]

		""" build the theano function """
		self._f_enc = theano.function(inputs = [x_forward],
									  outputs = [en_h_out, h1_init, h2_init],
									  allow_input_downcast = True)


		""" build for predicting """
		y = T.ivector("y")
		en_h = T.tensor3('en_h')
		en_h1 = T.matrix('en_h1')
		en_h2 = T.matrix('en_h2')
		ht_init = T.matrix('ht')

		""" decode part """
		prob, cost, h1_out, h2_out, ht_out = self.decoder.decode_forPredict(y, en_h, en_h1, en_h2, ht_init, self.de_embedding)

		""" build the theano function """
		self._f_dec = theano.function(inputs = [y, en_h, en_h1, en_h2, ht_init],
									  outputs = [prob, cost, h1_out, h2_out, ht_out],
									  allow_input_downcast = True)

	def gen_sample(self, seq, k, maxlen):
		"""
		>>
		"""
		samples_idxs  = [] # the indices
		samples_cost = [] # after softmax
		samples_prob = [] # the softmax's output, not the cost

		live_k = 1
		dead_k = 0

		""" encoder part """
		ctxs, h1_states_, h2_states_ = self._f_enc(seq)
		h1_states = np.tanh(np.dot(h1_states_, self.decoder.Winit_h1.get_value()) + self.decoder.b_h1.get_value())
		h2_states = np.tanh(np.dot(h2_states_, self.decoder.Winit_h2.get_value()) + self.decoder.b_h2.get_value())
		ht_states = np.array(np.zeros((1, self.hdim)), dtype = theano.config.floatX)


		""" decoder part """
		hyp_idxs  = [[] for i in xrange(live_k)]
		hyp_costs = np.zeros(live_k).astype(theano.config.floatX)
		hyp_probs = [[] for i in xrange(live_k)]
		hyp_h1_states = []
		hyp_h2_states = []

		y_idx = np.zeros((live_k, )).astype('int64') # np.int64(0)
		for iii in xrange(maxlen): # the decoded sequence length must be smaller than maxlen
			ctx = np.tile(ctxs, [live_k, 1])
			h1_init = h1_states#np.transpose(h1_states, (1, 0))
			h2_init = h2_states#np.transpose(h2_states, (1, 0))
			ht_init = ht_states

			cur_probs, \
			cur_costs, \
			h1_states, \
			h2_states, \
			ht_states = self._f_dec(y_idx, ctx, h1_init, h2_init, ht_init)

			costs = hyp_costs[:, None] + (cur_costs) # why sum()[in the nematus, it sums, there no needed] >> sum() because the dim of the next_p and np.log(next_p) is one more than needed, and when sum the dim will decrease one dim
			probs = (cur_probs)

			costs_flat = costs.flatten()
			probs_flat = probs.flatten()
			ranks_flat = costs_flat.argpartition(k - dead_k - 1)[: (k - dead_k)]

			iis = ranks_flat / self.tvsize
			jjs = ranks_flat % self.tvsize
			costs_ = costs_flat[ranks_flat]

			new_hyp_idxs = []
			new_hyp_costs = np.zeros(k - dead_k).astype(theano.config.floatX)
			new_hyp_probs = []
			new_hyp_h1_states = []
			new_hyp_h2_states = []
			new_hyp_ht_states = []

			for t, [ii, jj] in enumerate(zip(iis, jjs)):
				new_hyp_idxs.append(hyp_idxs[ii] + [jj])
				new_hyp_costs[t] = copy.copy(costs_[t])
				new_hyp_probs.append(hyp_probs[ii] + [probs_flat[ranks_flat[t]].tolist()])
				new_hyp_h1_states.append([copy.copy(h1_states[ii])])
				new_hyp_h2_states.append([copy.copy(h2_states[ii])])
				new_hyp_ht_states.append([copy.copy(ht_states[ii])])

			hyp_idxs = []
			hyp_costs = []
			hyp_probs = []
			hyp_h1_states = []
			hyp_h2_states = []
			hyp_ht_states = []

			new_live_k = 0

			for i in xrange(len(new_hyp_idxs)):
				if new_hyp_idxs[i][-1] == 1: # meet the eos tag
					samples_idxs.append(copy.copy(new_hyp_idxs[i]))
					samples_cost.append(new_hyp_costs[i])
					samples_prob.append(new_hyp_probs[i])

					dead_k += 1
				else:
					hyp_idxs.append(copy.copy(new_hyp_idxs[i]))
					hyp_costs.append(new_hyp_costs[i])
					hyp_probs.append(new_hyp_probs[i])
					hyp_h1_states.append(copy.copy(new_hyp_h1_states[i]))
					hyp_h2_states.append(copy.copy(new_hyp_h2_states[i]))
					hyp_ht_states.append(copy.copy(new_hyp_ht_states[i]))

					new_live_k += 1

			hyp_costs = np.array(hyp_costs)
			live_k = new_live_k

			if new_live_k < 1 or dead_k >= k:
				break

			y_idx = np.array([w[-1] for w in hyp_idxs])
			h1_states = [np.array(state1[0]) for state1 in hyp_h1_states] # the first dimension is len, and every time-step, the len is 1
			h2_states = [np.array(state2[0]) for state2 in hyp_h2_states]
			ht_states = [np.array(statet[0]) for statet in hyp_ht_states]

		if live_k > 0:
			for i in xrange(live_k):
				samples_idxs.append(hyp_idxs[i])
				samples_cost.append(hyp_costs[i])
				samples_prob.append(hyp_probs[i])

		return samples_idxs, samples_cost, samples_prob, None

	def translate(self, i_queue, o_queue, nbest, normal_alpha, maxlen):
		"""
		>>
		"""
		def _translate(seq):
			sample, score, word_prob, alignment = self.gen_sample(seq, self.beamsize, maxlen)

			# normalize scores according to seq len
			if normal_alpha > 0.:
				adjust_len = np.array([len(s) ** normal_alpha for s in sample]) # np.array([np.count_nonzeros(s) ** alpha for s in sample])
				score = score / adjust_len

			if  nbest:
				return sample, score, word_prob, alignment
			else:
				idx = np.argmin(score)
				return sample[idx], score[idx], word_prob[idx], None

		while True:
			request = i_queue.get()
			if request is None:
				break

			i, x = request[0], request[1]

			rslt = _translate(x)

			o_queue.put((i, rslt))

		return

	def checkNames(self):
		"""
		>>check the names and garentee that there is no identical name
		< reuturn : 
		"""
		tmp_dic = {}
		for name in self.names:
			if tmp_dic.has_key(name):
				CErrorMsg.showErrMsg("There are members' names are identical!")
			else:
				tmp_dic[name] = 0

	def resetBatch(self, newbatch):
		"""
		>>
		> newbatch : the new batch
		< return : 
		"""
		self.minibatch = newbatch
		self.en_embedding.resetBatch(newbatch)
		self.de_embedding.resetBatch(newbatch)
		self.forward_encoder.resetBatch(newbatch)
		self.bacward_encoder.resetBatch(newbatch)
		self.decoder.resetBatch(newbatch)

	def saveModel(self, e_n, dirname):
		"""
		>>
		> e_n : the num of the epoch
		> dirname : the directory name
		< return : 
		"""
		date = datetime.now().strftime("%Y.%m.%d")
		if dirname[-1] != '/':
			dirname += '/'
		filename = dirname + date + '-' + str(e_n) + '-'
		print '[Save] Saving the models of epoch/g_step <%d>...' % e_n
		self.en_embedding.saveModel(filename)
		self.de_embedding.saveModel(filename)
		self.forward_encoder.saveModel(filename)
		self.bacward_encoder.saveModel(filename)
		self.decoder.saveModel(filename)

		self.optimize.saveOptModel()
		print '[Save] Models are saved...'

		return filename

	def readModel(self, filename):
		"""
		>>
		> filename : the filename that contains the model
		< return :
		"""
		print '[Read] Reading the models from file : %s.*' % filename
		self.en_embedding.readModel(filename)
		self.de_embedding.readModel(filename)
		self.forward_encoder.readModel(filename)
		self.bacward_encoder.readModel(filename)
		self.decoder.readModel(filename)

		self.optimize.readOptModel(filename)
		print '[Read] Models are read...'
