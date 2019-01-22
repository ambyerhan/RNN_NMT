# rebuild project
# ambyer
# 2017.06.29 - 2017.06.29

# this file contains Encoder and Decoder class

from Layers import *
from ErrorMsg import CErrorMsg
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class CEncoder(object):
	""" the Encoder class """
	def __init__(self, edim, hdim, svsize, minibatch, name, inverted, dropout, isOrtho = True, ln = False, wn = False, unit = "sRNN", bptt_trun = -1):
		"""
		>>
		> edim : the embedding dim
		> hdim : the hidden dim
		> svsize : the source language vocabulary size
		> minibatch : the batch size
		> name : name of the Encoder, forward or backward
		> inverted : is the dropout is inverted
		> dropout : the hidden dropout prob
		> isOrtho : is the matrix is ortho matrix
		> ln : bool type, layer normalization or not
		> wn : bool type, weight normalization or not
		> unit : the hidden unit, sRNN or GRU or LSTM
		> bptt_trunc : used in scan
		< return :
		"""
		self.edim = edim
		self.hdim = hdim
		self.svsize = svsize
		self.minibatch = minibatch
		self.name = name
		self.inverted = inverted
		self.dp = dropout
		self.bptt_trun = bptt_trun

		self.trng = RandomStreams(1234)

		# hidden layer
		CErrorMsg.showErrExMsg((unit in ["sRNN", "GRU", "LSTM", "NGRU"]), "The hidden unit is wrong!")
		if unit == "sRNN":
			self.hidden_1 = CRNNLayer(idim = edim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_1', isOrtho = isOrtho, bptt_trunc = bptt_trun)
			self.hidden_2 = CRNNLayer(idim = hdim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_2', isOrtho = isOrtho, bptt_trunc = bptt_trun)
		if unit == "GRU":
			self.hidden_1 = CGRULayer(idim = edim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_1', isOrtho = isOrtho, bptt_trunc = bptt_trun)
			self.hidden_2 = CGRULayer(idim = hdim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_2', isOrtho = isOrtho, bptt_trunc = bptt_trun)
		if unit == "LSTM":
			print '[Warn] Not Done Yet!'
			exit()
			self.hidden_1 = CLSTMLayer(idim = edim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_1', isOrtho = isOrtho, bptt_trunc = bptt_trun)
			self.hidden_2 = CLSTMLayer(idim = hdim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_2', isOrtho = isOrtho, bptt_trunc = bptt_trun)
		if unit == "NGRU":
			self.hidden_1 = CGRULayerLNWN(idim = edim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_1', ln = ln, wn = wn, isOrtho = isOrtho, bptt_trunc = bptt_trun)
			self.hidden_2 = CGRULayerLNWN(idim = edim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_2', ln = ln, wn = wn, isOrtho = isOrtho, bptt_trunc = bptt_trun)

	def encode(self, inputs, isTrain):
		"""
		>>
		> inputs : the inputs 3D matrix or 2D matrix when predict
		> isTrain : if this encode for train
		< return : return the encoded hidden output
		"""
		if isTrain:
			x_dropout = self.get_dropMatrix((3, self.minibatch, self.edim), 1 - self.dp, self.inverted, isTrain) # <3 gate, batch, edim>
			x2_dropout = self.get_dropMatrix((3, self.minibatch, self.hdim), 1 - self.dp, self.inverted, isTrain) # <3 gate, batch, hdim>
			h_dropout = self.get_dropMatrix((2, 3, self.minibatch, self.hdim), 1 - self.dp, self.inverted, isTrain) # <level, 3 gate, batch, hdim>
		else:
			dropout = self.get_dropMatrix((1, 1), 1 - self.dp, self.inverted, False)
			x_dropout = dropout
			x2_dropout = dropout
			h_dropout = [dropout, dropout]

		h1_out, _ = theano.scan(fn = self.hidden_1.activate,
							 	sequences = inputs,
							 	outputs_info = [T.zeros((self.minibatch, self.hdim))],
								non_sequences = [x_dropout, h_dropout[0]],
							 	truncate_gradient = self.bptt_trun,
							 	allow_gc = True)

		h2_out, _ = theano.scan(fn = self.hidden_2.activate,
							 	sequences = h1_out,
							 	outputs_info = [T.zeros((self.minibatch, self.hdim))],
								non_sequences = [x2_dropout, h_dropout[1]],
							 	truncate_gradient = self.bptt_trun,
							 	allow_gc = True)

		return [h1_out, h2_out]

	def get_dropMatrix(self, shape, prob, inverted = True, isTrain = True):
		"""
		>>
		> shape : shape of the dropout matrix
		> prob : prob that set the value as 1, prob = 1 - dropout_prob
		> inverted : is it inverted dropout
		> isTrain : is it for training
		"""
		if isTrain:
			if inverted:
				drop_matrix = self.trng.binomial(shape, p = prob, n = 1, dtype = theano.config.floatX) / prob
			else:
				drop_matrix = self.trng.binomial(shape, p = prob, n = 1, dtype = theano.config.floatX)
		else:
			if inverted:
				drop_matrix = theano.shared(np.array([1.] * 3).astype(theano.config.floatX))
			else:
				drop_matrix = theano.shared(np.array([prob] * 3).astype(theano.config.floatX))

		return drop_matrix

	def resetBatch(self, newbatch):
		"""
		>>
		> newbatch : the new batch
		< return : 
		"""
		self.minibatch = newbatch
		self.hidden_1.resetBatch(newbatch)
		self.hidden_2.resetBatch(newbatch)

	def saveModel(self, dirname):
		"""
		>>
		> e_n : the num of the epoch
		> dirname : the directory name
		< return : 
		"""
		self.hidden_1.saveModel(dirname)
		self.hidden_2.saveModel(dirname)

	def readModel(self, filename):
		"""
		>>
		> filename : the filename that contains the model
		< return :
		"""
		self.hidden_1.readModel(filename)
		self.hidden_2.readModel(filename)

class CDecoder(object):
	""" the Decoder class """
	def __init__(self, edim, hdim, tvsize, minibatch, en2de, name, inverted, dropout, isOrtho = True, ln = False, wn = False, unit = "sRNN", beamsize = 12, bptt_trun = -1):
		"""
		>>
		> edim : the embedding dim
		> hdim : the hidden dim
		> tvsize : the target language vocabulary size
		> minibatch : the batch size
		> en2de : indicate that initalize the S0 with average or the last state of bac_enc
		> name : the name of the Decoder
		> inverted : is the dropout is inverted
		> dropout : the hidden dropout prob
		> isOrtho : is the matrix is ortho matrix
		> ln : bool type, layer normalization or not
		> wn : bool type, weight normalization or not
		> unit : the hidden unit, sRNN or GRU or LSTM
		> beamsize : the beamsize, used when predict
		> bptt_trun : used in scan
		< return : 
		"""
		self.edim = edim
		self.hdim = hdim
		self.tvsize = tvsize
		self.minibatch = minibatch
		self.en2de = en2de
		self.name = name
		self.Winit_h1_name = name + '_Winit_h1'
		self.Winit_h2_name = name + '_Winit_h2'
		self.b_h1_name = name + '_b_h1'
		self.b_h2_name = name + '_b_h2'
		self.W_t1_name = name + '_W_t1'
		self.b_t1_name = name + '_b_t1'
		self.W_t2_name = name + '_W_t2'
		self.b_t2_name = name + '_b_t2'
		self.W_t3_name = name + '_W_t3'
		self.b_t3_name = name + '_b_t3'
		self.inverted = inverted
		self.dp = dropout
		self.beamsize = beamsize
		self.bptt_trun = bptt_trun

		self.trng = RandomStreams(1234)

		# params
		# the Winit parameters are unique cuz they not belong to any layer, so i decide to create the parameters here, the EncoderDecoder-level.
		if en2de == 0:
			Winit_h1 = getThreshold(hdim, hdim) * np.random.randn(hdim, hdim).astype(theano.config.floatX)
			Winit_h2 = getThreshold(hdim, hdim) * np.random.randn(hdim, hdim).astype(theano.config.floatX)
		else:
			Winit_h1 = getThreshold(2 * hdim, hdim) * np.random.randn(2 * hdim, hdim).astype(theano.config.floatX)
			Winit_h2 = getThreshold(2 * hdim, hdim) * np.random.randn(2 * hdim, hdim).astype(theano.config.floatX)
		b_h1 = getThreshold(0, 0) * rnd.randn(hdim).astype(theano.config.floatX)
		b_h2 = getThreshold(0, 0) * rnd.randn(hdim).astype(theano.config.floatX)

		# params that used in initialize the decoder's state
		self.Winit_h1 = theano.shared(value = Winit_h1, name = self.Winit_h1_name, borrow = True)
		self.Winit_h2 = theano.shared(value = Winit_h2, name = self.Winit_h2_name, borrow = True)
		self.b_h1 = theano.shared(value = b_h1, name = self.b_h1_name, borrow = True)
		self.b_h2 = theano.shared(value = b_h2, name = self.b_h2_name, borrow = True)

		# params that used in output-level
		# t_j = tanh(s_j * W_t1 + y_j-1 * W_t2 + c_j * W_t3)
		# p(y_j | s_j, y_j-1, c_j) = softmax(t_j * W_o)
		"""
		W_t1 = getThreshold(hdim, edim) * np.random.randn(hdim, edim).astype(theano.config.floatX)
		b_t1 = getThreshold(0, 0) * rnd.randn(edim).astype(theano.config.floatX)
		W_t2 = getThreshold(edim, edim) * np.random.randn(edim, edim).astype(theano.config.floatX)
		b_t2 = getThreshold(0, 0) * rnd.randn(edim).astype(theano.config.floatX)
		W_t3 = getThreshold(2 * hdim, edim) * np.random.randn(2 * hdim, edim).astype(theano.config.floatX)
		b_t3 = getThreshold(0, 0) * rnd.randn(edim).astype(theano.config.floatX)

		self.W_t1 = theano.shared(value = W_t1, name = self.W_t1_name, borrow = True)
		self.b_t1 = theano.shared(value = b_t1, name = self.b_t1_name, borrow = True)
		self.W_t2 = theano.shared(value = W_t2, name = self.W_t2_name, borrow = True)
		self.b_t2 = theano.shared(value = b_t2, name = self.b_t2_name, borrow = True)
		self.W_t3 = theano.shared(value = W_t3, name = self.W_t3_name, borrow = True)
		self.b_t3 = theano.shared(value = b_t3, name = self.b_t3_name, borrow = True)
		"""
		self.params = [self.Winit_h1, self.b_h1, self.Winit_h2, self.b_h2]

		# hidden layer
		CErrorMsg.showErrExMsg((unit in ["sRNN", "GRU", "LSTM", "NGRU"]), "The hidden unit is wrong!")
		if unit == "sRNN":
			self.hidden_1 = CRNNLayer(idim = edim + hdim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_1', isOrtho = isOrtho, bptt_trunc = bptt_trun)
			self.hidden_2 = CRNNLayer(idim = hdim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_2', isOrtho = isOrtho, bptt_trunc = bptt_trun)
		if unit == "GRU":
			self.hidden_1 = CGRULayer(idim = edim + hdim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_1', isOrtho = isOrtho, bptt_trunc = bptt_trun)
			self.hidden_2 = CGRULayer(idim = hdim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_2', isOrtho = isOrtho, bptt_trunc = bptt_trun)
		if unit == "LSTM":
			print '[Warn] Not Done Yet!'
			exit()
			self.hidden_1 = CLSTMLayer(idim = edim + hdim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_1', isOrtho = isOrtho, bptt_trunc = bptt_trun)
			self.hidden_2 = CLSTMLayer(idim = hdim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_2', isOrtho = isOrtho, bptt_trunc = bptt_trun)
		if unit == "NGRU":
			self.hidden_1 = CGRULayerLNWN(idim = edim + hdim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_1', ln = ln, wn = wn, isOrtho = isOrtho, bptt_trunc = bptt_trun)
			self.hidden_2 = CGRULayerLNWN(idim = edim, hdim = hdim, minibatch = minibatch, name = self.name + '.hidden_2', ln = ln, wn = wn, isOrtho = isOrtho, bptt_trunc = bptt_trun)

		# attention layer
		self.attention = CAttentionLayer(hdim, tvsize, minibatch, self.name + '.attention')

		# output layer
		self.output = COutputLayer(hdim, tvsize, minibatch, self.name + '.output')

	def get_dropMatrix(self, shape, prob, inverted = True, isTrain = True):
		"""
		>>
		> shape : shape of the dropout matrix
		> prob : prob that set the value as 1, prob = 1 - dropout_prob
		> inverted : is it inverted dropout
		> isTrain : is it for training
		"""
		if isTrain:
			if inverted:
				drop_matrix = self.trng.binomial(shape, p = prob, n = 1, dtype = theano.config.floatX) / prob
			else:
				drop_matrix = self.trng.binomial(shape, p = prob, n = 1, dtype = theano.config.floatX)
		else:
			if inverted:
				drop_matrix = theano.shared(np.array([1.] * 3).astype(theano.config.floatX))
			else:
				drop_matrix = theano.shared(np.array([prob] * 3).astype(theano.config.floatX))

		return drop_matrix

	def link_forTrain(self, xdp, x2dp, hdp):
		"""
		>>link the network vertically and go through tgt word one by one, this for training not predicting
		< return : return the outputs that not gone through softmax yet
		"""
		def recurrence(y_1, h1_pre, h2_pre, ht_pre, en_h, broadcast_mask, xdp_, x2dp_, hdp_):
			# input-feedding
			y_1_con = T.concatenate((y_1, ht_pre), axis = 1) # y_1 and ht_pre are 2D matrix, shape : <minibatch, edim + hdim>

			# hidden layer 1
			h1_out = self.hidden_1.activate(y_1_con, h1_pre, xdp_, hdp_[0]) # <minibatch, hdim>

			# hidden layer 2
			h2_out = self.hidden_2.activate(h1_out, h2_pre, x2dp_, hdp_[1]) # <minibatch, hdim>

			# attention layer
			ht_out = self.attention.activate(h2_out, en_h, broadcast_mask) # <minibatch, hdim>, <minibatch, 2 * hdim>

			# output layer
			#o_out = self.output.activate(ht_out) # <minibatch, tvsize>
			#t_j = T.tanh(T.dot(ht_out, self.W_t1) + self.b_t1 + T.dot(y_1, self.W_t2) + self.b_t2 + T.dot(ct, self.W_t3) + self.b_t3) # <minibatch, edim>

			return [h1_out, h2_out, ht_out]
		[h1_out, h2_out, ht_out], _ = theano.scan(fn = recurrence,
												  sequences = self.inputs,
												  outputs_info = [self.h1_last, self.h2_last, np.zeros((self.minibatch, self.hdim), dtype = theano.config.floatX)],
												  non_sequences = [self.en_inputs, (T.ones((self.en_inputs.shape[0], self.en_inputs.shape[1], (self.en_inputs.shape[2] / 2)), dtype = theano.config.floatX)) * self.aMask, xdp, x2dp, hdp],
												  truncate_gradient = self.bptt_trun,
												  allow_gc = True)
		return ht_out# t_j

	def decode_forTrain(self, en_inputs, inputs, h1_last, h2_last, gold, aMask, oMask):
		"""
		>>
		> en_inputs : the encoder's last hidden outputs
		> inputs : the tgt word seq's 3D matrix
		> h1_last : 1st hidden layer's final output of the encoder
		> h2_last : 2cn hidden layer's final output of the encoder
		> gold : the gold indices at decoder side
		> aMask : mask that filt attention broadcast
		> oMask : mask that filt final outputlayer's output
		< return : return the final output and total word num
		"""
		self.en_inputs = en_inputs
		self.inputs = inputs
		self.h1_last = T.tanh(T.dot(h1_last, self.Winit_h1) + self.b_h1)
		self.h2_last = T.tanh(T.dot(h2_last, self.Winit_h2) + self.b_h2)
		self.aMask = aMask

		x_dropout = self.get_dropMatrix((3, self.minibatch, self.edim + self.hdim), 1 - self.dp, self.inverted, True)
		x2_dropout = self.get_dropMatrix((3, self.minibatch, self.hdim), 1 - self.dp, self.inverted, True)
		h_dropout = self.get_dropMatrix((2, 3, self.minibatch, self.hdim), 1 - self.dp, self.inverted, True)

		h_out = self.link_forTrain(x_dropout, x2_dropout, h_dropout)

		""" prepare gold indices """
		gold_flat = gold.flatten()
		gold_idx_flat = T.arange(gold_flat.shape[0]) * self.tvsize + gold_flat

		""" calc the cost """
		tmp_o_1 = h_out.reshape((gold.shape[0] * gold.shape[1], self.hdim))
		tmp_o_2 = -T.log(T.nnet.softmax(self.output.activate(tmp_o_1)))
		tmp_o_3 = tmp_o_2.flatten()[gold_idx_flat]
		cost = (tmp_o_3.reshape((gold.shape[0], gold.shape[1])) * oMask)

		return cost.mean()

	def link_forPredict(self, dp):
		"""
		>>link the network vertically and go through tgt word one by one, this for training not predicting
		< return : return the outputs that not gone through softmax yet
		"""
		def recurrence(idx, h1_pre, h2_pre, ht_pre, en_h, broadcast_mask, dp_):
			y_1 = self.embedding.activate(idx) # y_1 is a vector

			# input-feedding
			y_1_con = T.concatenate((y_1, ht_pre), axis = 1) # y_1 and ht_pre are vector, shape : <edim + hdim>

			# hidden layer 1
			h1_out = self.hidden_1.activate(y_1_con, h1_pre, dp_, dp_) # <1, hdim>, h1_out is a row, it means it is a matrix and shape is <1, hdim>

			# hidden layer 2
			h2_out = self.hidden_2.activate(h1_out, h2_pre, dp_, dp_) # <1, hdim>

			# attention layer
			ht_out = self.attention.activate(h2_out, en_h, broadcast_mask) # <1, hdim>

			# output
			# t_j = T.tanh(T.dot(ht_out, self.W_t1) + self.b_t1 + T.dot(y_1, self.W_t2) + self.b_t2 + T.dot(ct, self.W_t3) + self.b_t3)

			# output layer
			o_out = self.output.activate(ht_out) # <1, tvsize>

			# extract the word that have highest prob
			prob = T.nnet.softmax(o_out)
			output = -T.log(prob)
			#nidx = output.argmin()
			#cost = output.min(keepdims=True)

			return [prob, output, h1_out, h2_out, ht_out]#, theano.scan_module.until(T.eq(1, nidx))
		"""
		[prob, output, h1_out, h2_out, ht_out], _ = theano.scan(fn = recurrence,
																sequences = self.y_idx,
																outputs_info = [None, None, self.h1_last, self.h2_last, self.ht_init],
																non_sequences = [self.en_inputs, (T.ones((self.en_inputs.shape[0], self.en_inputs.shape[1], (self.en_inputs.shape[2] / 2))))],
																truncate_gradient = self.bptt_trun,
																allow_gc = True,
																n_steps = 1)

		return [prob, output, h1_out, h2_out, ht_out]
		"""

		return recurrence(self.y_idx, self.h1_last, self.h2_last, self.ht_init, self.en_inputs, (T.ones((self.en_inputs.shape[0], self.en_inputs.shape[1], (self.en_inputs.shape[2] / 2)))), dp)

	def decode_forPredict(self, y, en_inputs, h1_last, h2_last, ht_init, de_embedding):
		"""
		>>
		"""
		self.y_idx = y
		self.en_inputs = en_inputs
		self.h1_last = h1_last # this already done outside, don't need here -> T.tanh(T.dot(h1_last, self.Winit_h1) + self.b_h1)
		self.h2_last = h2_last # this already done outside, don't need here -> T.tanh(T.dot(h2_last, self.Winit_h2) + self.b_h2)
		self.ht_init = ht_init
		self.embedding = de_embedding

		dropout = self.get_dropMatrix((1, 1), 1 - self.dp, self.inverted, False)

		prob, cost, h1_out, h2_out, ht_out = self.link_forPredict(dropout)

		return [prob, cost, h1_out, h2_out, ht_out]

	def calcCost(self, y, y_gold, oMask):
		"""
		>>
		> y : the output of the output layer
		> y_gold : the expected indices, i.e. the target index matrix
		> oMask : the mask of the output, and this should be a 2D matrix, however, because this func used in theano.scan, so oMask here is a vector
		"""
		tmp_cost = T.sum(y[T.arange(y.shape[0]), y_gold] * oMask)
		return tmp_cost

	def resetBatch(self, newbatch):
		"""
		>>
		> newbatch : the new batch
		< return : 
		"""
		self.minibatch = newbatch
		self.hidden_1.resetBatch(newbatch)
		self.hidden_2.resetBatch(newbatch)
		self.attention.resetBatch(newbatch)
		self.output.resetBatch(newbatch)

	def saveModel(self, dirname):
		"""
		>>
		> e_n : the num of the epoch
		> dirname : the directory name
		< return : 
		"""
		filename = dirname + self.name + ".npz"

		print '    >>[Debug] Saving the model %s.' % filename
		np.savez(filename, Winit_h1 = self.Winit_h1.get_value(),
							Winit_h2 = self.Winit_h2.get_value(),
                            b_h1 = self.b_h1.get_value(),
                            b_h2 = self.b_h2.get_value())
		print '    >>[Debug] Model is saved...'

		# layer's params
		self.hidden_1.saveModel(dirname)
		self.hidden_2.saveModel(dirname)
		self.attention.saveModel(dirname)
		self.output.saveModel(dirname)

	def readModel(self, filename):
		"""
		>>
		> filename : the filename that contains the model
		< return :
		"""
		finit = filename + self.name + ".npz"

		print '    >>[Debug] Reading the model from file : %s' % finit
		model = np.load(finit)
		self.Winit_h1.set_value(model['Winit_h1'])
		self.Winit_h2.set_value(model['Winit_h2'])
		self.b_h1.set_value(model['b_h1'])
		self.b_h2.set_value(model['b_h2'])
		print '    >>[Debug] Model is read...'

		# layer's params
		self.hidden_1.readModel(filename)
		self.hidden_2.readModel(filename)
		self.attention.readModel(filename)
		self.output.readModel(filename)
