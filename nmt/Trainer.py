# rebuild project
# ambyer
# 2017.06.29 - 2017.06.29

# this file contains Trainer class and predicter class

import time
import os
import sys
from subprocess import Popen
from multiprocessing import Queue
from ErrorMsg import CErrorMsg
from utils import *

class CTrainer(object):
	""" the trainer class, this class accept a NN and train it """
	def __init__(self, arg_dict):
		"""
		>>
		> args : the argparse moudle, contains the configs
		< return : 
		"""
		self.srcfin		= arg_dict["src"]
		self.tgtfin		= arg_dict["tgt"]
		self.sdicfin	= arg_dict["swvocab"]
		self.tdicfin	= arg_dict["twvocab"]
		self.hdim		= arg_dict["hdim"]
		self.lrate		= arg_dict["lrate"]
		self.epochs		= arg_dict["epochs"]
		self.minibatch	= arg_dict["minibatch"]
		self.mode		= arg_dict["mode"]
		self.model		= arg_dict["model"]
		self.savenep	= arg_dict["savenep"]
		self.clip		= arg_dict["clip"]
		self.gc			= arg_dict["gc"]
		self.method		= arg_dict["method"]
		self.shuff		= arg_dict["shuff"]
		self.batchWin	= arg_dict["batchWin"]
		self.maxlen     = arg_dict["maxlen"]
		self.dispMethod = arg_dict["dispMethod"]
		self.svsize		= arg_dict["svsize"]
		self.tvsize		= arg_dict["tvsize"]
		self.valid      = arg_dict["vld"]
		self.use_valid  = arg_dict["use_validate"]
		self.dev        = arg_dict["dev"]
		self.rnum       = arg_dict["rnum"]
		self.dispFreq   = arg_dict["dispFreq"]
		self.saveFreq   = arg_dict["saveFreq"]
		self.validMethod= arg_dict["validMethod"]

	def trainNeuralNetwork(self, nmt):
		"""
		>>
		> nmt : the nmt class
		< return : 
		"""
		build_t = time.time()
		print '[Build] Building the network'
		nmt.buildNetwork()
		print '[Build] Done, costs %.2f sec' % (time.time() - build_t)

		fromn = 1
		if self.mode == 1:
			""" continue to train """
			print '[Debug] Reading the model...'
			nmt.readModel(self.model)
			fromn = get_N(self.model) + 1
			print '[Debug] Done, begin the training from epoch <%d>' % fromn

		print '[Prepare] Reading and preparing the training sets...'
		str_lines, str_lines_n = read_set_zipped(self.srcfin, self.tgtfin)
		print '[Prepare] Seen %d samples...\n[Prepare]convert the str to idx...' % (str_lines_n)
		idx_lines, idx_lines_n, src_word_n, tgt_word_n = str2idx_zipped(str_lines, self.maxlen)
		print '[Prepare] Done, %d samples(%d source words, %d target words) are seen after filter the sample that longer than %d' % (idx_lines_n, src_word_n, tgt_word_n, self.maxlen)

		win_size = self.batchWin * self.minibatch
		CErrorMsg.showErrExMsg((win_size <= idx_lines_n), 'The training sets is little than n mini-batches!')
		print '[Prepare] Filling the training set...'
		fill_set_nmt_zipped(idx_lines, idx_lines_n, win_size)
		idx_lines_n = len(idx_lines)
		print '[Prepare] Done, %d samples are seen after filling the set...' % idx_lines_n
		CErrorMsg.showErrExMsg((idx_lines_n % win_size) == 0, 'The training sets failed to filled!')
		print '[Prepare] Done, there are %s samples in a batch window' % win_size

		print '[Debug] Begin the training...'
		total_time = []
		global_step = 0
		for epoch in range(fromn - 1, self.epochs):
			e_beg = time.time()
			lines = []

			############################## shuffle the set ##############################
			CErrorMsg.showErrExMsg((self.shuff in [0, 1, 2]), 'The value of param shuff is unvalid!')
			if self.shuff == 1:
				print '[Shuff] Shuffling and sorting the training sets...'
				sh_beg = time.time()
				lines = shuf_sort_nmt_zipped(idx_lines, idx_lines_n, win_size)
				sh_end = time.time()
				print '[Shuff] End shuffling and sorting, cost %.3f sec...' % (sh_end - sh_beg)
			elif self.shuff == 2:
				print '[Shuff] Shuffling the training set, but no sort...'
				sh_beg = time.time()
				lines = shuf_nosort_nmt_zipped(idx_lines, idx_lines_n, win_size)
				sh_end = time.time()
				print '[Shuff] End shuffling without sort, cost %.3f sec...' % (sh_end - sh_beg)
			else:
				lines = noshuf_nosort_nmt_zipped(idx_lines, idx_lines_n, win_size)
			
			############################## train the model ##############################
			print '[Epoch] Begining of the epoch <%d>' % (epoch + 1)
			b_n_to = 0 # batch num in total set
			total_win = idx_lines_n / win_size
			total_batch = idx_lines_n / self.minibatch
			for ii in range(total_win): # traverse every window
				f_costs = 0.
				n_sents = win_size
				n_words = 0
				lines_win = lines[ii * win_size : (ii + 1) * win_size]
				n_tmp_words = [len(l[0]) + len(l[1]) for l in lines_win]
				n_words = sum(n_tmp_words) / 2

				if self.dispMethod == 0:
					print '\n   <-------------------------------------- Starting a window of batchs -------------------------------------->'
					print '    >>[Maxlen] The last sentence len(src, tgt) = (%d, %d)' % (len(lines_win[-1][0]), len(lines_win[-1][1]))
				t_win_beg = time.time()
				for jj in range(self.batchWin): # traverse every batch
					b_n_to += 1
					lines_sub = lines_win[jj * self.minibatch : (jj + 1) * self.minibatch]
					src = [sl[0] for sl in lines_sub]
					tgt = [tl[1] for tl in lines_sub]
					err = nmt.train(src, tgt)

					f_costs += err
					global_step += 1
					if self.dispMethod == 0:
						print '    >>[Cost] Epoch<%d / %d>::Batch<%d / %d>::{%d-%d} | {Win<%d / %d>, Batch<%d / %d>} >> Cost = %f' % ((epoch + 1), self.epochs, b_n_to, total_batch, (epoch + 1), b_n_to, ii + 1, total_win, jj + 1, self.batchWin, err)
					if self.validMethod == 1 and (global_step + 1) % self.saveFreq == 0:
						############################## saving the model ##############################
						cur_model = ""
						if (global_step + 1) % self.saveFreq == 0:
							modelfile = './model/epoch_%02d_gs_%d' % ((epoch + 1), (global_step + 1))
							if not os.path.exists(modelfile): # os.path.join('dir', 'sub_dir')
								os.makedirs(modelfile)
							cur_model = nmt.saveModel(global_step + 1, modelfile)

						############################## validating the model ##############################
						if self.use_valid and cur_model != "":
							print '[Valid] Using the external script to compute the BLEU'
							vbeg = time.time()
							p = Popen([self.use_valid, str(cur_model), self.valid, self.dev, str(self.rnum)])
							p.wait()
							print '[Valid] Done, cost %.3f sec totally' % (time.time() - vbeg)
				t_win_end = time.time()
				if self.dispMethod == 1:
					t_win = t_win_end - t_win_beg
					ave = f_costs / self.batchWin
					sps = n_sents / float(t_win)
					wps = n_words / float(t_win)
					print '    >>[Infor] Epoch<%d / %d>::Average Cost = %f, Time = %.3f sec, %.3f sents/s, %.3f words/s' % ((epoch + 1), self.epochs, ave, t_win, sps, wps)
			e_end = time.time()
			print '[Epoch] Ending of the epoch %d, and costs %.3f sec' % ((epoch + 1), (e_end - e_beg))
			total_time.append((e_end - e_beg))

			if self.validMethod == 0:
				############################## saving the model ##############################
				cur_model = ""
				if (epoch + 1) % self.savenep == 0:
					modelfile = './model/' + str(epoch + 1)
					if not os.path.exists(modelfile): # os.path.join('dir', 'sub_dir')
						os.makedirs(modelfile)
					cur_model = nmt.saveModel(epoch + 1, modelfile)

				############################## validating the model ##############################
				if self.use_valid and cur_model != "":
					print '[Valid] Using the external script to compute the BLEU'
					vbeg = time.time()
					p = Popen([self.use_valid, str(cur_model), self.valid, self.dev, str(self.rnum)])
					p.wait()
					print '[Valid] Done, cost %.3f sec totally' % (time.time() - vbeg)
		print '[Debug] End the training, costs %.3f sec totally, average %.3f sec per epoch ' % (sum(total_time), (sum(total_time) / len(total_time)))

class CPredicter(object):
	""" the predicter class, this class accept a NN and predict """
	def __init__(self, arg_dict):
		"""
		>>
		> args : the argparse moudle, contains the configs
		< return : 
		"""
		self.tstfin		= arg_dict["tst"]
		self.sdicfin	= arg_dict["swvocab"]
		self.tdicfin	= arg_dict["twvocab"]
		self.hdim		= arg_dict["hdim"]
		self.model		= arg_dict["model"]
		self.beamsize	= arg_dict["beamsize"]
		self.nbest      = arg_dict["nbest"]
		self.norm_alpha = arg_dict["norm_alpha"]
		self.dec_maxlen = arg_dict["dec_maxlen"]
		self.print_prob = arg_dict["print_prob"]
		self.svsize		= arg_dict["svsize"]
		self.tvsize		= arg_dict["tvsize"]

	def predictNeuralNetwork(self, nmt):
		"""
		>>
		> nmt : the nmt class
		< return : will predict test sequences and write it to a file
		"""
		nmt.buildTranslate()
		nmt.readModel(self.model)
		model_num = get_N(self.model)

		frs = self.tstfin + ".rslt"

		print '[Test] Begin testing of test-file %s using model<%d>...' % (frs, model_num)
		tst_str_lines, tst_n = read_set(self.tstfin)
		tst_lines = strIdx2intIdx(tst_str_lines)
		i_queue = Queue()
		o_queue = Queue()
		sentCnt = 0

		for n, line in enumerate(tst_lines):
			sentCnt += 1
			i_queue.put((n, line))
		i_queue.put(None)

		nmt.translate(i_queue, o_queue, self.nbest, self.norm_alpha, self.dec_maxlen)
		o_queue.put(None)

		total_score, total_words = write_to_file(frs, self.tdicfin, o_queue, self.beamsize, self.nbest, self.print_prob)
		score = sum(total_score)
		words = sum(total_words)
		print '    >>[PPL]  The perplexity is >> %.3f' % (np.exp(score / words))
		print '[Test] End the testing...'

