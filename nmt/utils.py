# rebuild project
# ambyer
# 2017.06.29 - 2017.06.29

# this file contains some utilities
# most of the functions below are come from pre-project's utils.py

import numpy as np
import json
import random
import sys
from datetime import datetime
import copy
from ErrorMsg import *


def show_Info(args, s, t):
	"""
	>>show the info of the configer
	> args : the argparse, is a model of python
	> s : the size of source vocabulary
	> t : the size of target vocabulary
	< return : 
	"""
	print "|------------------------------------------------------|"
	print "|                    Configuration                     |"
	print "|------------------------------------------------------|"
	
	print "|--%-10s = %-7d[0:train, 1:continue, 2:predict]" % ('mode', args.mode)
	print "|--%-10s = %-7d[the hidden dimensionality]" % ('hdim', args.hdim)
	print "|--%-10s = %-7d[the dimensionality of the word representation]" % ('edim', args.edim)
	print "|------------------------------------------------------|"
	if args.mode == 0 or args.mode == 1:
		print "|--%-10s = %-7d[the size of minibatch]" % ('minibatch', args.minibatch)
		print "|--%-10s = %-7d[how many batchs will be extracted at one time]" % ('batch_win', args.batchWin)
		print "|--%-10s = %-7d[the training epoch]" % ('epoch', args.epochs)
		print "|--%-10s = %-7.3f[used in gradient clip]" % ('clip', args.clip)
		print "|--%-10s = %-7.3f[the learning rate]" % ('lrate', args.lrate)
		print "|--%-10s = %-7d[the Optimum method [0:SGD, 1:ADADELTA, 2:ADAM]]" % ('method', args.method)
		print "|--%-10s = %-7d[whether shuffle the training set or not [0:No shuffle no sort, 1:Shuffle and Sort, 2:Shuffle but not Sort]]" % ('shuffle', args.shuff)
		print "|--%-10s = %-7d[save a model every n epoch]" % ('savenep', args.savenep)
		print "|--%-10s = %-7d[size of source vocabulary]" % ('svsize', s)
		print "|--%-10s = %-7d[size of target vocabulary]" % ('tvsize', t)
		if args.mode == 1:
			print "|--%-10s = %-7d[continue to train the model from n-th model]" % ('from_n', args.fromn)
			print "|--%-10s = %s [the model that start from]" % ('model', args.model)
	else:
		print "|--%-10s = %-7d[the size of beamsearch]" % ('beamsize',args.beamsize)
		print "|--%-10s = %s [the set that need to be predicted]" % ('tstfile', args.tst)
	print "|------------------------------------------------------|"
	print "|--%-10s = %s [source training set]" % ('srcfile', args.src)
	print "|--%-10s = %s [target training set]" % ('tgtfile', args.tgt)
	print "|--%-10s = %s [source lang vocabulary]" % ('swvocab', args.swvocab)
	print "|--%-10s = %s [target lang vocabulary]" % ('twvocab', args.twvocab)
	print "|------------------------------------------------------|"
	print "|                 Configuration  End                   |"
	print "|------------------------------------------------------|"

def show_Info_v2(arg_dic):
	date = datetime.now().strftime("%Y.%m.%d - %H:%M:%S")
	print "|------------------------------------------------------|"
	print "|                    Configuration                     |"
	print "|------------------------------------------------------|"
	print "|--%-20s = %-27s--|" % ('DATE AND TIME', date)
	tmp = sorted(arg_dic.iteritems(), key = lambda d: d[0], reverse = False)
	for key, val in tmp:
		print "|--%-20s = %-27s--|" % (key, val)
	print "|------------------------------------------------------|"
	print "|                 Configuration  End                   |"
	print "|------------------------------------------------------|"

def check_options_validation(args):
	CErrorMsg.showErrExMsg((args.mode in [0, 1, 2]), "The mode is uncorrect, please check!")
	CErrorMsg.showErrExMsg((args.method in [0, 1, 2]), "The method is uncorrect, please check!")
	CErrorMsg.showErrExMsg((args.shuff in [0, 1, 2]), "The shuff is uncorrect, please check!")
	CErrorMsg.showErrExMsg((args.unit in ['sRNN', 'GRU', 'LSTM', 'NGRU']), "The unit is uncorrect, please check!")
	CErrorMsg.showErrExMsg((args.ensemble_method in [0, 1]), "The ensemble method is uncorrect, please check!")
	CErrorMsg.showErrExMsg((args.en2de in [0, 1]), "The en2de is uncorrect, please check!")
	CErrorMsg.showErrExMsg((args.validMethod in [0, 1], "The valid Method is not correct!"))

def getThreshold(idim, odim):
	"""
	>>
	> idim : the input dimensionality
	> odim : the output dimensionality
	< return : return the threshold
	"""
	if idim == 0 and odim == 0:
		return 0
	else:
		return np.sqrt(6. / (idim + odim))

def strIdx2intIdx(lines):
	"""
	>>
	> lines : the index matrix that typed string
	< return : 
	"""
	idx = []
	for ln in lines:
		units = ln.strip().split(' ')
		tmp_r = []
		for unit in units:
			if unit == '\n' or unit == '':
				print 'wrongs'
				continue
			tmp_r.append(int(unit))
		idx.append(tmp_r)
	return idx

def str2idx_zipped(lines, maxlen):
	"""
	>>
	> lines : the lines that every elem of it is zipped. Although the lines is idx, it's type is string
	> maxlen : every sentence that longer than maxlen will be filtered
	< return : list that zipped the idx of lines, len of the list, source words num, target words num
	"""
	idx_lines = []
	swords_n = 0
	twords_n = 0
	for l in lines:
		sunits = l[0].strip().split(' ')
		tunits = l[1].strip().split(' ')
		stmp = []
		ttmp = []
		for sunit in sunits:
			if sunit in ['', '\n', '\r']:
				print '[Warn] meet illegal character'
				continue
			stmp.append(int(sunit))
		for tunit in tunits:
			if tunit in ['', '\n', '\r']:
				print '[Warn] meet illegal character'
				continue
			ttmp.append(int(tunit))
		if len(stmp) > maxlen or len(ttmp) > maxlen:
			continue
		else:
			idx_lines.append([stmp] + [ttmp])
			swords_n += len(stmp)
			twords_n += len(ttmp)
	return idx_lines, len(idx_lines), swords_n, twords_n

def get_rindex(sub_arr):
	"""
	>>
	> sub_arr : a mini-batch sentences that typed int
	< return : the indexes and rindexes
	"""
	rindexes = []
	for arr in sub_arr:
		rindex = []
		rindex = arr[: : -1]
		rindexes.append(rindex)
	return rindexes

def get_maxlen(sub_arr, isSrc = True):
	"""
	>>
	> sub_arr : a mini-batch sentences that typed int
	> isSrc : if true then return the last_pos, else just return maxlen
	< return : the maxlen in a mini-batch sentences and the vector contains the last used word's pos
	"""
	sent_lens = [len(sent) for sent in sub_arr]
	maxlen = max(length for length in sent_lens)
	#last_pos = [(length - (maxlen + 1)) for length in sent_lens]
	last_pos = [(length - 1) for length in sent_lens]
	if isSrc:
		return [maxlen, last_pos]
	else:
		return maxlen

def get_N(modelname):
	"""
	>>get the model's number, that refers to the model is epoch n's model
	> modelname : the name of the model, it contains the number of the epoch
	< return : return the number of the model, indicate that which epoch's model
	"""
	ii = modelname.find('-')
	jj = modelname.rfind('-')
	nn = int(modelname[ii + 1: jj])
	return nn

def set_pad(indexes, maxlen, rindexes = None):
	"""
	>>
	> sub_arr : a mini-batch sentences that typed int
	> maxlen : the longest sentence's len in a mini-batch sentences
	< return : the indexes and rindexes that already add the <pad> tag
	"""
	for ii in range(len(indexes)):
		if len(indexes[ii]) < maxlen:
			for jj in range(maxlen - len(indexes[ii])):
				indexes[ii].append(3)
				if rindexes != None:
					rindexes[ii].append(3)

def set_tgt_pad(indices, maxlen):
	"""
	>>
	> indices : the tgt indices seq
	> maxlen : the max-length of the tgt seq
	"""
	for ii in range(len(indices)):
		if len(indices[ii]) < maxlen:
			indices[ii].append(1)
			for jj in range(maxlen - len(indices) - 1):
				indices[ii].append(3)

def get_Hout_3DMask(ein, minibatch, hdim, maxlen):
	"""
	>>
	> ein : the src inputs, is a 2d matrix
	> minibatch : batch size
	> hdim : size of hidden layer
	> maxlen : the max seq_len
	"""
	_3DMask = np.zeros((maxlen, minibatch, hdim))
	for ii in range(minibatch):
		for jj in range(maxlen):
			if ein[ii][jj] == 3:
				break
			_3DMask[jj][ii] = 1
	return _3DMask

def get_Hout_2DMask(ein, minibatch, maxlen):
	"""
	>>
	> ein : the src inputs, is a 2d matrix
	> minibatch : batch size
	> maxlen : the max seq_len
	< return : the 2D mask
	"""
	_2DMask = np.not_equal(ein, 3).astype("int32")
	return _2DMask

def get_Oout_2DMask(y, minibatch, lp):
	"""
	>>
	> y : the target seqs
	> minibatch : the batch size
	> lp : the lastpostions of the tgt index matrix
	"""
	con = np.ones((minibatch, 1), dtype = np.int) * 3
	ty = np.array(y)
	ty = np.concatenate((ty[:, 1:], con), axis = 1)
	ty[np.arange(minibatch), lp] = 1 # we can locat the <eos> tags directly by using lp parameter, so no need to circle

	_2DMask = np.not_equal(ty, 3).astype("int32")
	return _2DMask, ty

def get_Oout_3DMask(y, minibatch, tvsize, maxlen):
	"""
	>>
	> y : the tgt sentences' indexes
	> minibatch : the minibatch
	> tvsize : size of the tgt vocabulary
	> maxlen : the longest sentence's len in a mini-batch sentences
	> return : get the 3DMask that used in loss function
	"""
	con = np.ones((minibatch, 1), dtype = np.int) * 3
	ty = np.array(y)
	ty = ty[:, 1:]
	ty = np.concatenate((ty, con), axis = 1)
	_3DMask = np.zeros((maxlen, minibatch, tvsize))
	for ii in range(minibatch):
		firstPadFlag = True
		for jj in range(maxlen):
			if ty[ii][jj] == 3:
				if firstPadFlag: # if meet the first <pad>, we have to change it to <eos>
					ty[ii][jj] = 1
					firstPadFlag = False
				else:
					break
			_3DMask[jj][ii][ty[ii][jj]] = 1
	return _3DMask

def get_Att_2DMask(h_2DMask, lp):
	"""
	>>
	"""
	_2DMask = np.not_equal(h_2DMask, 1).astype("int32") * float("-inf")
	_2DMask += h_2DMask
	return _2DMask

def read_set(filename):
	"""
	>>read the set from the file
	> filename : the filename or filedir+filename
	> return : return a list that contains all lines and a num that indecate the num of the lines
	"""
	lines = []
	line_n = 0
	try:
		with open(filename, "r") as f:
			lines = f.readlines()
			line_n = len(lines)
			f.close()
	except:
		print '[Error] Cannot open the file %s, No such file or directory!' % filename
		exit()
	return [lines, line_n]

def read_set_zipped(file1, file2):
	"""
	>>
	> file1 : the first file
	> file2 : the second file
	< return : a list that every elem of it is a tuple which zipped the corresponding line in file1 and file2, i.e. [(a1, b1), (a2, b2), ..., (an, bn)] and the len of list
	"""
	lines = []
	try:
		fin1 = open(file1, 'r')
		fin2 = open(file2, 'r')
		for s, t in zip(fin1, fin2):
			lines.append([s] + [t])
		fin1.close()
		fin2.close()
	except:
		print '[Error] Cannot open the file %s or %s!' % (file1, file2)
	return lines, len(lines)

def read_vocab_json(filename):
	"""
	>>read vocabulary from the file
	> filename : the file name
	> return the num of the vocab
	"""
	v_n = 0
	try:
		with open(filename, "r") as f:
			dic = json.load(f)
			v_n = len(dic)
			f.close()
	except:
		print '[Error] Cannot open the file %s, No such file or directory!' % filename
		exit()
	return v_n

def shuf_sort(src, win_len):
	"""
	>>
	> src : the source sentences
	> win_len : indicate how many batch in a window
	> minibatch : the batch num
	"""
	# make a shuffled list and zipped with sentence lens
	src_n = len(src)
	permu = np.random.permutation(src_n)
	src_zip = [(permu[i], len(src[permu[i]])) for i in range(src_n)]
	if src_n % win_len > 0:
		src_rest_zip = [(j, len(src[j])) for j in range(win_len - (src_n % win_len))] # the rest
		src_zip.extend(src_rest_zip)
	assert(((len(src_zip) % win_len) == 0))

	# sort list within a batch-window
	rslt = []
	for i in range((len(src_zip) / win_len)):
		segment = src_zip[i * win_len : (i + 1) * win_len]
		rslt = rslt + sorted(segment, key = lambda s : s[1])
	
	del permu
	del src_zip
	
	final = [src[r[0]][:] for r in rslt]
	return final

def shuf_sort_nmt(src, tgt, win_len):
	"""
	>>
	> src : the source sentences
	> tgt : the target sentences
	> win_len : indicate how many batch in a window
	> minibatch : the batch num
	"""
	# make a shuffled list and zipped with sentence lens
	src_n = len(src)
	permu = np.random.permutation(src_n)
	set_zip = [(permu[i], max(len(src[permu[i]]), len(tgt[permu[i]]))) for i in range(src_n)]
	if src_n % win_len > 0:
		set_rest_zip = [(j, max(len(src[j]), len(tgt[j]))) for j in range(win_len - (src_n % win_len))]  # the rest
		set_zip.extend(set_rest_zip)
	assert (((len(set_zip) % win_len) == 0))

	# sort list within a batch-window
	rslt = []
	for i in range((len(set_zip) / win_len)):
		segment = set_zip[i * win_len: (i + 1) * win_len]
		rslt = rslt + sorted(segment, key=lambda s: s[1])

	del permu
	del set_zip

	final_src = [src[r[0]][:] for r in rslt]
	final_tgt = [tgt[r[0]][:] for r in rslt]
	return (final_src, final_tgt)

def shuf_sort_nmt_zipped(lines, n, win_len):
	"""
	>>
	> lines : src and tgt that zipped in
	> n : len of lines, note that n % win_len = 0
	> win_len : windows size
	< return : the list that shuffled and sorted by length
	"""
	assert (n % win_len == 0)
	idx = range(n)
	random.shuffle(idx)
	idx_len_zip = [(i, max(len(lines[i][0]), len(lines[i][1]))) for i in idx]

	rslt = []
	for ii in range((n / win_len)):
		seg = idx_len_zip[ii * win_len: (ii + 1) * win_len]
		rslt = rslt + sorted(seg, key = lambda s : s[1])

	# if need to save the shuffle lines, can save the indices
	final = [copy.deepcopy(lines[r[0]]) for r in rslt]
	return final

def shuf_no_sort(src, win_len):
	"""
	>>
	> src : the source sentences
	> batch_n : indicate how many batch in a window
	> minibatch : the batch num
	"""
	# make a shuffled list and zipped with sentence lens
	src_n = len(src)
	permu = np.random.permutation(src_n)
	src_zip = [(permu[i], len(src[permu[i]])) for i in range(src_n)]
	if src_n % win_len > 0:
		src_rest_zip = [(j, len(src[j])) for j in range(win_len - (src_n % win_len))]  # the rest
		src_zip.extend(src_rest_zip)
	assert (((len(src_zip) % win_len) == 0))

	# sort list within a batch-window
	rslt = []
	for i in range((len(src_zip) / win_len)):
		segment = src_zip[i * win_len: (i + 1) * win_len]
		rslt = rslt + segment

	del permu
	del src_zip

	final = [src[r[0]][:] for r in rslt]
	return final

def shuf_no_sort_nmt(src, tgt, win_len):
	"""
	>>
	> src : the source sentences
	> tgt : the target sentences
	> batch_n : indicate how many batch in a window
	> minibatch : the batch num
	"""
	# make a shuffled list and zipped with sentence lens
	src_n = len(src)
	permu = np.random.permutation(src_n)
	set_zip = [(permu[i], max(len(src[permu[i]]), len(tgt[permu[i]]))) for i in range(src_n)]
	if src_n % win_len > 0:
		set_rest_zip = [(j, max(len(src[j]), len(tgt[j]))) for j in range(win_len - (src_n % win_len))]  # the rest
		set_zip.extend(set_rest_zip)
	assert (((len(set_zip) % win_len) == 0))

	# sort list within a batch-window
	rslt = []
	for i in range((len(set_zip) / win_len)):
		segment = set_zip[i * win_len: (i + 1) * win_len]
		rslt = rslt + segment

	del permu
	del set_zip

	final_src = [src[r[0]][:] for r in rslt]
	final_tgt = [tgt[r[0]][:] for r in rslt]
	return (final_src, final_tgt)

def shuf_nosort_nmt_zipped(lines, n, win_len):
	"""
	>>
	> lines : src and tgt that zipped in
	> n : len of lines, note that n % win_len = 0
	> win_len : windows size
	< return : the list that shuffled but not sorted by length
	"""
	assert (((n % win_len) == 0))
	idx = range(n)
	random.shuffle(idx)
	shuf_lines = [copy.deepcopy(lines[i]) for i in idx]

	return shuf_lines

def no_shuf_sort(src, src_n, win_len):
	"""
	>>
	> src : the source sentences
	> src_n : the num of the source training set
	> batch_n : indicate how many batch in a window
	> minibatch : the batch num
	"""
	if src_n % win_len > 0:
		tmp = src[0 : win_len - (src_n % win_len)]
		src.extend(tmp)
	return src

def no_shuf_sort_nmt(src, tgt, src_n, win_len):
	"""
	>>
	> src : the source sentences
	> tgt : the target sentences
	> src_n : the num of the source training set
	> batch_n : indicate how many batch in a window
	> minibatch : the batch num
	"""
	if src_n % win_len > 0:
		stmp = src[0 : win_len - (src_n % win_len)]
		ttmp = tgt[0 : win_len - (src_n % win_len)]
		src.extend(stmp)
		tgt.extend(ttmp)
	return src, tgt

def noshuf_nosort_nmt_zipped(lines, n, win_len):
	"""
	>>
	> lines : src and tgt that zipped in
	> n : len of lines, note that n % win_len = 0
	> win_len : windows size
	< return : the list that shuffled but not sorted by length
	"""
	final = [copy.deepcopy(l) for l in lines]
	return final

def fill_set(src, src_n, win_len):
	"""
	>>
	> src : the source set
	> src_n : the size of set
	> win_len : indicate how many sentences in a window (i.e. win_len = minibatch * n_batch)
	"""
	if src_n % win_len > 0:
		stmp = [src[s] for s in range(win_len - (src_n % win_len))]
		src.extend(stmp)

def fill_set_nmt(src, tgt, src_n, win_len):
	"""
	>>
	> src : the source set
	> tgt : the target set
	> src_n : the size of set
	> win_len : indicate how many sentences in a window (i.e. win_len = minibatch * n_batch)
	"""
	if src_n % win_len > 0:
		stmp = [src[s] for s in range(win_len - (src_n % win_len))]
		src.extend(stmp)
		ttmp = [tgt[t] for t in range(win_len - (src_n % win_len))]
		tgt.extend(ttmp)

def fill_set_nmt_zipped(lines, n, win_len):
	"""
	>>
	> lines : src and tgt that zipped in
	> n : len of lines
	> win_len : windows size
	< return : the filled list
	"""
	if n % win_len > 0:
		rand_idx = random.sample(range(n), (win_len - (n % win_len)))
		tmp = [lines[i] for i in rand_idx]
		lines.extend(tmp)

def write_to_file(filename, dic_filename, queue, bsize, nbest, print_prob):
	"""
	>>
	> filename : write the rslt seqs to this file
	> dic_filename : the dictionary's filename
	> queue : the queue contains the result sequences
	> bsize : the beam size
	> nbest : typed BOOL, and indicate whether output nbest seqs
	> print_prob : typed BOOL, when true this will print the prob of every word
	< return : the total word and total score
	"""
	def build_rdic(dic):
		rdic = {}
		for k, v in zip(dic.keys(), dic.values()):
			rdic[v] = k
		return rdic

	def idx2word(line, dic):
		words = [dic[k] for k in line]
		nline = ' '.join(words)
		return nline

	fdic = open(dic_filename, "rb")
	rdic = build_rdic(json.load(fdic))
	total_score = []
	total_words = []

	try:
		with open(filename, 'wb') as f:
			while True:
				tmp_seq = queue.get()
				if tmp_seq is None:
					break

				i, seq = tmp_seq[0], tmp_seq[1]
				idxs, score, probs = seq[0], seq[1], seq[2]

				#print '[Test] Translating %d sentences...' % i
				if nbest: # output format is : num ||| target_seq ||| the_score [ ||| prob_of_each_word]
					orders = np.argsort(score)
					for o in orders:
						prob_str = ""
						if print_prob:
							prob_str = " ||| " + ' '.join("{0:.4f}".format(prob) for prob in probs[o])
						print >> f, '{0} ||| {1} ||| {2}{3}'.format(i, idx2word(idxs[o][:-1] if idxs[o][-1] == 1 else idxs[o], rdic), score[o], prob_str)
						total_score.append(score[o])
						total_words.append(len(idxs[o]))
				else: # output format is : target_seq [ ||| prob_of_each_word]
					prob_str = ""
					if print_prob:
						prob_str = " ||| " + ' '.join("{0:.4f}".format(prob) for prob in probs)
					print >> f, '{0}{1}'.format(idx2word(idxs[:-1] if idxs[-1] == 1 else idxs, rdic), prob_str)

					total_score.append(score)
					total_words.append(len(idxs))

	except IOError, reason:
		sys.stderr.write('[Error] Some error occured when writing the result sequences to file %s:%s'.format(filename), reason)
		sys.exit(1)

	return total_score, total_words
