# rebuild project
# ambyer
# 2017.06.29 - 2017.06.29

# this file is the main func, the start of the whole system

import argparse

from nmt.nmt import CNMT
from nmt.Trainer import *
from nmt.ErrorMsg import CErrorMsg


if __name__ == "__main__":
	# options = local() # this can collect the vars that locally defined so far
	# defining the arguments
	parser = argparse.ArgumentParser(description = '<< Neural Machine Translation System >>')
	parser.add_argument('--src', metavar = 'sfile', dest = 'src', default = None, help = 'source training set (default: %(default)s)')
	parser.add_argument('--tgt', metavar = 'tfile', dest = 'tgt', default = None, help = 'target training set')
	parser.add_argument('--test', metavar = 'file', dest = 'tst', default = None, help = 'testing set')
	parser.add_argument('--valid', metavar = 'vfile', dest = 'vld', default = None, help = 'validation set')
	parser.add_argument('--dev', metavar = 'dfile', dest = 'dev', default = None, help = 'validation set')
	parser.add_argument('--rnum', metavar = 'rnum', dest = 'rnum', default = 4, help = 'the number of reference')
	parser.add_argument('--hdim', metavar = 'hdim', dest = 'hdim', type = int, default = 9, help = 'size of hidden level')
	parser.add_argument('--edim', metavar = 'edim', dest = 'edim', type = int, default = 7, help = 'size of embedding')
	parser.add_argument('--lrate', metavar = 'lrate', dest = 'lrate', type = float, default = 0.1, help = 'learning rate')
	parser.add_argument('--epochs', metavar = 'ep', dest = 'epochs', type = int, default = 3, help = 'size of epoch')
	parser.add_argument('--minibatch', metavar = 'mb', dest = 'minibatch', type = int, default = 3, help = 'size of minibatch')
	parser.add_argument('--swvocab', metavar = 'swvocab', dest = 'swvocab', help = 'source lang word vocabulary')
	parser.add_argument('--twvocab', metavar = 'twvocab', dest = 'twvocab', help = 'target lang word vocabulary')
	parser.add_argument('--mode', metavar = 'mode', dest = 'mode', type = int, default = 0, help = 'mode: [0: train, 1: continue, 2: exe]')
	parser.add_argument('--model', metavar = 'model', dest = 'model', default = None, help = 'model: there should be a model that already trained')
	parser.add_argument('--savenep', metavar = 'savenep', dest = 'savenep', type = int, default = 1, help = 'save the model every n epoch(s)')
	parser.add_argument('--fromn', metavar = 'fromn', dest = 'fromn', type = int, default = 1, help = 'start from the n(st/nd/rd/th) model')
	parser.add_argument('--beamsize', metavar = 'beamsize', dest = 'beamsize', type = int, default = 1, help = 'the beam size')
	parser.add_argument('--clip', metavar = 'clip', dest = 'clip', type = float, default = 5, help = 'the gradient clip')
	parser.add_argument('--method', metavar = 'method', dest = 'method', type = int, default = 0, help = 'the Optimum method of the deep learning [0:SGD, 1:ADADELTA, 2:ADAM]')
	parser.add_argument('--shuffle', metavar = 'shuff', dest = 'shuff', type = int, default = 0, help = 'whether shuffle the training set or not [0:No, 1:Yes, 2:Shuffle but no sort]')
	parser.add_argument('--unit', metavar = 'unit', dest = 'unit', default = 'GRU', help = 'the hidden unit [sRNN, GRU, LSTM]')
	parser.add_argument('--gc', metavar = 'gc', dest = 'gc', type = int, default = 3, help = 'used in prevence of gradient exploring')
	parser.add_argument('--batchWin', metavar = 'batchWin', dest = 'batchWin', type = int, default = 2, help = 'indicates that extract how many batches at one time')
	parser.add_argument('--maxlen', metavar = 'maxlen', dest = 'maxlen', type = int, default = 50, help = 'the maxlen of training set')
	parser.add_argument('--ensemble_method', metavar = 'ensemble_method', dest = 'ensemble_method', type = int, default = 0, help = 'the method of ensemble [0:ensemble the model, 1:ensemble the softmax score]')
	parser.add_argument('--dispFreq', metavar = 'dispFreq', dest = 'dispFreq', type = int, default = 13, help = 'the display frequent  (every n minibatch)')
	parser.add_argument('--saveFreq', metavar = 'saveFreq', dest = 'saveFreq' , type = int, default = 13, help = 'the save model frequent (every n minibatch)')
	parser.add_argument('--early_stop', metavar = 'early_stop', dest = 'early_stop' , type = int, default = 13, help = 'early stoping')
	parser.add_argument('--en2de', metavar = 'en2de', dest = 'en2de' , type = int, default = 0, help = 'the method that encoder\'s output to decoder [0: the last state of backward, 1: the average of the encoder\'s output]')
	parser.add_argument('--dispMethod', metavar = 'dispMethod', dest = 'dispMethod' , type = int, default = 0, help = 'display method [0: display the info every batch, 1: display the info every win_size]')
	parser.add_argument('--nbest', dest = 'nbest', action = 'store_true', help = 'whether print the nbest')
	parser.add_argument('--norm_alpha', metavar = 'norm_alpha', dest = 'norm_alpha' , type = float, default = 0.5, help = 'normalize scores by sentence length(when set 0, it means do not normalize)')
	parser.add_argument('--dec_maxlen', metavar = 'dec_maxlen', dest = 'dec_maxlen' , type = int, default = 30, help = 'the max length of the decoded sequences')
	parser.add_argument('--print_prob', dest = 'print_prob' , action = 'store_true', help = 'whether print the probabilities of each word')
	parser.add_argument('--use_validate', metavar = 'shfile', dest = 'use_validate', help = 'the shell script that test BLEU')
	parser.add_argument('--dropout', metavar = 'dropout', dest = 'dropout', type = float, default = 0.2, help = 'the dropout prob used in hidden layer')
	parser.add_argument('--inverted', dest = 'inverted' , action = 'store_true', help = 'is it inverted dropout')
	parser.add_argument('--layer_normalization', dest = 'layer_normalization' , action = 'store_true', help = 'use layer normalization or not')
	parser.add_argument('--weight_normalization', dest = 'weight_normalization' , action = 'store_true', help = 'use weight normalization or not')
	parser.add_argument('--close_ortho', dest = 'close_ortho', action = 'store_true', help = 'not set the weight matrix as ortho matrix')
	parser.add_argument('--validMethod', dest = 'validMethod', type = int, default = 0, help = '0:save model and test every epoch; 1:save every saveFreq and test it')
	args = parser.parse_args()
	arg_dic = vars(args)

	svsize = read_vocab_json(arg_dic["swvocab"])
	tvsize = read_vocab_json(arg_dic["twvocab"])
	arg_dic["svsize"] = svsize
	arg_dic["tvsize"] = tvsize
	arg_dic["nameNMT"] = 'myNMT'
	check_options_validation(args)

	if arg_dic["mode"] == 0 or arg_dic["mode"] == 1:
		foptions = open("./model/options.json", 'w')
		json.dump(arg_dic, foptions, indent = 2)
		foptions.close()
	elif arg_dic["mode"] == 2: # if this is for predicting, then change the arg_dic value, keep it equal to model's config_json
		fmodel = open("./model/options.json", 'r')
		dic = json.load(fmodel)
		fmodel.close()
		arg_dic["edim"] = dic["edim"]
		arg_dic["hdim"] = dic["hdim"]
		arg_dic["nameNMT"] = dic["nameNMT"]
		arg_dic["en2de"] = dic["en2de"]
		arg_dic["unit"] = dic["unit"]
		arg_dic["inverted"] = dic["inverted"]
		arg_dic["dropout"] = dic["dropout"]
		arg_dic["use_validate"] = dic["use_validate"]
		arg_dic["rnum"] = dic["rnum"]
		arg_dic["vld"] = dic["vld"]
		arg_dic["dev"] = dic["dev"]
		arg_dic["layer_normalization"] = dic["layer_normalization"]
		arg_dic["weight_normalization"] = dic["weight_normalization"]
		arg_dic["method"] = dic["method"]
		arg_dic["minibatch"] = 1

	if arg_dic["layer_normalization"] or arg_dic["weight_normalization"]:
		arg_dic["unit"] = "NGRU"
	show_Info_v2(arg_dic)
	nmt = CNMT(edim = arg_dic["edim"],
			   hdim = arg_dic["hdim"],
			   svsize = arg_dic["svsize"],
			   tvsize = arg_dic["tvsize"],
			   minibatch = arg_dic["minibatch"],
			   name = arg_dic["nameNMT"],
			   inverted = arg_dic["inverted"],
			   dropout = arg_dic["dropout"],
			   isOrtho = not arg_dic["close_ortho"],
			   en2de = arg_dic["en2de"],
			   unit = arg_dic["unit"],
			   lrate = arg_dic["lrate"],
			   method = arg_dic["method"],
			   ln = arg_dic["layer_normalization"],
			   wn = arg_dic["weight_normalization"],
			   beamsize = arg_dic["beamsize"],
			   maxlen = arg_dic["maxlen"],
               clip = arg_dic["clip"],
			   gc = arg_dic["gc"])

	print '[Debug] Constructed the CNMT class'
	if arg_dic["mode"] == 0 or arg_dic["mode"] == 1:
		trainer = CTrainer(arg_dic)
		trainer.trainNeuralNetwork(nmt)
	else:
		predicter = CPredicter(arg_dic)
		predicter.predictNeuralNetwork(nmt)
