# ambyer
# 2017.04.13

import sys
import json
import subprocess
import argparse

def build_rdic(dic):
	rdic = {}
	for k, v in zip(dic.keys(), dic.values()):
		rdic[v] = k
	return rdic

def postpro(fin, fout, fdic, eva):
	rdic = build_rdic(json.load(fdic))
	for line in fin:
		if line[-4 : -2] == ' 1':
			#line = line[1 : -2] # handle with <eos> tag. there is a '\n' at the end of the line
			line = line[1 : -5] # handle without <eos> tag.
		else:
			line = line[1 : -2]
		#print '>>', line
		subs = line.strip().split(', ')
		nline = ""
		for sub in subs:
			if sub == ' ' or sub == '':
				continue
			#print '>', line
			nline = nline + str(rdic[int(sub)]) + ' '
		#if eva == 0:
			#nline = line + ' |||| ' + nline
		print >> fout, nline[:-1]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'postprocess will let the indices change to the words...')
	parser.add_argument('--tst', metavar = 'tfile', dest = 'tst', default = None, help = 'the testing file')
	parser.add_argument('--ref', metavar = 'rfile', dest = 'ref', default = None, help = 'the reference file of the testing file')
	parser.add_argument('--dic', metavar = 'dic', dest = 'dic', default = None, help = 'the dictionary')
	parser.add_argument('--rnum', metavar = 'rnum', dest = 'rnum', type = int, default = 1, help = 'the num of the reference')
	parser.add_argument('--eva', metavar = 'eva', dest = 'eva', type = int, default = 0, help = 'evaluate or not [0:No, 1:Yes]')
	args = parser.parse_args()
	ftst = args.tst
	fref = args.ref
	fdic = args.dic
	tout = args.tst + ".out"
	rnum = args.rnum
	eva  = args.eva
	
	fin  = open(ftst, 'r')
	fout = open(tout, 'w')
	fdic = open(fdic, 'r')
	print '[POST]>>>processing...'
	postpro(fin, fout, fdic, eva)
	print '[POST]>>>done...'
	fin.close()
	fout.close()
	fdic.close()
	if eva == 1:
		print '[POST]>>>evaluating by BLEU'
		print '[POST]>>>generating xmls'
		arg1 = "-1f"
		arg2 = "-tf"
		arg3 = "-rnum"
		subprocess.call(["perl", "../evaluation/NiuTrans-generate-xml-for-mteval.pl", arg1, tout, arg2, fref, arg3, str(rnum)])
		print '[POST]>>>done'
		print '[POST]>>>caculating BLEU'
		arg1 = "-r"
		arg2 = "-s"
		arg3 = "-t"
		subprocess.call(["perl", "../evaluation/mteval-v13a.pl", arg1, "./ref.xml", arg2, "./src.xml", arg3, "./tst.xml"])
		print '[POST]>>>done'
