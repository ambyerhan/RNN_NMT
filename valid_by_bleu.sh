#!/bin/sh
# use cmd [sed -i 's/var//g' filename] to delete the ^M]
# args:         model_name      src_file        dev_file        ref_num

echo "hello world"

# theano device
device=gpu0

# Bleu path
bleu_file=./model/bleu_history.txt

# $0: the script
model_name=$1
src_file=$2
dev_file=$3
ref_num=$4

echo "model:$model_name   test_set:$src_file   dev:$dev_file   rnum:$ref_num   bleu_his:$bleu_file"

# translate the validation set
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device python ./main.py \
	--beamsize 12 \
	--mode 2 \
	--model $model_name \
	--swvocab ./data/c.40w.dict \
	--twvocab ./data/e.40w.dict \
	--norm_alpha 1.0 \
	--dec_maxlen 50 \
	--test $src_file

echo  "perl NiuTrans-generate-xml-for-mteval.pl -1f ${src_file}.rslt -tf ${dev_file} -rnum ${ref_num}"
order=`perl NiuTrans-generate-xml-for-mteval.pl -1f ${src_file}.rslt -tf ${dev_file} -rnum ${ref_num}`
echo "perl mteval-v13a.pl -r ref.xml -s src.xml -t tst.xml | grep 'NIST score' | cut -f 9 -d ' '"
bleu=`perl mteval-v13a.pl -r ref.xml -s src.xml -t tst.xml | grep "NIST score" | cut -f 9 -d " "`

`echo $bleu >> ${bleu_file}`
`rm ref.xml src.xml tst.xml ${src_file}.rslt ${src_file}.rslt.temp`

