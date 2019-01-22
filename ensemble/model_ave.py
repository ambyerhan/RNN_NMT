# ambyer
# 2017.08.23

# this file is for ensemble that averaging the model

import sys
import numpy as np
import time
import os

if len(sys.argv) < 4:
    print '[Error] 3 params at least, format: model_1 model_2 model_ave!'
    sys.exit(1)

suffix_enc_emb = 'myNMT.EnEmbedding.npz'
suffix_for_enc_hid1 = 'myNMT.forward_encode.hidden_1.npz'
suffix_for_enc_hid2 = 'myNMT.forward_encode.hidden_2.npz'
suffix_bac_enc_hid1 = 'myNMT.bacward_encode.hidden_1.npz'
suffix_bac_enc_hid2 = 'myNMT.bacward_encode.hidden_2.npz'

suffix_dec = 'myNMT.decoder.npz'
suffix_dec_emb = 'myNMT.DeEmbedding.npz'
suffix_dec_hid1 = 'myNMT.decoder.hidden_1.npz'
suffix_dec_hid2 = 'myNMT.decoder.hidden_2.npz'
suffix_dec_att = 'myNMT.decoder.attention.npz'
suffix_dec_out = 'myNMT.decoder.output.npz'

suffixes = [suffix_enc_emb, suffix_for_enc_hid1, suffix_for_enc_hid2, suffix_bac_enc_hid1, suffix_bac_enc_hid2, suffix_dec,
            suffix_dec_emb, suffix_dec_hid1, suffix_dec_hid2, suffix_dec_att, suffix_dec_out]

in_models = sys.argv[1: -1]
ave_model = sys.argv[-1]
ave_model_dir = '../model/ave_model/'
if not os.path.exists(ave_model_dir):
    print '[Debug] Making the new directory: %s' % ave_model_dir
    os.makedirs(ave_model_dir)

print '[Debug] Total %d models were found!' % (len(in_models))

begt = time.time()
print '[Debug] Ensembling the models to file %s' % (ave_model_dir + ave_model)

for suffix in suffixes:
    params = [np.load(in_model + suffix) for in_model in in_models]
    key_set = set()
    for param in params:
        for k in param.keys():
            key_set.add(k)

    ave_param = dict()
    for key in key_set:
        value_list = []
        for i, param in enumerate(params):
            if key not in param:
                print '[Error] The param %s not included in model %s' % (key, in_models[i])
            value_list.append(param[key])
        ave_param[key] = np.array(value_list).mean(axis = 0)
    np.savez((ave_model_dir + ave_model + suffix), **ave_param)

print '[Debug] Ensemble done and saved, total use of %.3f sec' % (time.time() - begt)
