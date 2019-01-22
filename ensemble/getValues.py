# ambyer
# 2017.08.23

# this file created for checking the ave_model is correct or not

import sys
import numpy as np

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


fmodel = sys.argv[1]

for suffix in suffixes:
    print '/////////////////////////////////////////////////'
    print 'the params of file: %s' % (fmodel + suffix)
    param = np.load(fmodel + suffix)
    key_set = set()
    for k in param.keys():
        key_set.add(k)

    for key in key_set:
        print '>>>>>the param of %s:' % key
        print param[key]

#fmodel.close()
