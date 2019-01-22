# ambyer
# 2017.08.23

# this file created to save models just like reNMT did, and for testing the model_ave_ensemble

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

mdl_1 = '../model/1/model_1-'
mdl_2 = '../model/2/model_2-'
mdl_3 = '../model/3/model_3-'

a1 = np.array([1,2,3])
a2 = np.array([1,2,3])
a3 = np.array([1,2,3])

b1 = np.array([[1,2,3], [4,5,6]])
b2 = np.array([[1,2,3], [4,5,6]])
b3 = np.array([[1,2,3], [4,5,6]])

c1 = np.array([[1,2,3], [4,5,6], [7,8,9]])
c2 = np.array([[4,5,6], [7,8,9], [1,2,3]])
c3 = np.array([[7,8,9], [1,2,3], [4,5,6]])

d1 = np.array([[[1,2,3,4], [2,3,4,5], [3,4,5,6]], [[4,5,6,7], [5,6,7,8], [6,7,8,9]]])
d2 = np.array([[[10,20,30,40], [20,3,4,5], [30,40,50,60]], [[40,50,60,70], [50,60,70,80], [60,70,80,90]]])
d3 = np.array([[[1,2,3,4], [2,3,4,5], [3,4,5,6]], [[4,5,6,7], [5,6,7,8], [6,7,8,9]]])

np.savez(mdl_1 + suffix_enc_emb, a = a1, b = b1)
np.savez(mdl_1 + suffix_for_enc_hid1, a = a1, c = c1)
np.savez(mdl_1 + suffix_for_enc_hid2, a = a1, c = c1)
np.savez(mdl_1 + suffix_bac_enc_hid1, b = b1, d = d1)
np.savez(mdl_1 + suffix_bac_enc_hid2, b = b1, d = d1)
np.savez(mdl_1 + suffix_dec, a = a1)
np.savez(mdl_1 + suffix_dec_emb, a = a1)
np.savez(mdl_1 + suffix_dec_hid1, b = b1)
np.savez(mdl_1 + suffix_dec_hid2, c = c1)
np.savez(mdl_1 + suffix_dec_att, a = a1, b = b1, c = c1, d = d1)
np.savez(mdl_1 + suffix_dec_out, d = d1)



np.savez(mdl_2 + suffix_enc_emb, a = a2, b = b2)
np.savez(mdl_2 + suffix_for_enc_hid1, a = a2, c = c2)
np.savez(mdl_2 + suffix_for_enc_hid2, a = a2, c = c2)
np.savez(mdl_2 + suffix_bac_enc_hid1, b = b2, d = d2)
np.savez(mdl_2 + suffix_bac_enc_hid2, b = b2, d = d2)
np.savez(mdl_2 + suffix_dec, a = a2)
np.savez(mdl_2 + suffix_dec_emb, a = a2)
np.savez(mdl_2 + suffix_dec_hid1, b = b2)
np.savez(mdl_2 + suffix_dec_hid2, c = c2)
np.savez(mdl_2 + suffix_dec_att, a = a2, b = b2, c = c2, d = d2)
np.savez(mdl_2 + suffix_dec_out, d = d2)



np.savez(mdl_3 + suffix_enc_emb, a = a3, b = b3)
np.savez(mdl_3 + suffix_for_enc_hid1, a = a3, c = c3)
np.savez(mdl_3 + suffix_for_enc_hid2, a = a3, c = c3)
np.savez(mdl_3 + suffix_bac_enc_hid1, b = b3, d = d3)
np.savez(mdl_3 + suffix_bac_enc_hid2, b = b3, d = d3)
np.savez(mdl_3 + suffix_dec, a = a3)
np.savez(mdl_3 + suffix_dec_emb, a = a3)
np.savez(mdl_3 + suffix_dec_hid1, b = b3)
np.savez(mdl_3 + suffix_dec_hid2, c = c3)
np.savez(mdl_3 + suffix_dec_att, a = a3, b = b3, c = c3, d = d3)
np.savez(mdl_3 + suffix_dec_out, d = d3)

