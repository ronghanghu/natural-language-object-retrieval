from __future__ import print_function, division

import os
import numpy as np

import util
import retriever

trn_imlist_file = './data/split/kitchen_trainval_imlist.txt'

image_dir = './datasets/Kitchen/images/Kitchen/'
query_file = './data/metadata/kitchen_query_dict.json'
vocab_file = './data/vocabulary.txt'

N_batch = 50  # batch size during training
T = 20  # unroll timestep of LSTM

save_image_list_file = './data/kitchen_train_image_list.txt'
save_hdf5_list_file  = './data/kitchen_train_hdf5_list.txt'
save_hdf5_dir = './data/kitchen_hdf5_50/'

imset = set(util.io.load_str_list(trn_imlist_file))
vocab_dict = retriever.build_vocab_dict_from_file(vocab_file)
query_dict = util.io.load_json(query_file)

train_pairs = []
for imname, des in query_dict.iteritems():
    if imname not in imset:
        continue
    train_pairs += [(imname, d) for d in des]

# random shuffle training pairs
np.random.seed(3)
perm_idx = np.random.permutation(np.arange(len(train_pairs)))
train_pairs = [train_pairs[n] for n in perm_idx]

num_train_pairs = len(train_pairs)
num_train_pairs = num_train_pairs - num_train_pairs % N_batch
train_pairs = train_pairs[:num_train_pairs]
num_batch = int(num_train_pairs // N_batch)

image_list = []
hdf5_list = []

# generate hdf5 files
if not os.path.isdir(save_hdf5_dir):
    os.mkdir(save_hdf5_dir)
for n_batch in range(num_batch):
    if (n_batch+1) % 10 == 0:
        print('writing batch %d / %d' % (n_batch+1, num_batch))
    begin = n_batch * N_batch
    end = (n_batch + 1) * N_batch
    cont_sentences = np.zeros([T, N_batch], dtype=np.float32)
    input_sentences = np.zeros([T, N_batch], dtype=np.float32)
    target_sentences = np.zeros([T, N_batch], dtype=np.float32)
    for n_pair in range(begin, end):
        # Append 0 as dummy label
        image_path = image_dir + train_pairs[n_pair][0] + '.JPEG 0' # 0 as dummy label
        image_list.append(image_path)

        stream = retriever.sentence2vocab_indices(train_pairs[n_pair][1], vocab_dict)
        if len(stream) > T-1:
            stream = stream[:T-1]
        pad = T - 1 - len(stream)
        cont_sentences[:, n_pair-begin] = [0] + [1] * len(stream) + [0] * pad
        input_sentences[:, n_pair-begin] = [0] + stream + [-1] * pad
        target_sentences[:, n_pair-begin] = stream + [0] + [-1] * pad
    h5_filename = save_hdf5_dir + '%d_to_%d.h5' % (begin, end)
    retriever.write_batch_to_hdf5(h5_filename, cont_sentences, input_sentences, target_sentences)
    hdf5_list.append(h5_filename)

util.io.save_str_list(image_list, save_image_list_file)
util.io.save_str_list(hdf5_list, save_hdf5_list_file)
