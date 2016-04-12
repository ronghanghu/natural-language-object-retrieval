from __future__ import print_function, division

import os
import numpy as np

import util
import retriever

trn_imlist_file = './data/split/referit_trainval_imlist.txt'

image_dir = './datasets/ReferIt/ImageCLEF/images/'
resized_imcrop_dir = './data/resized_imcrop/'
cached_context_features_dir = './data/referit_context_features/'

imcrop_dict_file = './data/metadata/referit_imcrop_dict.json'
imcrop_bbox_dict_file = './data/metadata/referit_imcrop_bbox_dict.json'
imsize_dict_file = './data/metadata/referit_imsize_dict.json'
query_file = './data/metadata/referit_query_dict.json'
vocab_file = './data/vocabulary.txt'

N_batch = 50  # batch size during training
T = 20  # unroll timestep of LSTM

save_imcrop_list_file = './data/training/train_bbox_context_imcrop_list.txt'
save_wholeim_list_file = './data/training/train_bbox_context_wholeim_list.txt'
save_hdf5_text_list_file = './data/training/train_bbox_context_hdf5_text_list.txt'
save_hdf5_bbox_list_file = './data/training/train_bbox_context_hdf5_bbox_list.txt'
save_hdf5_dir = './data/training/hdf5_50_bbox_context/'

imset = set(util.io.load_str_list(trn_imlist_file))
vocab_dict = retriever.build_vocab_dict_from_file(vocab_file)
query_dict = util.io.load_json(query_file)
imsize_dict = util.io.load_json(imsize_dict_file)
imcrop_bbox_dict = util.io.load_json(imcrop_bbox_dict_file)

train_pairs = []
for imcrop_name, des in query_dict.iteritems():
    imname = imcrop_name.split('_', 1)[0]
    if imname not in imset:
        continue
    imsize = np.array(imsize_dict[imname])
    bbox = np.array(imcrop_bbox_dict[imcrop_name])
    bbox_feat = retriever.compute_spatial_feat(bbox, imsize)
    context_feature = np.load(cached_context_features_dir + imname + '_fc7.npy')
    train_pairs += [(imcrop_name, d, bbox_feat, imname, context_feature) for d in des]

# random shuffle training pairs
np.random.seed(3)
perm_idx = np.random.permutation(np.arange(len(train_pairs)))
train_pairs = [train_pairs[n] for n in perm_idx]

num_train_pairs = len(train_pairs)
num_train_pairs = num_train_pairs - num_train_pairs % N_batch
train_pairs = train_pairs[:num_train_pairs]
num_batch = int(num_train_pairs // N_batch)

imcrop_list = []
wholeim_list = []
hdf5_text_list = []
hdf5_bbox_list = []

# generate hdf5 files
if not os.path.isdir(save_hdf5_dir):
    os.mkdir(save_hdf5_dir)
for n_batch in range(num_batch):
    if (n_batch+1) % 100 == 0:
        print('writing batch %d / %d' % (n_batch+1, num_batch))
    begin = n_batch * N_batch
    end = (n_batch + 1) * N_batch
    cont_sentences = np.zeros([T, N_batch], dtype=np.float32)
    input_sentences = np.zeros([T, N_batch], dtype=np.float32)
    target_sentences = np.zeros([T, N_batch], dtype=np.float32)
    bbox_coordinates = np.zeros([N_batch, 8], dtype=np.float32)
    fc7_context = np.zeros([N_batch, 4096], dtype=np.float32)
    for n_pair in range(begin, end):
        # Append 0 as dummy label
        imcrop_path = resized_imcrop_dir + train_pairs[n_pair][0] + '.png 0'
        imcrop_list.append(imcrop_path)
        # Append 0 as dummy label
        wholeim_path = image_dir + train_pairs[n_pair][3] + '.jpg 0'
        wholeim_list.append(wholeim_path)
        stream = retriever.sentence2vocab_indices(train_pairs[n_pair][1],
                                                  vocab_dict)
        if len(stream) > T-1:
            stream = stream[:T-1]
        pad = T - 1 - len(stream)
        cont_sentences[:, n_pair-begin] = [0] + [1] * len(stream) + [0] * pad
        input_sentences[:, n_pair-begin] = [0] + stream + [-1] * pad
        target_sentences[:, n_pair-begin] = stream + [0] + [-1] * pad
        bbox_coordinates[n_pair-begin, :] = np.squeeze(train_pairs[n_pair][2])
        fc7_context[n_pair-begin, :] = train_pairs[n_pair][4]
    h5_text_filename = save_hdf5_dir + 'text_%d_to_%d.h5' % (begin, end)
    h5_bbox_filename = save_hdf5_dir + 'bbox_context_%d_to_%d.h5' % (begin, end)
    retriever.write_batch_to_hdf5(h5_text_filename, cont_sentences,
                                  input_sentences, target_sentences)
    retriever.write_bbox_context_to_hdf5(h5_bbox_filename, bbox_coordinates,
                                         fc7_context)
    hdf5_text_list.append(h5_text_filename)
    hdf5_bbox_list.append(h5_bbox_filename)

util.io.save_str_list(imcrop_list, save_imcrop_list_file)
util.io.save_str_list(wholeim_list, save_wholeim_list_file)
util.io.save_str_list(hdf5_text_list, save_hdf5_text_list_file)
util.io.save_str_list(hdf5_bbox_list, save_hdf5_bbox_list_file)
