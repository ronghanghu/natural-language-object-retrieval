from __future__ import division, print_function

import os
import re
import numpy as np
import h5py
import skimage.io

# Compute vocabulary indices from sentence
MAX_WORDS = 20
UNK_IDENTIFIER = '<unk>' # <unk> is the word used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def sentence2vocab_indices(raw_sentence, vocab_dict):
    splits = SENTENCE_SPLIT_REGEX.split(raw_sentence.strip())
    sentence = [ s.lower() for s in splits if len(s.strip()) > 0 ]
    # remove .
    if sentence[-1] == '.':
        sentence = sentence[:-1]
    vocab_indices = [ (vocab_dict[s] if vocab_dict.has_key(s) else vocab_dict[UNK_IDENTIFIER])
        for s in sentence ]
    if len(vocab_indices) > MAX_WORDS:
        vocab_indices = vocab_indices[:MAX_WORDS]
    return vocab_indices

# Build vocabulary dictionary from file
def build_vocab_dict_from_file(vocab_file):
    vocab = ['<EOS>']
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
        vocab += [ word.strip() for word in lines ]
    vocab_dict = { vocab[n] : n for n in range(len(vocab)) }
    return vocab_dict

# Build vocabulary dictionary from captioner
def build_vocab_dict_from_captioner(captioner):
    vocab_dict = {captioner.vocab[n] : n for n in range(len(captioner.vocab))}
    return vocab_dict

def score_descriptors(descriptors, raw_sentence, captioner, vocab_dict):
    vocab_indices = sentence2vocab_indices(raw_sentence, vocab_dict)
    num_descriptors = descriptors.shape[0]
    scores = np.zeros(num_descriptors)

    net = captioner.lstm_net

    T = len(vocab_indices)
    N = descriptors.shape[0]
    # reshape only when necessary
    if list(net.blobs['cont_sentence'].shape) != [MAX_WORDS, N]:
        net.blobs['cont_sentence'].reshape(MAX_WORDS, N)
        net.blobs['input_sentence'].reshape(MAX_WORDS, N)
        net.blobs['image_features'].reshape(N, *net.blobs['image_features'].data.shape[1:])
        # print('LSTM net reshape to ' + str([MAX_WORDS, N]))

    cont_sentence = np.array([0] + [1 for v in vocab_indices[:-1] ]).reshape((-1, 1))
    input_sentence = np.array([0] + vocab_indices[:-1] ).reshape((-1, 1))

    net.blobs['cont_sentence'].data[:T, :] = cont_sentence
    net.blobs['input_sentence'].data[:T, :] = input_sentence
    net.blobs['image_features'].data[...] = descriptors
    net.forward()

    probs = net.blobs['probs'].data[:T, :, :]
    for t in range(T):
        scores += np.log(probs[t, :, vocab_indices[t] ])
    return scores

def score_descriptors_context(descriptors, raw_sentence, fc7_context, captioner, vocab_dict):
    vocab_indices = sentence2vocab_indices(raw_sentence, vocab_dict)
    num_descriptors = descriptors.shape[0]
    scores = np.zeros(num_descriptors)

    net = captioner.lstm_net

    T = len(vocab_indices)
    N = descriptors.shape[0]
    # reshape only when necessary
    if list(net.blobs['cont_sentence'].shape) != [MAX_WORDS, N]:
        net.blobs['cont_sentence'].reshape(MAX_WORDS, N)
        net.blobs['input_sentence'].reshape(MAX_WORDS, N)
        net.blobs['image_features'].reshape(N, *net.blobs['image_features'].data.shape[1:])
        net.blobs['fc7_context'].reshape(N, *net.blobs['fc7_context'].data.shape[1:])
        # print('LSTM net reshape to ' + str([MAX_WORDS, N]))

    cont_sentence = np.array([0] + [1 for v in vocab_indices[:-1] ]).reshape((-1, 1))
    input_sentence = np.array([0] + vocab_indices[:-1] ).reshape((-1, 1))

    net.blobs['cont_sentence'].data[:T, :] = cont_sentence
    net.blobs['input_sentence'].data[:T, :] = input_sentence
    net.blobs['image_features'].data[...] = descriptors
    net.blobs['fc7_context'].data[...] = fc7_context
    net.forward()

    probs = net.blobs['probs'].data[:T, :, :]
    for t in range(T):
        scores += np.log(probs[t, :, vocab_indices[t] ])
    return scores


# all boxes are [xmin, ymin, xmax, ymax] format, 0-indexed, including xmax and ymax
def compute_iou(boxes, target):
    assert(target.ndim == 1 and boxes.ndim == 2)
    A_boxes = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    A_target = (target[2] - target[0] + 1) * (target[3] - target[1] + 1)
    assert(np.all(A_boxes >= 0))
    assert(np.all(A_target >= 0))
    I_x1 = np.maximum(boxes[:, 0], target[0])
    I_y1 = np.maximum(boxes[:, 1], target[1])
    I_x2 = np.minimum(boxes[:, 2], target[2])
    I_y2 = np.minimum(boxes[:, 3], target[3])
    A_I = np.maximum(I_x2 - I_x1 + 1, 0) * np.maximum(I_y2 - I_y1 + 1, 0)
    IoUs = A_I / (A_boxes + A_target - A_I)
    assert(np.all(0 <= IoUs) and np.all(IoUs <= 1))
    return IoUs

def crop_edge_boxes(image, edge_boxes):
    # load images
    if type(image) in (str, unicode):
        image = skimage.io.imread(image)
    if image.dtype == np.float32:
        image *= 255
        image = image.astype(np.uint8)
    # Gray scale to RGB
    if image.ndim == 2:
        image = np.tile(image[..., np.newaxis], (1, 1, 3))
    # RGBA to RGB
    image = image[:, :, :3]
    x1, y1, x2, y2 = edge_boxes[:, 0], edge_boxes[:, 1], edge_boxes[:, 2], edge_boxes[:, 3]
    crops = [image[y1[n]:y2[n]+1, x1[n]:x2[n]+1, :] for n in range(edge_boxes.shape[0])]
    return crops

def compute_descriptors_edgebox(captioner, image, edge_boxes, output_name='fc8'):
    crops = crop_edge_boxes(image, edge_boxes);
    return compute_descriptors(captioner, crops, output_name)

def preprocess_image(captioner, image, verbose=False):
    if type(image) in (str, unicode):
        image = skimage.io.imread(image)
    if image.dtype == np.float32:
        image *= 255
        image = image.astype(np.uint8)
    # Gray scale to RGB
    if image.ndim == 2:
        image = np.tile(image[..., np.newaxis], (1, 1, 3))
    # RGBA to RGB
    image = image[:, :, :3]
    preprocessed_image = captioner.transformer.preprocess('data', image)
    return preprocessed_image

def compute_descriptors(captioner, image_list, output_name='fc8'):
    batch = np.zeros_like(captioner.image_net.blobs['data'].data)
    batch_shape = batch.shape
    batch_size = batch_shape[0]
    descriptors_shape = (len(image_list), ) + \
        captioner.image_net.blobs[output_name].data.shape[1:]
    descriptors = np.zeros(descriptors_shape)
    for batch_start_index in range(0, len(image_list), batch_size):
        batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]
        for batch_index, image_path in enumerate(batch_list):
            batch[batch_index:(batch_index + 1)] = preprocess_image(captioner, image_path)
        current_batch_size = min(batch_size, len(image_list) - batch_start_index)
        captioner.image_net.forward(data=batch)
        descriptors[batch_start_index:(batch_start_index + current_batch_size)] = \
            captioner.image_net.blobs[output_name].data[:current_batch_size]
    return descriptors

# normalize bounding box features into 8-D feature
def compute_spatial_feat(bboxes, image_size):
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape((1, 4))
    im_w = image_size[0]
    im_h = image_size[1]
    assert(np.all(bboxes[:, 0] < im_w) and np.all(bboxes[:, 2] < im_w))
    assert(np.all(bboxes[:, 1] < im_h) and np.all(bboxes[:, 3] < im_h))

    feats = np.zeros((bboxes.shape[0], 8))
    feats[:, 0] = bboxes[:, 0] * 2.0 / im_w - 1  # x1
    feats[:, 1] = bboxes[:, 1] * 2.0 / im_h - 1  # y1
    feats[:, 2] = bboxes[:, 2] * 2.0 / im_w - 1  # x2
    feats[:, 3] = bboxes[:, 3] * 2.0 / im_h - 1  # y2
    feats[:, 4] = (feats[:, 0] + feats[:, 2]) / 2  # x0
    feats[:, 5] = (feats[:, 1] + feats[:, 3]) / 2  # y0
    feats[:, 6] = feats[:, 2] - feats[:, 0]  # w
    feats[:, 7] = feats[:, 3] - feats[:, 1]  # h
    return feats

# Write a batch of sentences to HDF5
def write_batch_to_hdf5(filename, cont_sentences, input_sentences,
                        target_sentences, dtype=np.float32):
    h5file = h5py.File(filename, 'w')
    dataset = h5file.create_dataset('cont_sentence',
        shape=cont_sentences.shape, dtype=np.float32)
    dataset[:] = cont_sentences
    dataset = h5file.create_dataset('input_sentence',
        shape=input_sentences.shape, dtype=np.float32)
    dataset[:] = input_sentences
    dataset = h5file.create_dataset('target_sentence',
        shape=target_sentences.shape, dtype=np.float32)
    dataset[:] = target_sentences
    h5file.close()

# Write a batch of sentences to HDF5
def write_bbox_to_hdf5(filename, bbox_coordinates, dtype=np.float32):
    h5file = h5py.File(filename, 'w')
    dataset = h5file.create_dataset('bbox_coordinate',
        shape=bbox_coordinates.shape, dtype=np.float32)
    dataset[:] = bbox_coordinates
    h5file.close()

# Write a batch of sentences to HDF5
def write_bbox_context_to_hdf5(filename, bbox_coordinates, fc7_context, dtype=np.float32):
    h5file = h5py.File(filename, 'w')
    dataset = h5file.create_dataset('bbox_coordinate',
        shape=bbox_coordinates.shape, dtype=np.float32)
    dataset[:] = bbox_coordinates
    dataset = h5file.create_dataset('fc7_context',
        shape=fc7_context.shape, dtype=np.float32)
    dataset[:] = fc7_context
    h5file.close()
