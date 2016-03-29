from __future__ import print_function, division

import sys
import os
import numpy as np
import skimage.io
sys.path.append('./external/caffe-natural-language-object-retrieval/python/')
sys.path.append('./external/caffe-natural-language-object-retrieval/examples/coco_caption/')
import caffe

import util
from captioner import Captioner


vgg_weights_path = './models/VGG_ILSVRC_16_layers.caffemodel'
gpu_id = 0

image_dir = './datasets/ReferIt/ImageCLEF/images/'
cached_context_features_dir = './data/referit_context_features/'


image_net_proto = './prototxt/VGG_ILSVRC_16_layers_deploy.prototxt'
lstm_net_proto = './prototxt/scrc_word_to_preds_full.prototxt'
vocab_file = './data/vocabulary.txt'

captioner = Captioner(vgg_weights_path, image_net_proto, lstm_net_proto, vocab_file, gpu_id)
batch_size = 100
captioner.set_image_batch_size(batch_size)

imlist = util.io.load_str_list('./data/split/referit_all_imlist.txt')
num_im = len(imlist)

# Load all images into memory
loaded_images = []
for n_im in range(num_im):
    if n_im % 200 == 0:
        print('loading image %d / %d into memory' % (n_im, num_im))

    im = skimage.io.imread(image_dir + imlist[n_im] + '.jpg')
    # Gray scale to RGB
    if im.ndim == 2:
        im = np.tile(im[..., np.newaxis], (1, 1, 3))
    # RGBA to RGB
    im = im[:, :, :3]
    loaded_images.append(im)

# Compute fc7 feature from loaded images, as whole image contextual feature
descriptors = captioner.compute_descriptors(loaded_images, output_name='fc7')

# Save computed contextual features
if not os.path.isdir(cached_context_features_dir):
    os.mkdir(cached_context_features_dir)
for n_im in range(num_im):
    if n_im % 200 == 0:
        print('saving contextual features %d / %d' % (n_im, num_im))
    save_path = cached_context_features_dir + imlist[n_im] + '_fc7.npy'
    np.save(save_path, descriptors[n_im, :])
