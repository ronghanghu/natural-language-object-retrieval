from __future__ import division, print_function

import sys
import numpy as np
import skimage.io
sys.path.append('./external/caffe-natural-language-object-retrieval/python/')
sys.path.append('./external/caffe-natural-language-object-retrieval/examples/coco_caption/')
import caffe

import util
from captioner import Captioner
import retriever

################################################################################
# Test Parameters

# distractor_set can be either "kitchen" or "imagenet"
# For "kitchen" experiment, the distractors are sampled from test set itsef
# For "imagenet" experiment, the distractors are sampled from ImageNET distractor images
distractor_set = "kitchen"
# Number of distractors sampled for each object
distractor_per_object = 10

pretrained_weights_path = './models/scrc_kitchen.caffemodel'

gpu_id = 0  # the GPU to test the SCRC model

tst_imlist_file = './data/split/kitchen_test_imlist.txt'
################################################################################

image_dir = './datasets/Kitchen/images/Kitchen/'

if distractor_set == "kitchen":
    distractor_dir = image_dir
    distractor_imlist_file = tst_imlist_file
else:
    distractor_dir = './datasets/Kitchen/images/ImageNET/'
    distractor_imlist_file = './data/split/kitchen_imagenet_imlist.txt'

query_file = './data/metadata/kitchen_query_dict.json'
vocab_file = './data/vocabulary.txt'

# utilize the captioner module from LRCN
lstm_net_proto = './prototxt/scrc_word_to_preds_no_spatial_no_context.prototxt'
image_net_proto = './prototxt/VGG_ILSVRC_16_layers_deploy.prototxt'
captioner = Captioner(pretrained_weights_path, image_net_proto, lstm_net_proto,
                      vocab_file, gpu_id)
captioner.set_image_batch_size(50)
vocab_dict = retriever.build_vocab_dict_from_captioner(captioner)

# Load image and caption list
imlist = util.io.load_str_list(tst_imlist_file)
num_im = len(imlist)
query_dict = util.io.load_json(query_file)

# Load distractors
distractor_list = util.io.load_str_list(distractor_imlist_file)
num_distractors = len(distractor_list)

# Sample distractor images for each test image
distractor_ids_per_im = {}
np.random.seed(3)  # fix random seed for test repeatibility
for imname in imlist:
    # Sample distractor_per_object*2 distractors to make sure the test image
    # itself is not among the distractors (this)
    distractor_ids = np.random.choice(num_distractors,
                                      distractor_per_object*2, replace=False)
    distractor_names = [distractor_list[n] for n in distractor_ids[:distractor_per_object]]
    # Use the second half if the imname is among the first half
    if imname not in distractor_names:
        distractor_ids_per_im[imname] = distractor_ids[:distractor_per_object]
    else:
        distractor_ids_per_im[imname] = distractor_ids[distractor_per_object:]

# Compute descriptors for both object images and distractor images
image_path_list = [image_dir+imname+'.JPEG' for imname in imlist]
distractor_path_list = [distractor_dir+imname+'.JPEG' for imname in distractor_list]

obj_descriptors = captioner.compute_descriptors(image_path_list)
dis_descriptors = captioner.compute_descriptors(distractor_path_list)

################################################################################
# Test top-1 precision
correct_num = 0
total_num = 0
for n_im in range(num_im):
    print('testing image %d / %d' % (n_im, num_im))
    imname = imlist[n_im]
    for sentence in query_dict[imname]:
        # compute test image (target object) score given the description sentence
        obj_score = retriever.score_descriptors(obj_descriptors[n_im:n_im+1, :],
                                                sentence, captioner, vocab_dict)[0]
        # compute distractor scores given the description sentence
        dis_idx = distractor_ids_per_im[imname]
        dis_scores = retriever.score_descriptors(dis_descriptors[dis_idx, :],
                                                 sentence, captioner, vocab_dict)

        # for a retrieval to be correct, the object image must score higher than
        # all distractor images
        correct_num += np.all(obj_score > dis_scores)
        total_num += 1

print('Top-1 precision on the whole test set: %f' % (correct_num/total_num))
################################################################################
