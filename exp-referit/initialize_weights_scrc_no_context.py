from __future__ import division, print_function

import sys
import numpy as np
sys.path.append('./external/caffe-natural-language-object-retrieval/python/')
import caffe

old_prototxt = './prototxt/coco_pretrained.prototxt'
old_caffemodel = './models/coco_pretrained_iter_100000.caffemodel'
new_prototxt = './prototxt/scrc_no_context_vgg_buffer_50.prototxt'
new_caffemodel = './exp-referit/caffemodel/scrc_no_context_vgg_init.caffemodel'
old_net = caffe.Net(old_prototxt, old_caffemodel, caffe.TRAIN)
new_net = caffe.Net(new_prototxt, old_caffemodel, caffe.TRAIN)

new_net.params['lstm2-extended'][0].data[...] = old_net.params['lstm2'][0].data[...]
new_net.params['lstm2-extended'][1].data[...] = old_net.params['lstm2'][1].data[...]
new_net.params['lstm2-extended'][2].data[:, :1000] = old_net.params['lstm2'][2].data[...]
new_net.params['lstm2-extended'][2].data[:, 1000:] = 0
new_net.params['lstm2-extended'][3].data[...] = old_net.params['lstm2'][3].data[...]

new_net.save(new_caffemodel)
