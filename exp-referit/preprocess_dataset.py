# -*- coding: utf-8 -*-

from __future__ import division, print_function

import scipy.io as sio
import numpy as np
import os
import util

def load_imcrop(imlist, mask_dir):
    imcrop_dict = {im_name:[] for im_name in imlist}
    imcroplist = []
    masklist = os.listdir(mask_dir)
    for mask_name in masklist:
        imcrop_name = mask_name.split('.', 1)[0]
        imcroplist.append(imcrop_name)
        im_name = imcrop_name.split('_', 1)[0]
        imcrop_dict[im_name].append(imcrop_name)
    return imcroplist, imcrop_dict

def load_referit_annotation(imcroplist, annotation_file):
    print('loading ReferIt dataset annotations...')
    query_dict = {imcrop_name:[] for imcrop_name in imcroplist}
    with open(annotation_file) as f:
        raw_annotation = f.readlines()
    for s in raw_annotation:
        # example annotation line:
        # 8756_2.jpg~sunray at very top~.33919597989949750~.023411371237458192
        splits = s.strip().split('~', 2)
        # example: 8756_2 (segmentation regions)
        imcrop_name = splits[0].split('.', 1)[0]
        # example: 'sunray at very top'
        description = splits[1]
        # construct imcrop_name - discription list dictionary
        # an image crop can have zero or mutiple annotations
        query_dict[imcrop_name].append(description)
        return query_dict

def load_imcrop_bbox(mask_dir):
    print('loading image crop bounding boxes...')
    imcrop_bbox_dict = {}
    masklist = os.listdir(mask_dir)
    for n in range(len(masklist)):
        if n % 200 == 0:
            print('processing %d / %d' % (n, len(masklist)))
        mask_name = masklist[n]
        mask = sio.loadmat(mask_dir + mask_name)['segimg_t']
        idx = np.nonzero(mask == 0)
        x_min, x_max = np.min(idx[1]), np.max(idx[1])
        y_min, y_max = np.min(idx[0]), np.max(idx[0])
        bbox = [x_min, y_min, x_max, y_max]
        imcrop_name = mask_name.split('.', 1)[0]
        imcrop_bbox_dict[imcrop_name] = bbox
    return imcrop_bbox_dict

def main():
    mask_dir = './datasets/ReferIt/ImageCLEF/mask/'
    annotation_file = './datasets/ReferIt/ReferitData/RealGames.txt'

    trn_imlist_file = './data/split/referit_trainval_imlist.txt'
    tst_imlist_file = './data/split/referit_test_imlist.txt'
    metadata_dir = './data/metadata/'

    trn_imlist = util.io.load_str_list(trn_imlist_file)
    tst_imlist = util.io.load_str_list(tst_imlist_file)
    imlist = trn_imlist + tst_imlist

    imcroplist, imcrop_dict = load_imcrop(imlist, mask_dir)
    query_dict = load_referit_annotation(imcroplist, annotation_file)
    imcrop_bbox_dict = load_imcrop_bbox(mask_dir)

    util.io.save_json(imcrop_dict, metadata_dir + 'referit_imcrop_dict.json')
    util.io.save_json(query_dict,  metadata_dir + 'referit_query_dict.json')
    util.io.save_json(imcrop_bbox_dict, metadata_dir + 'referit_imcrop_bbox_dict.json')

if __name__ == '__main__':
    main()
