from __future__ import division, print_function

import os
import numpy as np
import scipy.io as sio
import skimage
import skimage.io
import skimage.transform

import util


def load_imcrop(imlist, mask_dir):
    imcrop_dict = {im_name: [] for im_name in imlist}
    imcroplist = []
    masklist = os.listdir(mask_dir)
    for mask_name in masklist:
        imcrop_name = mask_name.split('.', 1)[0]
        imcroplist.append(imcrop_name)
        im_name = imcrop_name.split('_', 1)[0]
        imcrop_dict[im_name].append(imcrop_name)
    return imcroplist, imcrop_dict


def load_image_size(imlist, image_dir):
    num_im = len(imlist)
    imsize_dict = {}
    for n_im in range(num_im):
        if n_im % 200 == 0:
            print('processing image %d / %d' % (n_im, num_im))
        im = skimage.io.imread(image_dir + imlist[n_im] + '.jpg')
        imsize_dict[imlist[n_im]] = [im.shape[1], im.shape[0]]  # [width, height]
    return imsize_dict


def load_referit_annotation(imcroplist, annotation_file):
    print('loading ReferIt dataset annotations...')
    query_dict = {imcrop_name: [] for imcrop_name in imcroplist}
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


def load_and_resize_imcrop(mask_dir, image_dir, resized_imcrop_dir):
    print('loading image crop bounding boxes...')
    imcrop_bbox_dict = {}
    masklist = os.listdir(mask_dir)
    if not os.path.isdir(resized_imcrop_dir):
        os.mkdir(resized_imcrop_dir)
    for n in range(len(masklist)):
        if n % 200 == 0:
            print('processing image crop %d / %d' % (n, len(masklist)))
        mask_name = masklist[n]
        mask = sio.loadmat(mask_dir + mask_name)['segimg_t']
        idx = np.nonzero(mask == 0)
        x_min, x_max = np.min(idx[1]), np.max(idx[1])
        y_min, y_max = np.min(idx[0]), np.max(idx[0])
        bbox = [x_min, y_min, x_max, y_max]
        imcrop_name = mask_name.split('.', 1)[0]
        imcrop_bbox_dict[imcrop_name] = bbox

        # resize the image crops
        imname = imcrop_name.split('_', 1)[0] + '.jpg'
        image_path = image_dir + imname
        im = skimage.io.imread(image_path)
        # Gray scale to RGB
        if im.ndim == 2:
            im = np.tile(im[..., np.newaxis], (1, 1, 3))
        # RGBA to RGB
        im = im[:, :, :3]
        resized_im = skimage.transform.resize(im[y_min:y_max+1,
                                                 x_min:x_max+1, :], [224, 224])
        save_path = resized_imcrop_dir + imcrop_name + '.png'
        skimage.io.imsave(save_path, resized_im)
    return imcrop_bbox_dict


def main():
    image_dir = './datasets/ReferIt/ImageCLEF/images/'
    mask_dir = './datasets/ReferIt/ImageCLEF/mask/'
    annotation_file = './datasets/ReferIt/ReferitData/RealGames.txt'
    imlist_file = './data/split/referit_all_imlist.txt'
    metadata_dir = './data/metadata/'
    resized_imcrop_dir = './data/resized_imcrop/'

    imlist = util.io.load_str_list(imlist_file)
    imsize_dict = load_image_size(imlist, image_dir)
    imcroplist, imcrop_dict = load_imcrop(imlist, mask_dir)
    query_dict = load_referit_annotation(imcroplist, annotation_file)
    imcrop_bbox_dict = load_and_resize_imcrop(mask_dir, image_dir,
                                              resized_imcrop_dir)

    util.io.save_json(imsize_dict, metadata_dir + 'referit_imsize_dict.json')
    util.io.save_json(imcrop_dict, metadata_dir + 'referit_imcrop_dict.json')
    util.io.save_json(query_dict,  metadata_dir + 'referit_query_dict.json')
    util.io.save_json(imcrop_bbox_dict, metadata_dir + 'referit_imcrop_bbox_dict.json')

if __name__ == '__main__':
    main()
