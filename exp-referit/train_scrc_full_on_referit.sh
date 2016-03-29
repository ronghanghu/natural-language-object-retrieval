#!/bin/bash
GPU_ID=0
WEIGHTS=./exp-referit/caffemodel/scrc_full_vgg_init.caffemodel

caffe train \
    -solver ./prototxt/scrc_full_vgg_solver.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID 2>&1
