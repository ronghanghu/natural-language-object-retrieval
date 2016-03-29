#!/bin/bash
GPU_ID=0
WEIGHTS=./exp-referit/caffemodel/scrc_no_context_vgg_init.caffemodel

caffe train \
    -solver ./prototxt/scrc_no_context_vgg_solver.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID 2>&1
