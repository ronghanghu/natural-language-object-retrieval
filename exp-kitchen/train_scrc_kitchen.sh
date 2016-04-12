#!/bin/bash
GPU_ID=0
WEIGHTS=./models/coco_pretrained_iter_100000.caffemodel

caffe train \
    -solver ./prototxt/scrc_kitchen_solver.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID 2>&1
