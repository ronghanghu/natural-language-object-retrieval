#!/bin/bash
wget -O ./datasets/Kitchen.tar.gz http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/Kitchen.tar.gz
tar -xzvf ./datasets/Kitchen.tar.gz -C ./datasets/
cp ./datasets/Kitchen/split/*.txt ./data/split/
cp ./datasets/Kitchen/annotation/*.json ./data/metadata/
