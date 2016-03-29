#!/bin/bash
wget -O ./datasets/ReferIt/ReferitData/ReferitData.zip http://tamaraberg.com/referitgame/ReferitData.zip
unzip ./datasets/ReferIt/ReferitData/ReferitData.zip -d ./datasets/ReferIt/ReferitData/
wget -O ./datasets/ReferIt/ImageCLEF/referitdata.tar.gz http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referitdata.tar.gz
tar -xzvf ./datasets/ReferIt/ImageCLEF/referitdata.tar.gz -C ./datasets/ReferIt/ImageCLEF/
