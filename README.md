# Natural Language Object Retrieval
This repository contains the code for the following paper:

* R. Hu, H. Xu, M. Rohrbach, J. Feng, K. Saenko, T. Darrell, *Natural Language Object Retrieval*, in Computer Vision and Pattern Recognition (CVPR), 2016 ([PDF](http://arxiv.org/pdf/1511.04164))
```
@article{hu2015natural,
  title={Natural Language Object Retrieval},
  author={Hu, Ronghang and Xu, Huazhe and Rohrbach, Marcus and Feng, Jiashi and Saenko, Kate and Darrell, Trevor},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2016}
}
```

Project Page: http://ronghanghu.com/text_obj_retrieval  
License: BSD 2-Clause license

## Installation
1. Download this repository or clone with Git, and then `cd` into the root directory of the repository.
2. Run `./external/download_caffe.sh` to download the SCRC Caffe version for this experiment. It will be downloaded and unzipped into `external/caffe-natural-language-object-retrieval`. This version is modified from the [Caffe LRCN implementation](http://jeffdonahue.com/lrcn/).
3. Build the SCRC Caffe version in `external/caffe-natural-language-object-retrieval`, following the [Caffe installation instruction](http://caffe.berkeleyvision.org/installation.html). **Remember to also build pycaffe.**

## SCRC demo
1. Download the pretrained models with `./models/download_trained_models.sh`.  
2. Run the SCRC demo in `./demo/retrieval_demo.ipynb` with [Jupyter Notebook (IPython Notebook)](http://ipython.org/notebook.html).

![Image](http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/scrc_demo.jpg)

## Train and evaluate SCRC model on ReferIt Dataset
1. Download the ReferIt dataset: `./datasets/download_referit_dataset.sh`.
2. Download pre-extracted EdgeBox proposals: `./data/download_edgebox_proposals.sh`.
3. Preprocess the ReferIt dataset to generate metadata needed for training and evaluation: `python ./exp-referit/preprocess_dataset.py`.
4. Cache the scene-level contextual features to disk: `python ./exp-referit/cache_referit_context_features.py`.
5. Build training image lists and HDF5 batches: `python ./exp-referit/cache_referit_training_batches.py`.
6. Initialize the model parameters and train with SGD: `python ./exp-referit/initialize_weights_scrc_full.py && ./exp-referit/train_scrc_full_on_referit.sh`.
7. Evaluate the trained model: `python ./exp-referit/test_scrc_on_referit.py`.

Optionally, you may also train a SCRC version without contextual feature, using `python ./exp-referit/initialize_weights_scrc_no_context.py && ./exp-referit/train_scrc_no_context_on_referit.sh`.

## Train and evaluate SCRC model on Kitchen Dataset
1. Download the Kitchen dataset: `./datasets/download_kitchen_dataset.sh`.
2. Build training image lists and HDF5 batches: `python exp-kitchen/cache_kitchen_training_batches.py`.
3. Train with SGD: `./exp-kitchen/train_scrc_kitchen.sh`.
4. Evaluate the trained model: `python exp-kitchen/test_scrc_on_kitchen.py`.
