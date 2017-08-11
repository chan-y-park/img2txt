# ``img2txt``
End-to-end deep learning model to generate a summary of the content of an image in a sentence.

[Overview](#overview)

[Acknowledgements](#acknowledgements)

## Overview

The model architecture is based on

> "Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning
Challenge."
> Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan.
> *IEEE transactions on pattern analysis and machine intelligence (2016).*
> http://arxiv.org/abs/1609.06647

and the following code in the TensorFlow model zoo is frequently used as a reference,

https://github.com/tensorflow/models/tree/master/im2txt

but the code is written from scratch using TensorFlow APIs, except Inception models whose codes are obtained in the current master version (325609e) of https://github.com/tensorflow/models/tree/master/slim.


## Requirements

### Library
``img2txt`` is developed on the following environment.
* Ubuntu 16.04.2 LTS
* Python 3.6
* NumPy
* TensorFlow 1.2
* Pillow
* NLTK (NLTK data needed for tokenization; only ``nltk_data/tokenizers/punkt/PY3/english.pickle`` needed.)
* Flask (for web UI)

### Datasets
#### MS COCO
#### Flickr 8k/30k
#### PASCAL


### Pre-trained models
#### Inception (v3, v4)
* Get checkpoints from https://github.com/tensorflow/models/tree/master/slim#Pretrained, and put the uncompressed checkpoint files in ``img2txt/pretrained``.
#### VGG16
* Copy Keras' pretrained model ``~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5`` to in ``img2txt/pretrained/vgg16_weights.h5``.

## Acknowledgements
