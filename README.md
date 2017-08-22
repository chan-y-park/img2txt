# ``img2txt``
End-to-end deep learning model to generate a summary of the content of an image in a sentence.

[Overview](#overview)

[Requirements](#requirements)

[How to run ``img2txt``](#how-to-run-img2txt)


## Overview

For a quick overview, please see [the slides for the 5-min demo of this project](https://docs.google.com/presentation/d/15HSGZaFE7pUj2iNZjHJtZn6TfuuKw8dXq3zad61jTfQ/edit?usp=sharing).

### Acknowledgement

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

And its web UI requires the following libraries.
* Flask
* Bokeh (for word embedding visualization)

### Datasets
``img2txt.dataset`` contains convenient wrappers for various public caption datasets including MS COCO, Flickr 8k/30k, and PASCAL. Put each downloaded dataset in a separate directory, which will be used during the training of the model.

### Using pre-trained convnet models
#### Inception (v3, v4)
* Get checkpoints from https://github.com/tensorflow/models/tree/master/slim#Pretrained, and put the uncompressed checkpoint files in ``img2txt/pretrained``.
#### VGG16
* Copy Keras' pretrained model ``~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5`` to in ``img2txt/pretrained/vgg16_weights.h5``.

## How to run ``img2txt``

### Training
Please see https://github.com/chan-y-park/img2txt/blob/master/img2txt_api_example.ipynb for a step-by-step guide.

### Inference
After training the model, put saved files in ``img2txt/inference``, more specifically
* the checkpoint files as ``img2txt/inference/img2txt.*``, 
* the configuration file as ``img2txt/inference/config.json``, and 
* the vocabulary file as ``img2txt/inference/vocabulary.json``. 
The run ``img2txt/web_app.wsgi``, open a web browser, and go to http://localhost:9999 to use the web UI for inference.
![web_ui_screenshot](https://github.com/chan-y-park/img2txt/blob/master/web_ui_screenshot.png "web UI screenshot")

## Performance
When trained on MS COCO training dataset for 500k weight updates, where each update is a training on a minibatch of 32 image-caption pairs, the model gets 25.9 BLEU-4 score and 86.4 CIDEr score, which are evaluated using 4k random selections from MS COCO validation dataset and MS COCO Caption Evaluation API (https://github.com/tylin/coco-caption).
