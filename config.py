from convnet_config import vgg16
from dataset_config import pascal

img2txt_using_lstm = {
    'dataset': pascal,
    'minibatch_size': 32,
    'input_image_shape': [224, 224, 3],
    'vocabulary_size': None,
    'embedding_size': 256,
    'max_sequence_length': 20,
    'rnn_cell': {
#        'type': 'lstm',
        'type': 'lstm_block',
        'num_units': 256,
        'forget_bias': 1.0,
        'use_peepholes': False,
        'dropout_keep_probability': 0.7,
    },
    'variable_initializer': {
        'mean': 0,
        'stddev': 0.02,
    },
    'optimizer': {
        'initial_learning_rate': 2.0,
        'learning_rate_decay_rate': 0.5,
        'num_epochs_per_decay': 8.0,
        'gradient_clip_norm': 5.0,
    },
    'convnet': {
        'name': 'vgg16',
        'train_dataset': 'imagenet',
    },
    'num_examples_per_epoch': None,
    'num_training_epochs': 15,
}
