from convnet_config import vgg16
from dataset_config import pascal

img2txt_using_lstm = {
    'log_dir': 'logs',
    'checkpoint_dir': 'checkpoints',
    'config_dir': 'configs',
    'dataset': pascal,
#    'minibatch_size': 3,
    'input_image_shape': [224, 224, 3],
    'vocabulary_size': None,
    'embedding_size': 10,
    'max_sequence_length': 20,
    'rnn_cell': {
        'type': 'lstm',
#        'type': 'lstm_block',
        'num_units': 10,
        'forget_bias': 1.0,
        'use_peepholes': False,
    },
    'variable_initializer': {
        'mean': 0,
        'stddev': 0.02,
    },
    'sgd': {
        'initial_learning_rate': 2.0,
        'learning_rate_decay_factor': 0.5,
        'num_epochs_per_decay': 8.0,
        'clip_gradients': 5.0,
    },
    'convnet': {
        'name': 'vgg16',
        'train_dataset': 'imagenet',
    },
    'num_training_iterations': 100,
}
