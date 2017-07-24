img2txt_using_lstm ={
    'dataset': 'pascal',
    'minibatch_size': 3,
    'input_image_shape': [224, 224, 3],
    'vocabulary_size': 100,
    'embedding_size': 10,
    'max_sequence_length': 10,
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
    'convnet': vgg16_full_config,
}

vgg16_no_fc_config = {
    'weights_file_path': (
        'pretrained/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    ),
    'network': [
        ('block1',
            (
                ('conv1', {'W': [3, 3, 3, 64], 'b': [64]}),
                ('conv2', {'W': [3, 3, 64, 64], 'b': [64]}),
                ('pool', {'k': [2, 2], 's': [2, 2]}),
            )
        ),
        ('block2',
            (
                ('conv1', {'W': [3, 3, 64, 128], 'b': [128]}),
                ('conv2', {'W': [3, 3, 128, 128], 'b': [128]}),
                ('pool', {'k': [2, 2], 's': [2, 2]}),
            )
        ),
        ('block3',
            (
                ('conv1', {'W': [3, 3, 128, 256], 'b': [256]}),
                ('conv2', {'W': [3, 3, 256, 256], 'b': [256]}),
                ('conv3', {'W': [3, 3, 256, 256], 'b': [256]}),
                ('pool', {'k': [2, 2], 's': [2, 2]}),
            )
        ),
        ('block4',
            (
                ('conv1', {'W': [3, 3, 256, 512], 'b': [512]}),
                ('conv2', {'W': [3, 3, 512, 512], 'b': [512]}),
                ('conv3', {'W': [3, 3, 512, 512], 'b': [512]}),
                ('pool', {'k': [2, 2], 's': [2, 2]}),
            )
        ),
        ('block5', 
            (
                ('conv1', {'W': [3, 3, 512, 512], 'b': [512]}),
                ('conv2', {'W': [3, 3, 512, 512], 'b': [512]}),
                ('conv3', {'W': [3, 3, 512, 512], 'b': [512]}),
                ('pool', {'k': [2, 2], 's': [2, 2]}),
            )
        ),
    ],
}


vgg16_full_config = {
    'weights_file_path': (
        'pretrained/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    ),
    'network': vgg16_no_fc_config['network'] + [
        ('top',
            (
                ('flatten', ()),
                ('fc1', (4096)),
                ('fc2', (4096)),
                ('predictions', (1000)),
            )
        ),
    ]
}
