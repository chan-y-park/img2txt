import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image

from inception_v3 import inception_v3
from inception_v4 import inception_v4
from inception_utils import inception_arg_scope

def build_inception(
    name,
    input_layer,
    minibatch_size,
    tf_session,
    tf_graph,
    pretrained_model_file_path=None,
    reuse=False,
    scope=None,
):
    if name == 'inception_v3':
        build_fn = inception_v3
    elif name == 'inception_v4':
        build_fn = inception_v4
    with slim.arg_scope(inception_arg_scope()):
        logits, endpoints = build_fn(
            input_layer,
            create_aux_logits=False,
            is_training=False,
            reuse=reuse,
        )
    if not reuse:
        var_dict = {}
        for var in tf_graph.get_collection('variables', scope=scope.name):
            var_name_prefix = '{}/'.format(scope.name)
            var_name_suffix = ':0'
            saved_var_name = var.name[len(var_name_prefix)
                                      :-len(var_name_suffix)]
            var_dict[saved_var_name] = var
        saver = tf.train.Saver(
            var_list=var_dict,
        )
        saver.restore(
            tf_session,
            save_path=pretrained_model_file_path,
        )

    return endpoints

def build_vgg16(
    input_layer,
    minibatch_size,
    pretrained_model_file_path='pretrained/vgg16_weights.h5'
):
    network_config = [
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
        ('top',
            (
                ('flatten', ()),
                ('fc1', (4096)),
                ('fc2', (4096)),
                ('predictions', (1000)),
            )
        ),
    ]
    weights_f = h5py.File(
        pretrained_model_file_path,
        mode='r',
    )
    prev_layer = input_layer
    for block_name, block_conf in network_config:
        with tf.variable_scope(block_name):
            for layer_name, layer_conf in block_conf:
                with tf.variable_scope(layer_name):
                    block_layer_name = block_name + '_' + layer_name
                    if 'conv' in layer_name:
                        conv_var = {}
                        for var_name, var_shape in layer_conf.items():
                            conv_var[var_name] = get_vgg16_weights(
                                weights_f,
                                block_layer_name,
                                var_name,
                                var_shape,
                            )
                        tensor = tf.nn.conv2d(
                            prev_layer,
                            conv_var['W'],
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                        )
                        tensor = tf.nn.bias_add(
                            tensor,
                            conv_var['b']
                        )
                        new_layer = tf.nn.relu(
                            tensor,
                        )
                    elif 'pool' in layer_name:
                        new_layer = tf.nn.max_pool(
                            prev_layer,
                            ksize=([1] + layer_conf['k'] + [1]),
                            strides=([1] + layer_conf['s'] + [1]),
                            padding='SAME',
                        )
                    elif 'flatten' in layer_name:
                        new_layer = tf.reshape(
                            prev_layer,
                            [minibatch_size, -1],
                        )
                    elif (
                        'fc' in layer_name
                        or 'predictions' in layer_name
                    ):
                        if 'fc' in layer_name:
                            f_layer = tf.nn.relu    
                        elif 'predictions' in layer_name:
                            f_layer = tf.nn.softmax

                        input_dim = prev_layer.shape[-1].value
                        output_dim = layer_conf
                        layer_var = {}
                        for var_name, var_shape in (
                            ('W', (input_dim, output_dim)),
                            ('b', (output_dim)),
                        ):
                            layer_var[var_name] = get_vgg16_weights(
                                weights_f,
                                layer_name,
                                var_name,
                                var_shape,
                            )
                        preactivation = tf.add(
                            tf.matmul(prev_layer, layer_var['W']),
                            layer_var['b'],
                            name='preactivation',
                        )
                        new_layer = f_layer(
                            preactivation,
                            name='activation',
                        )
                    
                    else:
                        raise NotImplementedError

                    # End of building a layer.
                    prev_layer = new_layer

    # End of building VGG16.


def get_vgg16_weights(weights_f, block_layer_name, var_name, var_shape):
    dset_name = block_layer_name + '_' + var_name + '_1:0'
    return tf.get_variable(     
        var_name,
        shape=var_shape,    
        initializer=tf.constant_initializer(
            weights_f  
            [block_layer_name]
            [dset_name]     
            .value 
        ),
        trainable=False,
    )

def resize_image(image, size, crop=True):
    if crop:
        width, height = image.size
        min_size = min(width, height)
        left = int((width - min_size) / 2.0)
        upper = int((height - min_size) / 2.0)
        image = image.crop((left, upper, left + min_size, upper + min_size))
    image = image.resize((size, size))
    return image


def preprocess_image(convnet_name, image, size=None):
    if size is not None:
        image = resize_image(image, size)

    width, height = image.size
    assert(width == height)

    x = np.array(image, dtype=np.float32)
    if len(x.shape) == 2:
        rgbimg = Image.new("RGB", image.size)
        rgbimg.paste(image)
        x = np.array(rgbimg, dtype=np.float32)

    if convnet_name == 'vgg16':
        assert(width == 224)
        # Substracting the mean, from Keras' imagenet_utils.preprocess_input.
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    elif 'inception' in convnet_name:
        assert(width == 299)
        x /= (np.iinfo(np.uint8).max / 2.0)
        x -= 1.0
    else:
        print('Unknown convolutional network: {}'.format(convnet_name))
        raise NotImplementedError

    return x

