import h5py
import numpy as np
import tensorflow as tf
from convnet_config import vgg16

def build_vgg16(input_layer, minibatch_size):
    weights_f = h5py.File(
        vgg16['weights_file_path'],
        mode='r',
    )
    prev_layer = input_layer
    for block_name, block_conf in vgg16['network']:
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

def preprocess_image(convnet_name, image, size):
    if convnet_name == 'vgg16':
        image = image.resize((size, size))
        x = np.array(image, dtype=np.float32)
        if len(x.shape) == 2:
            raise NotImplementedError
        # Substracting the mean, from Keras' imagenet_utils.preprocess_input.
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    else:
        raise NotImplementedError

    return x

