import os
import numpy as np
import tensorflow as tf
import h5py

from img2txt_datasets import PASCAL, Flickr

class Image2Text:
    def __init__(
        self,
        config=None,
        training=None,
        gpu_memory_fraction=None,
        gpu_memory_allow_growth=True,
        save_path=None,
        save_var_list=None,
    ):
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        if not os.path.exists(CFG_DIR):
            os.makedirs(CFG_DIR)

        if training is None:
            raise ValueError('Set training either to be True or False.')
        else:
            self._training = training

        self._load_data()

        self._tf_session = None
        self._tf_coordinator = tf.train.Coordinator()

        self._tf_config = tf.ConfigProto()
        self._tf_config.gpu_options.allow_growth = gpu_memory_allow_growth
        if gpu_memory_fraction is not None:
            self._tf_config.gpu_options.per_process_gpu_memory_fraction = (
                gpu_memory_fraction
            )

        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            if self._training:
                with tf.variable_scope('input_queue'):
                    self._build_input_queue()

            self._build_network()

            if self._training:
                with tf.variable_scope('summary'):
                    self._build_summary_ops()

            self._tf_session = tf.Session(config=self._tf_config)
            self._tf_session.run(tf.global_variables_initializer())

            self._tf_saver = tf.train.Saver()
            if save_path is not None:
                self._tf_saver.restore(self._tf_session, save_path)
                self._iter = get_step_from_checkpoint(save_path)
            else:
                self._iter = None

    def _load_data(self):
        cfg_dataset = self._config['dataset']
        dataset_name = cfg_dataset['name']

        if dataset_name == 'pascal':
            dataset = PASCAL(
                caption_filename=cfg_dataset['caption_filename']
                data_dir=cfg_dataset['data_dir']
            )
        elif dataset_name == 'flickr_8k':
            dataset = Flickr(
                caption_filename=cfg_dataset['caption_filename']
                data_dir=cfg_dataset['data_dir']
            )
        else:
            raise ValueError('Unknown dataset name: {}.'.format(dataset_name))

    def _build_input_queue(self):
        minibatch_size = self._config['minibatch_size']
        input_image_shape = self.config['input_image_shape']

        images = tf.placeholder(
            dtype=tf.float32,
            shape=(minibatch_size, *input_image_shape),
            name='images',
        )
        captions = tf.placeholder(
            dtype=tf.int32,
            shape=(minibatch_size, None),
            name='captions',
        )
        targets = tf.placeholder(
            dtype=tf.int32,
            shape=(minibatch_size, None),
            name='targets',
        )
        masks = tf.placeholder(
            dtype=tf.int32,
            shape=(minibatch_size, None),
            name='masks',
        )

        queue_capacity = 2 * minibatch_size

#        queue = tf.FIFOQueue(
        queue = tf.PaddingFIFOQueue(
            capacity=queue_capacity,
            dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
            shapes=[
                (minibatch_size, *input_image_shape),
                (minibatch_size, None),
                (minibatch_size, None),
                (minibatch_size, None),
            ],
            name='image_and_caption_queue',
        )

        close_op = queue.close(
            cancel_pending_enqueues=True,
            name='close_op',
        )

        enqueue_op = queue.enqueue_many(
            (images, captions, targets, masks),
            name='enqueue_op',
        )

        dequeued_inputs = queue.dequeue_many(
            minibatch_size,
            name='dequeued_inputs',
        )

        size_op = queue.size(
            name='size',
        )

    def _build_network(self):
        minibatch_size = self._config['minibatch_size']
        input_image_shape = self.config['input_image_shape']
        vocabulary_size = self._config['vocabulary_size']
        embedding_size = self._config['embedding_size']

        with tf.variable_scope('embedding'):
            word_embedding = tf.get_variable(
                name='embedding',
                shape=[vocabulrary_size, embedding_size],
                initializer=self._get_variable_initializer(),
            )

        if self._training:
#            images, captions = self._tf_graph.get_tensor_by_name(
#                'input_queue/dequeued_inputs:0',
#            )
#
#            # TODO: Don't hard-code the max sequence length,
#            # but use a variable one for each minibatch.
#            max_sequence_length = self._config['max_sequence_length']
#            minibatch_sequence_length = minibatch_size * max_sequence_length
#
#            images, captions, targets, masks = self._get_padded_inputs(captions)
#
#            assert(captions.shape.as_list()
#                   == [minibatch_size, max_sequence_length])
#            caption_embeddings = tf.nn.embedding_lookup(
#                word_embedding,
#                captions,
#            )
#            assert(caption_embeddings.shape.as_list()
#                   == [minibatch_size, max_sequence_length, embedding_size])

            (images, captions,
             targets, masks) = self._tf_graph.get_tensor_by_name(
                'input_queue/dequeued_inputs:0',
            )
            caption_embeddings = tf.nn.embedding_lookup(
                word_embedding,
                captions,
            )
        else:
            images = tf.placeholder(
                dtype=tf.float32,
                shape=input_image_shape,
                name='input_image',
            )
            rnn_state = tf.placeholder(
                dtype=tf.float32,
                #shape=,
                name='rnn_state',
            )
            prev_word = tf.placeholder(
                dtype=tf.int32,
                #shape=,
                name='prev_word',
            )
            input_embedding = tf.nn.embedding_lookup(
                word_embedding,
                prev_word,
            )

        with tf.variable_scope('convnet'):
            image_embeddings = self._build_convnet(
                images,
            )

        with tf.variable_scope('rnn') as scope:
            # TODO: Use DNC instead of LSTM.
            cfg_rnn_cell = self._config['rnn_cell']
            lstm_kwargs = {
                'num_units': cfg_rnn_cell['num_units'],
                'forget_bias': cfg_rnn_cell['forget_bias'],
            }
            # TODO: Use LSTMBlockCell.
            if cfg_rnn_cell['type'] == 'lstm_block':
                tf_lstm_cell = tf.contrib.rnn.LSTMBlockCell
                lstm_kwargs['use_peephole'] = cfg_rnn_cell['use_peepholes']
            elif cfg_rnn_cell['type'] == 'lstm':
                tf_lstm_cell = tf.nn.rnn_cell.LSTMCell
                lstm_kwargs['use_peepholes'] = cfg_rnn_cell['use_peepholes']
            else:
                raise ValueError

            rnn_cell = tf_lstm_cell(**lstm_kwargs)
            rnn_zero_state = rnn_cell.zero_state(
                batch_size=minibatch_size,
                dtype=tf.float32,
            )
            _, rnn_initial_state = rnn_cell(
                image_embeddings,
                rnn_zero_state,
            )

            scope.reuse_variables()

            if self._training:
                # TODO: Use dynamic_rnn.
                rnn_outputs = self._build_unrolled_rnn(
                    rnn_cell,
                    caption_embeddings,
                    rnn_initial_state,
                    max_sequence_length,
                    scope,
                )
                assert(
                    rnn_outputs.shape.to_list()
                    == [minibatche_size, max_sequence_length, rnn_output_size])
                )
                rnn_outputs = tf.reshape(
                    rnn_outputs,
                    (minibatch_sequence_length, rnn_output_size),
                )
                masks = tf.reshape(
                    masks,
                    (minibatch_sequence_length), 
                )
            else:
                rnn_output, rnn_state = rnn_cell(
                    input_embedding,
                    rnn_state,
                )

            with tf.variable_scope('fc'):
                W = tf.get_variable(
                    name='W',
                    shape=(rnn_output_size, vocabulary_size),
                    initializer=self._get_variable_initializer(),
                )
                b = tf.get_variable(
                    name='b',
                    shape=(vocabulary_size),
                    initializer=self._get_variable_initializer(),
                )
                word_log_probabilities = tf.matmul(rnn_outputs, W) + b
                word_logits, words = tf.nn.top_k(word_log_probabilities, k=1)

        if self.training:
            with tf.variable_scope('train'):
                loss_function = tf.nn.sparse_softmax_cross_entropy_with_logits
                unmasked_losses = loss_function(
                    labels=targets,
                    logits=word_logits,
                    name='unmasked_losses',
                )
                minibatch_loss = tf.div(
                    tf.reduce_sum(tf.multiply(unmasked_losses, masks)),
                    tf.reduce_sum(masks),
                    name='minibatch_loss'
                )
                tf.losses.add_loss(minibatch_loss)

                # TODO: Use learning rate decay and clipping.
                # Using contrib.layers.optimize_loss?
                sgd = tf.train.GradientDescentOptimizer(
                    learning_rate=self._config['sgd']['learning_rate']
                )
                train_op = sgd.minimize(
                    loss=tf.losses.get_total_loss(),
                    name='minimize_loss'
                )
        else:
            word_probabilities = tf.nn.softmax(
                word_log_probabilities,
                name='word_probabilities',
            )
            

    def _get_padded_inputs(self, captions):
        pass

    def _build_convnet(self, input_images):
        prev_layer = input_images
        cfg_convnet = self._config['convnet']

        if cfg_convnet['name'] == 'VGG16':
            for block_name, block_conf in cfg_convnet['network']:
                with tf.variable_scope(block_name):
                    for layer_name, layer_conf in block_conf:
                        with tf.variable_scope(layer_name):
                            block_layer_name = block_name + '_' + layer_name
                            if 'conv' in layer_name:
                                conv_var = {}
                                for var_name, var_shape in layer_conf.items():
                                    conv_var[var_name] = self._get_weights(
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
                                rv = tf.nn.max_pool(
                                    prev_layer,
                                    ksize=([1] + layer_conf['k'] + [1]),
                                    strides=([1] + layer_conf['s'] + [1]),
                                    padding='SAME',
                                )
                                if self.visualize:
                                    new_layer, switches = rv
                                    self._max_pool_switches[
                                        block_layer_name
                                    ] = switches
                                else:
                                    new_layer = rv

                            elif 'flatten' in layer_name:
                                new_layer = tf.reshape(
                                    prev_layer,
                                    [self.batch_size, -1],
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
                                    layer_var[var_name] = self._get_weights(
                                        layer_name,
                                        var_name,
                                        var_shape,
                                    )
                                activation = tf.add(
                                    tf.matmul(prev_layer, layer_var['W']),
                                    layer_var['b'],
                                    name='activation',
                                )
                                new_layer = f_layer(
                                    activation,
                                )
                            
                            else:
                                raise NotImplementedError

                            # End of building a layer.

                            self._forward_layers[block_layer_name] = new_layer
                            prev_layer = new_layer

        # End of building a convnet.

        minibatch_size, convnet_output_size = prev_layer.shape.to_list()
        with tf.variable_scope('image_embedding'):
            W = tf.get_variable(
                name='W',
                shape=(convnet_output_size, embedding_size),
                initializer=self._get_variable_initializer(),
            )
            b = tf.get_variable(
                name='b',
                shape=(embedding_size),
                initializer=self._get_variable_initializer(),
            )
            image_embeddings = tf.matmul(rnn_outputs, W) + b

        return image_embeddings

    def _build_unrolled_rnn(
        self,
        cell,
        inputs,
        initial_state,
        max_sequence_length,
        scope,
    ):
        rnn_outputs = []
        input_splits = tf.split(
            inputs,
            max_sequence_length,
            axis=1,
        )

        for t in range(max_sequence_length):
            rnn_output, rnn_state = rnn_cell(
                input_splits[t],
                rnn_state,
            )
            rnn_outputs.append(rnn_output)

        return tf.stack(rnn_outputs, axis=1)

    def _get_variable_initializer(self):
        return tf.truncated_normal_initializer(
            dtype=tf.float32,
            **self._config['variable_initializer']
        )
