import os
import random
import time
import threading

import numpy as np
import tensorflow as tf

from dataset import PASCAL, Flickr
from convnet import build_vgg16

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
        self._config = config

        for directory in (
            self._config['log_dir'],
            self._config['checkpoint_dir'],
            self._config['config_dir'],
        ):
            if not os.path.exists(directory):
                os.makedirs(directory)

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
        input_image_size = self._config['input_image_shape'][0]
        cfg_dataset = self._config['dataset']
        dataset_name = cfg_dataset['name']
        # TODO: Use a TF queue.
        self._data_queue = []

        if dataset_name == 'pascal':
            dataset = PASCAL(
                caption_filename=cfg_dataset['caption_filename'],
                data_dir=cfg_dataset['data_dir'],
            )
        elif dataset_name == 'flickr_8k':
            dataset = Flickr(
                caption_filename=cfg_dataset['caption_filename'],
                data_dir=cfg_dataset['data_dir'],
            )
        else:
            raise ValueError('Unknown dataset name: {}.'.format(dataset_name))

#        num_processed = 0
#        total_num_to_process = len(dataset._img_ids)
#        for img_id in dataset._img_ids:
#            num_processed += 1
#            print(
#                '{}/{} of dataset {} processed.'
#                .format(num_processed, total_num_to_process, dataset_name),
#                end='\r',
#            )
#            captions = dataset.get_captions(img_id)
#            for caption_id in range(len(captions)):
#                image_array = dataset.get_image(
#                    img_id,
#                    to_array=True,
#                    size=input_image_size,
#                )
#                preprocessed_caption = dataset.get_preprocessed_caption(
#                    img_id,
#                    caption_id,
#                )
#                self._data_queue.append(
#                    (image_array, preprocessed_caption)
#                )
        num_processed = 0
        total_num_to_process = len(dataset._img_ids)
        for img_id in dataset._img_ids:
            captions = dataset.get_captions(img_id)
            for caption_id in range(len(captions)):

                self._data_queue.append(
                    (img_id, caption_id)
                )
        self._dataset = dataset

    def _build_input_queue(self):
        minibatch_size = self._config['minibatch_size']
        input_image_shape = self._config['input_image_shape']

        image = tf.placeholder(
            dtype=tf.float32,
            shape=input_image_shape,
            name='image',
        )
        caption = tf.placeholder(
            dtype=tf.int32,
            name='caption',
        )
        target = tf.placeholder(
            dtype=tf.int32,
            name='target',
        )
        mask = tf.placeholder(
            dtype=tf.int32,
            name='mask',
        )

        queue_capacity = 2 * minibatch_size

#        queue = tf.FIFOQueue(
        queue = tf.PaddingFIFOQueue(
            capacity=queue_capacity,
            dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
            shapes=[
                input_image_shape,
                [None],
                [None],
                [None],
            ],
            name='image_and_caption_queue',
        )

        close_op = queue.close(
            cancel_pending_enqueues=True,
            name='close_op',
        )

#        enqueue_op = queue.enqueue_many(
        enqueue_op = queue.enqueue(
            (image, caption, target, mask),
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
        input_image_shape = self._config['input_image_shape']
        vocabulary_size = self._config['vocabulary_size']
        embedding_size = self._config['embedding_size']

        with tf.variable_scope('embedding'):
            word_embedding = tf.get_variable(
                name='embedding',
                shape=[vocabulary_size, embedding_size],
                initializer=self._get_variable_initializer(),
            )
        # NOTE: Training runs for an unrolled RNN via tf.nn.dynamic_rnn,
        # inference runs for a single RNN cell pass.
        if self._training:
            # XXX: When using PaddedFIFOQueue, all captions are padded
            # to the same maximum sequence length.
            images, captions, targets, masks = [
                self._tf_graph.get_tensor_by_name(
                    'input_queue/dequeued_inputs:{}'.format(i),
                ) for i in range(4)
            ]
            # TODO: Don't hard-code the max sequence length,
            # but use a variable one for each minibatch.
#            max_sequence_length = self._config['max_sequence_length']
#            minibatch_sequence_length = minibatch_size * max_sequence_length

#            max_sequence_length = captions.shape.as_list()[1]

#            assert(captions.shape.as_list()
#                   == [minibatch_size, max_sequence_length])

            caption_embeddings = tf.nn.embedding_lookup(
                word_embedding,
                captions,
            )
#            assert(caption_embeddings.shape.as_list()
#                   == [minibatch_size, max_sequence_length, embedding_size])
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

        with tf.variable_scope('rnn'):
            # TODO: Use DNC instead of LSTM.
            cfg_rnn_cell = self._config['rnn_cell']
            # XXX Check RNN output size.
            rnn_output_size = cfg_rnn_cell['num_units']
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

            with tf.variable_scope('lstm_cell') as scope:
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
                    rnn_outputs, rnn_final_state = tf.nn.dynamic_rnn(
                        cell=rnn_cell,
                        inputs=caption_embeddings,
                        sequence_length = tf.reduce_sum(masks, axis=1),
                        initial_state=rnn_initial_state,
                        dtype=tf.float32,
                        scope=scope,
                    )
                    rnn_outputs = tf.reshape(
                        rnn_outputs,
                        [-1, rnn_output_size],
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
                word_logits, words = tf.nn.top_k(
                    word_log_probabilities,
                    k=1,
                    name='predictions',
                )

        if self._training:
            with tf.variable_scope('train'):
                loss_function = tf.nn.sparse_softmax_cross_entropy_with_logits
                targets = tf.reshape(targets, [-1])
                unmasked_losses = loss_function(
                    labels=targets,
                    logits=word_log_probabilities,
                    name='unmasked_losses',
                )

                masks = tf.to_float(tf.reshape(masks, [-1]))
                minibatch_loss = tf.div(
                    tf.reduce_sum(tf.multiply(unmasked_losses, masks)),
                    tf.reduce_sum(masks),
                    name='minibatch_loss'
                )
                tf.losses.add_loss(minibatch_loss)

                # TODO: Use learning rate decay and clipping.
                # Using contrib.layers.optimize_loss?
                sgd = tf.train.GradientDescentOptimizer(
                    learning_rate=self._config['sgd']['initial_learning_rate']
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
        minibatch_size = self._config['minibatch_size']
        embedding_size = self._config['embedding_size']
        convnet = self._config['convnet']

        if convnet == 'vgg16':
            convnet_top_layer = build_vgg16(
                input_images,
                minibatch_size,
            )
        else:
            raise NotImplementedError

        _, convnet_output_size = convnet_top_layer.shape.as_list()
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
            image_embeddings = tf.matmul(convnet_top_layer, W) + b

        return image_embeddings

    def _build_unrolled_rnn(
        self,
        rnn_cell,
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

        rnn_state = initial_state
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

    def _enqueue_thread(self):
        minibatch_size = self._config['minibatch_size']
        input_image_size = self._config['input_image_shape'][0]
        dataset = self._dataset

        num_data = len(self._data_queue)
        i = 0

        enqueue_op = self._tf_graph.get_operation_by_name(
            'input_queue/enqueue_op'
        )
        input_queue_image = self._tf_graph.get_tensor_by_name(
            'input_queue/image:0'
        )
        input_queue_caption = self._tf_graph.get_tensor_by_name(
            'input_queue/caption:0'
        )
        input_queue_target = self._tf_graph.get_tensor_by_name(
            'input_queue/target:0'
        )
        input_queue_mask = self._tf_graph.get_tensor_by_name(
            'input_queue/mask:0'
        )

        random.shuffle(self._data_queue)

        while not self._tf_coordinator.should_stop():
            if i >= num_data:
                i %= num_data
                random.shuffle(self._data_queue)
            data_to_enqueue = self._data_queue[i] 
            i += 1
            try:
#                image, caption = data_to_enqueue
                img_id, caption_id = data_to_enqueue
                image = dataset.get_image(
                    img_id,
                    to_array=True,
                    size=input_image_size,
                )
                caption = dataset.get_preprocessed_caption(
                    img_id,
                    caption_id,
                )
                input_sequence_length = len(caption) - 1
                mask = np.ones(input_sequence_length)
                self._tf_session.run(
                    enqueue_op,
                    feed_dict={
                        input_queue_image: image,
                        input_queue_caption: caption[:-1],
                        input_queue_target: caption[1:],
                        input_queue_mask: mask,
                    }
                )
            except tf.errors.CancelledError:
#                print('Input queue closed.')
                pass

    def _build_summary_ops(self):
        pass

    def train(
        self,
        run_name=None,
        max_num_iters=None,
        additional_num_iters=None,
    ):
        if not self._training:
            raise RuntimeError

        if run_name is None:
            run_name = (
                '{:02}{:02}_{:02}{:02}{:02}'.format(*time.localtime()[1:6])
            )

        summary_writer = tf.summary.FileWriter(
            logdir='{}/{}'.format(self._config['log_dir'], run_name),
            graph=self._tf_graph,
        )

        if self._iter is None:
            self._iter = 1
        if max_num_iters is not None:
            self._config['num_training_iterations'] = max_num_iters
        if additional_num_iters is not None:
           self._config['num_training_iterations'] += additional_num_iters

        num_training_iterations = self._config['num_training_iterations']
        display_iterations = num_training_iterations // 100
        save_iterations = num_training_iterations // 10

        queue_threads = [threading.Thread(target=self._enqueue_thread)]
        for t in queue_threads:
            t.start()
#        self._enqueue_thread()

        fetches = {}
        for var_name in [
            'train/minibatch_loss',
        ]:
            fetches[var_name] = self._tf_graph.get_tensor_by_name(
                var_name + ':0'
            )

        for op_name in [
            'train/minimize_loss',
        ]:
            fetches[op_name] = self._tf_graph.get_operation_by_name(op_name)

        try:
            for i in range(self._iter, num_training_iterations + 1):
                if self._tf_coordinator.should_stop():
                    break

                rd = self._tf_session.run(
                    fetches=fetches,
                )

#                summary_writer.add_summary(
#                    summary=rd['summary/merged/merged'],
#                    global_step=i,
#                )

                if i % display_iterations == 0:
                    print(
                        '{:g}% : L_x = {:g}'
                        .format(
                            (i / num_training_iterations * 100),
                            rd['train/minibatch_loss'],
                        ),
                    )
                if (
                    i % save_iterations == 0
                    or i == num_training_iterations
                ):
                    save_path = self._tf_saver.save(
                        self._tf_session,
                        'checkpoints/{}'.format(run_name),
                        i,
                    )
                    print('checkpoint saved at {}'.format(save_path))

            # End of iteration for-loop.

        except tf.errors.OutOfRangeError:
            raise RuntimeError

        finally:
            self._tf_coordinator.request_stop()
            self._tf_session.run(
                self._tf_graph.get_operation_by_name(
                    'input_queue/close_op'
                )
            )

        self._tf_coordinator.join(queue_threads)

        with open('{}/{}'.format(CFG_DIR, run_name), 'w') as fp:
            json.dump(self._config, fp)

        summary_writer.close()

        return rd

def get_step_from_checkpoint(save_path):
    return int(save_path.split('-')[-1])
