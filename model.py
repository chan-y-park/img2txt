import os
import random
import time
import threading

import numpy as np
import tensorflow as tf

from dataset import PASCAL, Flickr
from convnet import build_vgg16

# TODO:
# Check inference.
# Use dropout wrapper for lstm.
# Use beam search in sentence generation.
# Implement file name queue.
# Check input queue consumption and add more enqueue threads.

# XXX: To transfer learning between different datasets, 
# need to have a single vocabulary for all datasets.


class Image2Text:
    def __init__(
        self,
        config=None,
        training=None,
        gpu_memory_fraction=None,
        gpu_memory_allow_growth=True,
        save_path=None,
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

            # List of variables to save and restore using tf.train.Saver.
            self._saver_var_list = self._tf_graph.get_collection(
                'trainable_variables'
            )
            self._tf_saver = tf.train.Saver(
                var_list=self._saver_var_list,
            )
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

        self._config['vocabulary_size'] = dataset.get_vocabulary_size()
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
        input_seq = tf.placeholder(
            dtype=tf.int32,
            name='input_seq',
        )
        target_seq = tf.placeholder(
            dtype=tf.int32,
            name='target_seq',
        )
        mask = tf.placeholder(
            dtype=tf.int32,
            name='mask',
        )

        queue_capacity = 2 * minibatch_size

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

        enqueue_op = queue.enqueue(
            (image, input_seq, target_seq, mask),
            name='enqueue_op',
        )

        dequeued_inputs = queue.dequeue_many(
            minibatch_size,
            name='dequeued_inputs',
        )

        size_op = queue.size(
            name='size',
        )

    def _build_network(self, minibatch_size=None):
        if minibatch_size is None:
            minibatch_size = self._config['minibatch_size']
        input_image_shape = self._config['input_image_shape']
        vocabulary_size = self._config['vocabulary_size']
        embedding_size = self._config['embedding_size']

        # NOTE: Training runs for an unrolled RNN via tf.nn.dynamic_rnn,
        # inference runs for a single RNN cell pass.
        if self._training:
            # XXX: When using PaddedFIFOQueue, all captions are padded
            # to the same maximum sequence length.
            images, input_seqs, target_seqs, masks = [
                self._tf_graph.get_tensor_by_name(
                    'input_queue/dequeued_inputs:{}'.format(i),
                ) for i in range(4)
            ]
        else:
            images = tf.placeholder(
                dtype=tf.float32,
                shape=([minibatch_size] + input_image_shape),
                name='input_images',
            )
            input_seqs = tf.placeholder(
                dtype=tf.int32,
                shape=[minibatch_size, 1],
                name='input_seqs',
            )

        with tf.variable_scope('convnet'):
            image_embeddings = self._build_convnet(
                images,
            )

        with tf.variable_scope('rnn'):
            word_embedding = tf.get_variable(
                name='word_embedding',
                shape=[vocabulary_size, embedding_size],
                initializer=self._get_variable_initializer(),
            )
            input_embeddings = tf.nn.embedding_lookup(
                word_embedding,
                input_seqs,
            )

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
                rnn_zero_states = rnn_cell.zero_state(
                    batch_size=minibatch_size,
                    dtype=tf.float32,
                )
                _, rnn_initial_states = rnn_cell(
                    image_embeddings,
                    rnn_zero_states,
                )

                scope.reuse_variables()

                if self._training:
                    rnn_outputs, rnn_final_state = tf.nn.dynamic_rnn(
                        cell=rnn_cell,
                        inputs=input_embeddings,
                        sequence_length = tf.reduce_sum(masks, axis=1),
                        initial_state=rnn_initial_states,
                        dtype=tf.float32,
                        scope=scope,
                    )
                else:
                    prev_rnn_states = tf.placeholder(
                        dtype=tf.float32,
                        shape=[minibatch_size, sum(rnn_cell.state_size)],
                        name='prev_rnn_states',
                    )
                    rnn_output, rnn_state = rnn_cell(
                        input_embeddings,
                        tf.split(
                            value=prev_rnn_states,
                            num_or_size_splits=2,
                            axis=1,
                        ),
                    )

            if self._training:
                tf.identity(
                    rnn_outputs,
                    name='dynamic_rnn_outputs',
                )
                rnn_output = tf.reshape(
                    rnn_outputs,
                    [-1, rnn_output_size],
                    name='reshaped_rnn_output',
                )
            else:
                tf.concat(
                    axis=1,
                    value=rnn_initial_state,
                    name='rnn_initial_state',
                )
                tf.concat(
                    axis=1,
                    value=rnn_state,
                    name='rnn_state',
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
                word_log_probabilities = tf.add(
                    tf.matmul(rnn_output, W),
                    b,
                    name='word_log_probabilities',
                )
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

    def _build_convnet(self, input_images):
        minibatch_size = self._config['minibatch_size']
        embedding_size = self._config['embedding_size']
        convnet_name = self._config['convnet']['name']

        if convnet_name == 'vgg16':
            convnet_top_layer = build_vgg16(
                input_images,
                minibatch_size,
            )
        else:
            raise NotImplementedError

        tf.identity(
            convnet_top_layer,
            name='predictions',
        )

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
            image_embeddings = tf.add(
                tf.matmul(convnet_top_layer, W),
                b,
                name='image_embeddings',
            )

        return image_embeddings

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
        image = self._tf_graph.get_tensor_by_name(
            'input_queue/image:0'
        )
        input_seq = self._tf_graph.get_tensor_by_name(
            'input_queue/input_seq:0'
        )
        target_seq = self._tf_graph.get_tensor_by_name(
            'input_queue/target_seq:0'
        )
        mask = self._tf_graph.get_tensor_by_name(
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
                        image: image,
                        input_seq: caption[:-1],
                        target_seq: caption[1:],
                        mask: mask,
                    }
                )
            except tf.errors.CancelledError:
                print('Input queue closed.')

    def _build_summary_ops(self):
        summaries = [
            tf.summary.scalar(
                name='minibatch_loss',
                tensor=self._tf_graph.get_tensor_by_name(
                    'train/minibatch_loss:0'
                )
            ),
            tf.summary.scalar(
                name='queue_size',
                tensor=self._tf_graph.get_tensor_by_name(
                    'input_queue/size:0'
                ),
            )
        ]
        summary_op = tf.summary.merge(
            summaries,
            name='merged',
        )

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

        fetches = {}
        for var_name in [
            'train/minibatch_loss',
            'convnet/image_embedding/image_embeddings',
            'convnet/predictions',
            'rnn/dynamic_rnn_outputs',
            'rnn/reshaped_rnn_output',
            'rnn/fc/word_log_probabilities',
            'summary/merged/merged',
        ]:
            fetches[var_name] = self._tf_graph.get_tensor_by_name(
                var_name + ':0'
            )

        for i, var_name in enumerate(
            ['images', 'input_seqs', 'target_seqs', 'masks']
        ):
            fetches[var_name] = self._tf_graph.get_tensor_by_name(
                'input_queue/dequeued_inputs:{}'.format(i),
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

                summary_writer.add_summary(
                    summary=rd['summary/merged/merged'],
                    global_step=i,
                )

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

        with open('{}/{}'.format(self.config['cfg_dir'], run_name), 'w') as fp:
            json.dump(self._config, fp)

        summary_writer.close()

        return rd

    def generate_text(self, images):
        max_sequence_length = self._config['max_sequence_length']
        # Feed images, fetch RNN initial states.

        # For max_sequence_length, feed input seqs
        # and fetch word probabilities & new RNN states.

    def decode_convnet_predictions(self, predictions):
        convnet_train_dataset = self._config['convnet']['train_dataset']
        if convnet_train_dataset == 'imagenet':
            from keras.applications.imagenet_utils import decode_predictions
        else:
            raise NotImplementedError
        
        return decode_predictions(pred)

def get_step_from_checkpoint(save_path):
    return int(save_path.split('-')[-1])


