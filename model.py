import os
import random
import time
import threading
import json
import queue

import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import (
    decode_predictions
)

#from dataset import PASCAL, Flickr
from convnet import build_vgg16, build_inception, preprocess_image


LOG_DIR = 'logs'
CHECKPOINT_DIR = 'checkpoints'
CONFIG_DIR = 'configs'
NUM_ENQUEUE_THREADS = 2


class Image2Text:
    def __init__(
        self,
        config=None,
#        training=None,
        training_dataset=None,
        validation_dataset=None,
        vocabulary=None,
#        minibatch_size=None,
        gpu_memory_fraction=None,
        gpu_memory_allow_growth=True,
        save_path=None,
    ):
        if save_path is not None: 
            run_name, steps = parse_checkpoint_save_path(save_path)
            self._step = get_step_from_checkpoint(save_path)
            if config is None:
                with open('{}/{}'.format(CONFIG_DIR, run_name), 'r') as fp:
                    config = json.load(fp)
        else:
            self._step = None
                
        self._config = config

        for directory in (
            LOG_DIR,
            CHECKPOINT_DIR,
            CONFIG_DIR,
        ):
            if not os.path.exists(directory):
                os.makedirs(directory)

#        if training is None:
#            raise ValueError('Set training either to be True or False.')
#        else:
#            self._training = training

        self._training_dataset = training_dataset
        self._validation_dataset = validation_dataset
        self._vocabulary = vocabulary

        minibatch_size = self._config['minibatch_size']
        data_queue_size = 4 * minibatch_size * NUM_ENQUEUE_THREADS
        self._data_queue = queue.Queue(maxsize=data_queue_size)
#        self._load_dataset()

        self._tf_session = None
        self._tf_coordinator = None 

        self._tf_config = tf.ConfigProto()
        self._tf_config.gpu_options.allow_growth = gpu_memory_allow_growth
        if gpu_memory_fraction is not None:
            self._tf_config.gpu_options.per_process_gpu_memory_fraction = (
                gpu_memory_fraction
            )

        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            self._tf_session = tf.Session(config=self._tf_config)

#            if self._training:
#                with tf.variable_scope('input_queue'):
#                    self._build_input_queue()
            with tf.variable_scope('input_queue'):
                self._build_input_queue()

            self._build_network()

#            if self._training:
#                with tf.variable_scope('summary'):
#                    self._build_summary_ops()
            with tf.variable_scope('summary'):
                self._build_summary_ops()

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

#    def _load_dataset(self):
#        minibatch_size = self._config['minibatch_size']
#        cfg_dataset = self._config['dataset']
#        dataset_name = cfg_dataset['name']
#        # TODO: Use a TF queue.
#        data_queue_size = 4 * minibatch_size * NUM_ENQUEUE_THREADS
#        self._data_queue = queue.Queue(maxsize=data_queue_size)
#
#        if dataset_name == 'pascal':
#            dataset = PASCAL(
#                caption_filename=cfg_dataset['caption_filename'],
#                data_dir=cfg_dataset['data_dir'],
#            )
#        elif dataset_name == 'flickr_8k':
#            dataset = Flickr(
#                caption_filename=cfg_dataset['caption_filename'],
#                data_dir=cfg_dataset['data_dir'],
#            )
#        else:
#            raise ValueError('Unknown dataset name: {}.'.format(dataset_name))
#
#        # XXX: Move to dataset API
#        self._config['vocabulary_size'] = dataset.get_vocabulary_size()
#        for img_id in dataset._img_ids:
#            captions = dataset.get_captions(img_id)
#            for caption_id in range(len(captions)):
#                self._data_queue.append(
#                    (img_id, caption_id)
#                )
#        self._dataset = dataset
#        self._config['num_examples_per_epoch'] = len(self._data_queue)

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

        queue_capacity = 2 * minibatch_size * NUM_ENQUEUE_THREADS

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

    def _build_network(self):
        minibatch_size = self._config['minibatch_size']
        input_image_shape = self._config['input_image_shape']
        embedding_size = self._config['embedding_size']
        vocabulary_size = self._vocabulary.get_size() 

        # NOTE: Training runs for an unrolled RNN via tf.nn.dynamic_rnn,
        # inference runs for a single RNN cell pass.

        # Training.
        # NOTE: When using PaddedFIFOQueue, all captions are padded
        # to the same maximum sequence length.
        images, input_seqs, target_seqs, masks = [
            self._tf_graph.get_tensor_by_name(
                'input_queue/dequeued_inputs:{}'.format(i),
            ) for i in range(4)
        ]

        # Inference.
        inference_input_images = tf.placeholder(
            dtype=tf.float32,
            shape=([minibatch_size] + input_image_shape),
            name='inference_input_images',
        )
        inference_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=[minibatch_size, 1],
            name='inference_inputs',
        )

        with tf.variable_scope('convnet') as scope:
            # Training.
            image_embeddings = self._build_convnet(
                images,
                scope=scope,
            )

            # Inference.
            scope.reuse_variables()
            inference_image_embeddings = self._build_convnet(
                inference_input_images,
                reuse=True,
                scope=scope,
            )

        with tf.variable_scope('rnn'):
            word_embedding = tf.get_variable(
                name='word_embedding',
                shape=[vocabulary_size, embedding_size],
                initializer=self._get_variable_initializer(),
            )

            # Training.
            input_embeddings = tf.nn.embedding_lookup(
                word_embedding,
                input_seqs,
            )

            # Inference.
            inference_input_embeddings = tf.nn.embedding_lookup(
                word_embedding,
                inference_inputs,
            )
            inference_input_embeddings = tf.squeeze(
                input_embeddings,
                axis=1,
                name='inference_input_embedding',
            )
            inference_prev_rnn_states = tf.placeholder(
                dtype=tf.float32,
                shape=[minibatch_size,
                       (2 * self._config['rnn_cell']['num_units'])],
                name='inference_prev_states',
            )
            inference_prev_rnn_states = tf.split(
                inference_prev_rnn_states,
                num_or_size_splits=2,
                axis=1,
            )

            # TODO: Use DNC instead of LSTM.
            cfg_rnn_cell = self._config['rnn_cell']
            rnn_output_size = cfg_rnn_cell['num_units']
            lstm_kwargs = {
                'num_units': cfg_rnn_cell['num_units'],
                'forget_bias': cfg_rnn_cell['forget_bias'],
            }
            if cfg_rnn_cell['type'] == 'lstm_block':
                tf_lstm_cell = tf.contrib.rnn.LSTMBlockCell
                lstm_kwargs['use_peephole'] = cfg_rnn_cell['use_peepholes']
            elif cfg_rnn_cell['type'] == 'lstm':
                tf_lstm_cell = tf.nn.rnn_cell.LSTMCell
                lstm_kwargs['use_peepholes'] = cfg_rnn_cell['use_peepholes']
            else:
                raise ValueError

#            with tf.variable_scope('lstm_cell') as scope:
            with tf.variable_scope('cell') as scope:
                rnn_cell = tf_lstm_cell(**lstm_kwargs)
#                if self._training:
#                    keep_prob = cfg_rnn_cell['dropout_keep_probability']
#                    rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
#                        rnn_cell,
#                        input_keep_prob=keep_prob,
#                        output_keep_prob=keep_prob,
#                        state_keep_prob=1.0,
#                        variational_recurrent=False,
#                    )

                # Training.
                keep_prob = cfg_rnn_cell['dropout_keep_probability']
                dropout_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
                    rnn_cell,
                    input_keep_prob=keep_prob,
                    output_keep_prob=keep_prob,
                    state_keep_prob=1.0,
                    variational_recurrent=False,
                )

                rnn_zero_states = rnn_cell.zero_state(
                    batch_size=minibatch_size,
                    dtype=tf.float32,
                )
                _, rnn_initial_states = dropout_rnn_cell(
                    image_embeddings,
                    rnn_zero_states,
                )

                scope.reuse_variables()

                rnn_outputs, rnn_final_state = tf.nn.dynamic_rnn(
                    cell=dropout_rnn_cell,
                    inputs=input_embeddings,
                    sequence_length = tf.reduce_sum(masks, axis=1),
                    initial_state=rnn_initial_states,
                    dtype=tf.float32,
                    scope=scope,
                )

                # Inference.
                inference_rnn_zero_states = rnn_cell.zero_state(
                    batch_size=minibatch_size,
                    dtype=tf.float32,
                )
                _, inference_rnn_initial_states = rnn_cell(
                    image_embeddings,
                    inference_rnn_zero_states,
                )

                inference_rnn_outputs, inference_new_rnn_states = rnn_cell(
                    inference_input_embeddings,
                    inference_prev_rnn_states,
                )

            # Training.
            tf.identity(
                rnn_outputs,
                name='dynamic_rnn_outputs',
            )
            rnn_outputs = tf.reshape(
                rnn_outputs,
                [-1, rnn_output_size],
                name='reshaped_rnn_output',
            )

            #Inference.
            tf.concat(
                inference_rnn_initial_states,
                axis=1,
                name='inference_initial_states',
            )
            tf.concat(
                inference_new_rnn_states,
                axis=1,
                name='inference_new_states',
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

                # Training.
                word_log_probabilities = tf.add(
                    tf.matmul(rnn_outputs, W),
                    b,
                    name='word_log_probabilities',
                )
                word_logits, word_ids = tf.nn.top_k(
                    word_log_probabilities,
                    k=1,
#                    name='predictions',
                )

                # Inference.
                inference_word_log_probabilities = tf.add(
                    tf.matmul(inference_rnn_outputs, W),
                    b,
                    name='inference_word_log_probabilities',
                )
                inference_word_probabilities = tf.nn.softmax(
                    inference_word_log_probabilities,
                    name='inference_word_probabilities',
                )
                inference_word_logits, inference_word_ids = tf.nn.top_k(
                    inference_word_log_probabilities,
                    k=1,
                    name='inference_predictions',
                )

        # Training.
        output_seqs = tf.reshape(
            word_ids,
            shape=[minibatch_size, -1],
            name='output_seqs'
        )

        # Inference.
        inference_outputs = tf.reshape(
            inference_word_ids,
            shape=[minibatch_size, -1],
            name='inference_outputs'
        )

        loss_function = tf.nn.sparse_softmax_cross_entropy_with_logits
        with tf.variable_scope('train'):
            lr = tf.placeholder(
                dtype=tf.float32,
                shape=[],
                name='learning_rate',
            )
            targets = tf.reshape(target_seqs, [-1])
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
#            tf.losses.add_loss(minibatch_loss)

            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=lr,
            )
            grads_and_vars = optimizer.compute_gradients(
                loss=minibatch_loss,
            )
            gradients, variables = zip(*grads_and_vars)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients,
                self._config['optimizer']['gradient_clip_norm'],
            )
            clipped_grads_and_vars = list(
                zip(clipped_gradients, variables)
            )
            train_op = optimizer.apply_gradients(
                clipped_grads_and_vars,
                name='minimize_loss',
            )

        with tf.variable_scope('eval'):
            eval_targets = tf.placeholder(
                dtype=tf.int32,
                shape=[minibatch_size, 1],
                name='targets',
            )
            eval_masks = tf.placeholder(
                dtype=tf.int32,
                shape=[minibatch_size, 1],
                name='masks',
            )

            eval_targets = tf.reshape(eval_targets, [-1])
            unmasked_eval_losses = loss_function(
                labels=eval_targets,
                logits=inference_word_log_probabilities,
                name='unmasked_losses',
            )

            eval_masks = tf.to_float(tf.reshape(eval_masks, [-1]))
            eval_loss = tf.div(
                tf.reduce_sum(tf.multiply(unmasked_losses, masks)),
                tf.reduce_sum(masks),
                name='loss'
            )

            # Placeholders for evaluation summaries.
            input_image_ids = tf.placeholder(
                dtype=tf.string,
                name='input_image_ids',
            )
            target_sentences = tf.placeholder(
                dtype=tf.string,
                name='target_sentences',
            )
            output_sentences = tf.placeholder(
                dtype=tf.string,
                name='output_sentences',
            )
            eval_total_loss = tf.placeholder(
                dtype=tf.int32,
                name='total_loss',
            )

    def _build_convnet(self, input_images, reuse=None, scope=None):
        minibatch_size = self._config['minibatch_size']
        embedding_size = self._config['embedding_size']
        convnet_cfg = self._config['convnet']
        name = convnet_cfg['name']
        pretrained_model_file_path = convnet_cfg['pretrained_model_file_path']

        if name == 'vgg16':
            build_vgg16(
                input_images,
                minibatch_size,
                pretrained_model_file_path=pretrained_model_file_path,
            )
            convnet_features = self._tf_graph.get_tensor_by_name(
                'convnet/top/fc2/activation:0',
            )
            predictions = tf.identity(
                self._tf_graph.get_tensor_by_name(
                    'convnet/top/predictions/activation:0',
                ),
                name='predictions',
            )
        elif name[:len('inception')] == 'inception':
            endpoints = build_inception(
                name,
                input_images,
                minibatch_size,
                tf_session=self._tf_session,
                tf_graph=self._tf_graph,
                pretrained_model_file_path=pretrained_model_file_path,
                reuse=reuse,
                scope=scope,
            )
            convnet_features = tf.squeeze(
                endpoints['PreLogits'],
                axis=[1, 2],
                name='features',
            )
            predictions = tf.identity(
                endpoints['Predictions'],
                name='predictions',
            )
        else:
            raise NotImplementedError

        _, convnet_output_size = convnet_features.shape.as_list()
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
                tf.matmul(convnet_features, W),
                b,
                name='image_embeddings',
            )

        return image_embeddings

    def _get_variable_initializer(self):
        return tf.truncated_normal_initializer(
            dtype=tf.float32,
            **self._config['variable_initializer']
        )

    def _data_queue_put_thread(self):
        i = 0
        data = self._training_dataset._data
        num_data = len(data)
        random.shuffle(data)
        while not self._tf_coordinator.should_stop():
            if i >= num_data:
                i %= num_data
                random.shuffle(self._data_queue)
            self._data_queue.put(data[i])
            i += 1

    def _get_preprocessed_input(self, img_id, caption_id):
        convnet_name = self._config['convnet']['name']
        input_image_size = self._config['input_image_shape'][0]
        # TODO: Use TensorFlow for image resizing/crop.
        image_array = preprocess_image(
            convnet_name=convnet_name,
            image=self._dataset.get_image(img_id),
            size=input_image_size,
        )

        caption = self._dataset.get_captions(img_id)[caption_id]
        caption_sequence = self._vocabulary.get_preprocessed_sentence(
            caption,
        )
        caption_sequence_length = len(caption_sequence) - 1
        mask = np.ones(
            caption_sequence_length,
            dtype=np.int32,
        )
        return (image_array, caption_sequence, mask)

    def _input_queue_enqueue_thread(self):
#        input_image_size = self._config['input_image_shape'][0]
#        convnet_name = self._config['convnet']['name']
        dataset = self._dataset

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

        while not self._tf_coordinator.should_stop():
            data_to_enqueue = self._data_queue.get() 
            try:
                img_id, caption_id = data_to_enqueue
                # TODO: Use TensorFlow for image resizing/crop.
#                image_array = preprocess_image(
#                    convnet_name=convnet_name,
#                    image=dataset.get_image(img_id),
#                    size=input_image_size,
#                )
#
#                raw_caption = self._dataset.get_captions(img_id)[caption_id]
#                caption = self._vocabulary.get_preprocessed_sentence(
#                    raw_caption,
#                )
#                input_sequence_length = len(caption) - 1
#                mask_array = np.ones(
#                    input_sequence_length,
#                    dtype=np.int32,
#                )
                rv = self._get_preprocessed_input(
                    img_id, caption_id,
                )
                image_array, caption_seq, mask_array = rv
                self._tf_session.run(
                    enqueue_op,
                    feed_dict={
                        image: image_array,
                        input_seq: caption_seq[:-1],
                        target_seq: caption_seq[1:],
                        mask: mask_array,
                    }
                )
            except tf.errors.CancelledError:
                print('Input queue closed.')

    def _build_summary_ops(self):
        with tf.variable_scope('train'):
            train_summaries = [
                tf.summary.scalar(
                    name='minibatch_loss',
                    tensor=self._tf_graph.get_tensor_by_name(
                        'train/minibatch_loss:0'
                    )
                ),
                tf.summary.scalar(
                    name='learning_rate',
                    tensor=self._tf_graph.get_tensor_by_name(
                        'train/learning_rate:0'
                    )
                ),
                tf.summary.scalar(
                    name='queue_size',
                    tensor=self._tf_graph.get_tensor_by_name(
                        'input_queue/size:0'
                    ),
                )
            ]
            train_summary_op = tf.summary.merge(
                train_summaries,
                name='merged',
            )

        with tf.variable_scope('eval'):
            eval_summaries = [
                tf.summary.text(
                    name='input_image_ids',
                    tensor=self._tf_graph.get_tensor_by_name(
                        'eval/input_image_ids:0'
                    ),
                ),
                tf.summary.text(
                    name='target_sentences',
                    tensor=self._tf_graph.get_tensor_by_name(
                        'eval/target_sentences:0'
                    ),
                ),
                tf.summary.text(
                    name='output_sentences',
                    tensor=self._tf_graph.get_tensor_by_name(
                        'eval/output_sentences:0'
                    ),
                ),
                tf.summary.scalar(
                    name='total_loss',
                    tensor=self._tf_graph.get_tensor_by_name(
                        'eval/total_loss:0'
                    ),
                )
            ]
            eval_summary_op = tf.summary.merge(
                eval_summaries,
                name='merged',
            )


    def _get_decayed_learning_rate(self, step):
        minibatch_size = self._config['minibatch_size']
        num_examples_per_epoch = self._training_dataset.get_size()
        num_steps_per_epoch = (num_examples_per_epoch / minibatch_size)

        cfg_optimizer = self._config['optimizer']
        lr_i = cfg_optimizer['initial_learning_rate']
        decay_rate = cfg_optimizer['learning_rate_decay_rate']
        num_epochs_per_decay = cfg_optimizer['num_epochs_per_decay']

        decay_steps = int(num_steps_per_epoch * num_epochs_per_decay) 

        decayed_learning_rate = lr_i * (decay_rate ** (step // decay_steps))

        return decayed_learning_rate
        

    def train(
        self,
        run_name=None,
        max_num_steps=None,
        additional_num_steps=None,
    ):
        num_examples_per_epoch = self._training_dataset.get_size()
        num_training_epochs = self._config['num_training_epochs']
        minibatch_size = self._config['minibatch_size']
        num_steps_per_epoch = (num_examples_per_epoch / minibatch_size)

#        if not self._training:
#            raise RuntimeError

        if run_name is None:
            run_name = (
                '{:02}{:02}_{:02}{:02}{:02}'.format(*time.localtime()[1:6])
            )

        summary_writer = tf.summary.FileWriter(
            logdir='{}/{}'.format(LOG_DIR, run_name),
            graph=self._tf_graph,
        )

        if self._step is None:
            self._step = 0

        # XXX: Clean-up the following.
        if additional_num_steps is not None:
            max_num_steps = self._step + additional_num_steps
            num_training_epochs = max_num_steps / num_steps_per_epoch
        elif max_num_steps is None:
            if num_training_epochs is None:
                num_training_epochs = 1
            max_num_steps = num_steps_per_epoch * num_training_epochs
        else:
            num_training_epochs = max_num_steps / num_steps_per_epoch

        print(
            'Training for {} steps ({:g} epochs).'
            .format(max_num_steps, num_training_epochs)
        )

        display_step_interval = max_num_steps // 100
        save_step_interval = max_num_steps // 10

        self._tf_coordinator = tf.train.Coordinator()
        queue_threads =  [
            threading.Thread(target=self._data_queue_put_thread)
        ] + [
            threading.Thread(target=self._input_queue_enqueue_thread)
            for i in range(NUM_ENQUEUE_THREADS)
        ]
        for t in queue_threads:
            t.start()

        fetch_dict = {}
        for var_name in [
            'train/minibatch_loss',
            'convnet/image_embedding/image_embeddings',
            'convnet/predictions',
            'output_seqs',
            'summary/train/merged/merged',
        ]:
            fetch_dict[var_name] = self._tf_graph.get_tensor_by_name(
                var_name + ':0'
            )

        for i, var_name in enumerate(
            ['images', 'input_seqs', 'target_seqs', 'masks']
        ):
            fetch_dict[var_name] = self._tf_graph.get_tensor_by_name(
                'input_queue/dequeued_inputs:{}'.format(i),
            )

        for op_name in [
            'train/minimize_loss',
        ]:
            fetch_dict[op_name] = self._tf_graph.get_operation_by_name(op_name)

        try:
            while self._step < max_num_steps:
                self._step += 1

                if self._tf_coordinator.should_stop():
                    break

                learning_rate = self._get_decayed_learning_rate(self._step)

                feed_dict = {
                    self._tf_graph.get_tensor_by_name(
                        'train/learning_rate:0'
                    ): learning_rate,
                }

                rd = self._tf_session.run(
                    fetches=fetch_dict,
                    feed_dict=feed_dict,
                )

                summary_writer.add_summary(
                    summary=rd['summary/train/merged/merged'],
                    global_step=self._step,
                )

                if self._step % display_step_interval == 0:
                    print(
                        '{:g}% : minibatch_loss = {:g}'
                        .format(
                            (self._step / max_num_steps * 100),
                            rd['train/minibatch_loss'],
                        ),
                    )
                    predictions = np.reshape(
                        rd['convnet/predictions'],
                        (minibatch_size, 1000),
                    )
                    print('Image predictions')
                    for _, obj, prob in decode_predictions(predictions)[0]:
                        print('{}: {:g}'.format(obj, prob))

                    get_sentence = self._vocabulary.get_sentence_from_word_ids
                    input_len = sum(rd['masks'][0])
                    input_sentence = get_sentence(
                        rd['input_seqs'][0][1:input_len]
                    )
                    output_sentence = get_sentence(rd['output_seqs'][0])
                    print('input: {}'.format(input_sentence))
                    print('output: {}'.format(output_sentence))
                    print('\n')

                    self.evaluate_validation()
                    summary_writer.add_summary(
                        summary=rd['summary/eval/merged/merged'],
                        global_step=self._step,
                    )

                if (
                    self._step % save_step_interval == 0
                    or self._step == max_num_steps
                ):
                    save_path = self._tf_saver.save(
                        self._tf_session,
                        save_path='checkpoints/{}'.format(run_name),
                        global_step=self._step,
                    )
                    print('checkpoint saved at {}'.format(save_path))


            # End of one training step for-loop.

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

        with open('{}/{}'.format(CONFIG_DIR, run_name), 'w') as fp:
            json.dump(self._config, fp)

        summary_writer.close()

        return rd

    def get_inference(
        self,
        images_array,
        targets=None,
        masks=None,
    ):
        minibatch_size = self._config['minibatch_size']
        max_sequence_length = self._config['max_sequence_length']

        # Feed images, fetch RNN initial states.
        feed_dict = {
            self._tf_graph.get_tensor_by_name(
                'inference_input_images:0'
            ): images_array,
        }
        fetch_dict = {
            'rnn/inference_initial_states': self._tf_graph.get_tensor_by_name(
                'rnn/inference_initial_states:0'
            ),
            'convnet/predictions': (
                self._tf_graph.get_tensor_by_name(
                    'convnet/predictions:0'
                )
            ),
        }
        rd = self._tf_session.run(
            fetches=fetch_dict,
            feed_dict=feed_dict,
        )

        convnet_predictions = rd['convnet/predictions']
        prev_rnn_states = rd['rnn/inference_initial_states']

        inputs = np.array(
            [[self._vocabulary.start_word_id] for i in range(minibatch_size)]
        )
        output_seqs = np.zeros(
            shape=(minibatch_size, max_sequence_length),
        )
        total_loss = 0
        # For max_sequence_length, feed input seqs
        # and fetch word probabilities & new RNN states.
        for t in range(max_sequence_length):
            feed_dict = {
                self._tf_graph.get_tensor_by_name(
                    'inference_inputs:0'
                ): inputs,
                self._tf_graph.get_tensor_by_name(
                    'rnn/inference_prev_states:0'
                ): prev_rnn_states,
            }
            fetch_dict = {
                'rnn/inference_new_states': self._tf_graph.get_tensor_by_name(
                    'rnn/inference_new_states:0'
                ),
                'rnn/fc/inference_word_ids': self._tf_graph.get_tensor_by_name(
                    'rnn/fc/inference_predictions:1'
                ),
            }
            if targets is not None:
                feed_dict[self._tf_graph.get_tensor_by_name(
                    'eval/targets:0'
                )] = targets[:, t]
                feed_dict[self._tf_graph.get_tensor_by_name(
                    'eval/masks:0'
                )] = masks[:, t]
                fetch_dict['eval/loss'] = self._tf_graph.get_tensor_by_name(
                    'eval/loss:0'
                )
            rd = self._tf_session.run(
                fetches=fetch_dict,
                feed_dict=feed_dict,
            )
            prev_rnn_states = rd['rnn/inference_new_states']
            inputs = rd['rnn/fc/inference_word_ids']
            output_seqs[:, t] = rd['rnn/fc/inference_word_ids'][:, 0]
            if targets is not None:
                total_loss += rd['eval/loss']

        rd = {
            'convnet_predictions': convnet_predictions,
            'output_sequences': output_seqs,
            'total_loss': total_loss,
        }

        return rd

    def evaluate_validation(self):
        minibatch_size = self._config['minibatch_size']
        convnet_name = self._config['convnet']['name']
        input_image_shape = self._config['input_image_shape']
        max_sequence_length = self._config['max_sequence_length']

        data = random.choices(
            self._validation_dataset._data,
            k=minibatch_size,
        )
        img_ids = []
        target_sentences = []
        images_array = np.empty(
            shape=(minibatch_size, *input_image_shape),
            dtype=np.float32,
        )
        targets = np.empty(
            shape=(minibatch_size, max_sequence_length),
            dtype=np.int32,
        )
        masks = np.empty(
            shape=(minibatch_size, max_sequence_length),
            dtype=np.int32,
        )
        for i, img_id, caption_id in enumerate(data):
            img_ids.append(img_id)
            image_array, target_seq, mask = self._get_preprocessed_input(
                img_id, caption_id,
            )
            images_array[i] = image_array
            targets[i] = target_seq[:max_sequence_length]
            masks[i] = mask[:max_sequence_length]
            target_sentences.append(
                self._validation_dataset.get_captions(img_id)[caption_id]
            )

        rd = self.get_inference(images_array, targets, masks)
        output_sentences = [
            self._vocabulary.get_sentence_from_word_ids(seq)
            for seq in rd['output_sequences']
        ]

        fetch_dict = {
            'summary/eval/merged/merged': self._tf_graph.get_tensor_by_name(
                'summary/eval/merged/merged:0'
            )
        }
        feed_dict = {
            # TODO: Display images instead of their ids.
            self._tf_graph.get_tensor_by_name(
                'eval/input_image_ids:0'
            ) : img_ids,
            self._tf_graph.get_tensor_by_name(
                'eval/target_sentences:0'
            ) : target_sentences,
            self._tf_graph.get_tensor_by_name(
                'eval/output_sentences:0'
            ) : output_sentences,
        }
        rd = self._tf_session.run(
            fetches=fetch_dict,
            feed_dict=feed_dict,
        )
        return rd

    def generate_text(self, images_array):
        rd = self.get_inference(images_array)

        sentences = [
            self._vocabulary.get_sentence_from_word_ids(seq)
            for seq in rd['output_sequences']
        ]

        return sentences

    def decode_convnet_predictions(self, predictions):
        convnet_train_dataset = self._config['convnet']['train_dataset']
        if convnet_train_dataset == 'imagenet':
            from keras.applications.imagenet_utils import decode_predictions
        else:
            raise NotImplementedError
        
        return decode_predictions(predictions)

def get_step_from_checkpoint(save_path):
    return int(save_path.split('-')[-1])

def parse_checkpoint_save_path(save_path):
    filename = save_path.split('/')[-1]
    run_name, steps_str = filename.split('-')
    return (run_name, int(steps_str))
