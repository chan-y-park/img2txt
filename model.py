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

from convnet import build_vgg16, build_inception, preprocess_image
from dataset import Vocabulary


LOG_DIR = 'logs'
CHECKPOINT_DIR = 'checkpoints'
CONFIG_DIR = 'configs'
NUM_SIMILAR_WORDS = 40

inception_v3_config = {
    'name': 'inception_v3',
    'train_dataset': 'imagenet',
    'pretrained_model_file_path': 'pretrained/inception_v3.ckpt',
    'input_image_shape': [299, 299, 3],
}
inception_v4_config = {
    'name': 'inception_v4',
    'train_dataset': 'imagenet',
    'pretrained_model_file_path': 'pretrained/inception_v4.ckpt',
}
vgg16_config = {
    'name': 'vgg16',
    'train_dataset': 'imagenet',
    'pretrained_model_file_path': 'pretrained/vgg16_weights.h5',
    'input_image_shape': [224, 224, 3],
}

default_config = {
    'minibatch_size': 32,
    'embedding_size': 512,
    'max_sequence_length': 20,
    'rnn_cell': {
#        'type': 'lstm',
#        'type': 'lstm_block',
        'type': 'gru',
#        'type': 'gru_block',
        'num_units': 512,
#        'forget_bias': 1.0,
#        'use_peepholes': False,
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
    'convnet': inception_v3_config,
    'beam_size': 3,
    'num_enqueue_threads': 4,
    'data_queue_size': 200,
    'input_queue_capacity': 1000,
    'training_dataset_name': None,
    'validation_dataset_name': None,
    'vocabulary_file_path': None,
}


class Image2Text:
    def __init__(
        self,
        config_file_path=None,
        training_dataset=None,
        validation_dataset=None,
        vocabulary_file_path=None,
        gpu_memory_fraction=None,
        gpu_memory_allow_growth=True,
        save_path=None,
        inference_only=False,
        minibatch_size=None,
    ):
        if save_path is not None: 
            run_name, steps = parse_checkpoint_save_path(save_path)
            self._step = get_step_from_checkpoint(save_path)
            if config_file_path is None:
                config_file_path = os.path.join(
                    CONFIG_DIR,
                    (run_name + '.json'),
                )
        else:
            self._step = None

        if config_file_path is None: 
            self._config = default_config
        else:
            with open(config_file_path, 'r') as fp:
                self._config = json.load(fp)

        if training_dataset is not None:
            self._config['training_dataset_name'] = training_dataset.name

        if validation_dataset is not None:
            self._config['validation_dataset_name'] = validation_dataset.name

        if vocabulary_file_path is None:
            raise ValueError('vocabulary_file_path must be provided.')
        else:
            self._config['vocabulary_file_path'] = vocabulary_file_path
            self._vocabulary = Vocabulary(
                file_path=vocabulary_file_path
            )

        if minibatch_size is not None:
            self._config['minibatch_size'] = minibatch_size
        else:
            minibatch_size = self._config['minibatch_size']

        for directory in (
            LOG_DIR,
            CHECKPOINT_DIR,
            CONFIG_DIR,
        ):
            if not os.path.exists(directory):
                os.makedirs(directory)

        self._training_dataset = training_dataset
        self._validation_dataset = validation_dataset

        minibatch_size = self._config['minibatch_size']
        data_queue_size = (
            self._config['data_queue_size']
            * self._config['num_enqueue_threads']
        )
        self._data_queue = queue.Queue(maxsize=data_queue_size)

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

            if not inference_only:
                with tf.variable_scope('input_queue'):
                    self._build_input_queue()

            if inference_only:
                self._build_network(
                    with_training=False,
                    with_inference=True,
                    with_validation=False,
                    minibatch_size=minibatch_size,
                )
            else:
                self._build_network(
                    use_input_queue=True,
                    with_training=True,
                    with_inference=True,
                    with_validation=True,
                    minibatch_size=minibatch_size,
                )

            if not inference_only:
                with tf.variable_scope('summary'):
                    self._build_summary_ops()

            init_var_list = []
            for var in self._tf_graph.get_collection('variables'):
                if 'Inception' not in var.name:
                    init_var_list.append(var)

            self._tf_session.run(tf.variables_initializer(init_var_list))

            # List of variables to save and restore using tf.train.Saver.
            self._saver_var_list = self._tf_graph.get_collection(
                'trainable_variables'
            )
            self._tf_saver = tf.train.Saver(
                var_list=self._saver_var_list,
                max_to_keep=10,
            )
            if save_path is not None:
                self._tf_saver.restore(self._tf_session, save_path)

    def _build_input_queue(self):
        minibatch_size = self._config['minibatch_size']
        input_image_shape = self._config['convnet']['input_image_shape']

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

        queue_capacity = (
            self._config['input_queue_capacity']
            * self._config['num_enqueue_threads']
        )

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

    def _build_network(
        self,
        use_input_queue=True,
        with_training=True,
        with_inference=True,
        with_validation=True,
        minibatch_size=None,
    ):
#        minibatch_size = self._config['minibatch_size']
        input_image_shape = self._config['convnet']['input_image_shape']
        embedding_size = self._config['embedding_size']
        vocabulary_size = self._vocabulary.get_size() 

        if not with_training and not with_inference:
            raise ValueError(
                'Either with_training or with_inference '
                'should be True.'
            )

        # NOTE: Training runs for an unrolled RNN via tf.nn.dynamic_rnn,
        # inference runs for a single RNN cell pass.

        if with_training:
            if use_input_queue:
                # NOTE: When using PaddedFIFOQueue, all captions are padded
                # to the same maximum sequence length.
                images, input_seqs, target_seqs, masks = [
                    self._tf_graph.get_tensor_by_name(
                        'input_queue/dequeued_inputs:{}'.format(i),
                    ) for i in range(4)
                ]
            else:
                max_sequence_length = self._config['max_sequence_length']
                
                images = tf.placeholder(
                    dtype=tf.float32,
                    shape=[minibatch_size] + input_image_shape,
                    name='training_images'
                )
                input_seqs = tf.placeholder(
                    dtype=tf.int32,
                    shape=[minibatch_size, max_sequence_length],
                    name='training_input_seqs'
                )
                target_seqs = tf.placeholder(
                    dtype=tf.int32,
                    shape=[minibatch_size, max_sequence_length],
                    name='training_target_seqs'
                )
                masks = tf.placeholder(
                    dtype=tf.int32,
                    shape=[minibatch_size, max_sequence_length],
                    name='training_masks'
                )

        if with_inference:
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
            if with_training:
                convnet_features, predictions = self._build_convnet(
                    images,
                    scope=scope,
                )
                tf.identity(
                    predictions,
                    name='training_predictions',
                )
                scope.reuse_variables()
                reuse_convnet = True
            else:
                reuse_convnet = False

            if with_inference:
                (inference_convnet_features,
                 inference_predictions) = self._build_convnet(
                    inference_input_images,
                    reuse=reuse_convnet,
                    scope=scope,
                )
                tf.identity(
                    inference_predictions,
                    name='inference_predictions',
                )

        with tf.variable_scope('image_embedding') as scope:
            if with_training:
                _, convnet_output_size = convnet_features.shape.as_list()
            elif with_inference:
                _, convnet_output_size = (
                    inference_convnet_features.shape.as_list()
                )
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
            if with_training:
                image_embeddings = tf.add(
                    tf.matmul(convnet_features, W),
                    b,
                    name='image_embeddings',
                )
            if with_inference:
                inference_image_embeddings = tf.add(
                    tf.matmul(inference_convnet_features, W),
                    b,
                    name='inference_image_embeddings',
                )
            
        with tf.variable_scope('rnn'):
            cfg_rnn_cell = self._config['rnn_cell']
            rnn_output_size = cfg_rnn_cell['num_units']

            word_embedding = tf.get_variable(
                name='word_embedding',
                shape=[vocabulary_size, embedding_size],
                initializer=self._get_variable_initializer(),
            )
            if with_training:
                input_embeddings = tf.nn.embedding_lookup(
                    word_embedding,
                    input_seqs,
                )

            if with_inference:
                inference_input_embeddings = tf.nn.embedding_lookup(
                    word_embedding,
                    inference_inputs,
                )
                inference_input_embeddings = tf.squeeze(
                    inference_input_embeddings,
                    axis=1,
                )
                if 'lstm' in cfg_rnn_cell['type']:
                    inference_prev_rnn_states = tf.placeholder(
                        dtype=tf.float32,
                        shape=[minibatch_size,
                               (2 * rnn_output_size)],
                        name='inference_prev_states',
                    )
                    inference_prev_rnn_states = tf.split(
                        inference_prev_rnn_states,
                        num_or_size_splits=2,
                        axis=1,
                    )
                elif 'gru' in cfg_rnn_cell['type']:
                    inference_prev_rnn_states = tf.placeholder(
                        dtype=tf.float32,
                        shape=[minibatch_size, rnn_output_size],
                        name='inference_prev_states',
                    )
                else:
                    raise ValueError


            # TODO: Use DNC for LSTM.
            rnn_kwargs = {}
            if cfg_rnn_cell['type'] == 'lstm_block':
                tf_rnn_cell = tf.contrib.rnn.LSTMBlockCell
                rnn_kwargs['use_peephole'] = cfg_rnn_cell['use_peepholes']
                rnn_kwargs['forget_bias'] = cfg_rnn_cell['forget_bias']
            elif cfg_rnn_cell['type'] == 'lstm':
                tf_rnn_cell = tf.nn.rnn_cell.LSTMCell
                rnn_kwargs['use_peepholes'] = cfg_rnn_cell['use_peepholes']
                rnn_kwargs['forget_bias'] = cfg_rnn_cell['forget_bias']
            elif cfg_rnn_cell['type'] == 'gru':
                tf_rnn_cell = tf.contrib.rnn.GRUCell
            elif cfg_rnn_cell['type'] == 'gru_block':
                tf_rnn_cell = tf.contrib.rnn.GRUBlockCell
            else:
                raise ValueError

            with tf.variable_scope('cell') as scope:
                rnn_cell = tf_rnn_cell(cfg_rnn_cell['num_units'], **rnn_kwargs)

                if with_training:
                    # Training uses dropout in RNN.
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
                if with_inference:
                    inference_rnn_zero_states = rnn_cell.zero_state(
                        batch_size=minibatch_size,
                        dtype=tf.float32,
                    )
                    _, inference_rnn_initial_states = rnn_cell(
                        inference_image_embeddings,
                        inference_rnn_zero_states,
                    )
                    scope.reuse_variables()
                
                # XXX: Where to place the following?
#                scope.reuse_variables()

                if with_training:
                    rnn_outputs, rnn_final_state = tf.nn.dynamic_rnn(
                        cell=dropout_rnn_cell,
                        inputs=input_embeddings,
                        sequence_length=tf.reduce_sum(masks, axis=1),
                        initial_state=rnn_initial_states,
                        dtype=tf.float32,
                        scope=scope,
                    )

                if with_inference:
                    inference_rnn_outputs, inference_new_rnn_states = rnn_cell(
                        inference_input_embeddings,
                        inference_prev_rnn_states,
                    )
                # End of cell scope.

            if with_training:
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

            if with_inference:
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

                if with_training:
                    word_log_probabilities = tf.add(
                        tf.matmul(rnn_outputs, W),
                        b,
                        name='word_log_probabilities',
                    )
                    word_logits, word_ids = tf.nn.top_k(
                        word_log_probabilities,
                        k=self._config['beam_size'],
                    )

                if with_inference:
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
                        k=self._config['beam_size'],
                        name='inference_predictions',
                    )
            # End of fc scope.

        # End of rnn scope.

        if with_training:
            output_seqs = tf.reshape(
                word_ids,
                shape=[minibatch_size, -1],
                name='output_seqs'
            )

        if with_inference:
            inference_outputs = tf.reshape(
                inference_word_ids,
                shape=[minibatch_size, -1],
                name='inference_outputs'
            )

        loss_function = tf.nn.sparse_softmax_cross_entropy_with_logits

        if with_training:
            with tf.variable_scope('train'):
                lr = tf.placeholder(
                    dtype=tf.float32,
                    shape=[],
                    name='learning_rate',
                )
                targets = tf.reshape(target_seqs, [-1])
                # NOTE: The name of unmaksed_losses is
                # train/unmasked_losses/unmasked_losses:0.
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

                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=lr,
                )
                train_var_list = []
                for var in self._tf_graph.get_collection('trainable_variables'):
                    if 'convnet' not in var.name:
                        train_var_list.append(var)
                grads_and_vars = optimizer.compute_gradients(
                    var_list=train_var_list,
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

        if with_inference and with_validation:
            with tf.variable_scope('eval'):
                eval_targets = tf.placeholder(
                    dtype=tf.int32,
                    shape=[minibatch_size],
                    name='targets',
                )
                unmasked_eval_losses = loss_function(
                    labels=eval_targets,
                    logits=inference_word_log_probabilities,
                    name='unmasked_losses',
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
                eval_minibatch_loss = tf.placeholder(
                    dtype=tf.float32,
                    name='minibatch_loss',
                )
        
#        if with_inference:
#            with tf.variable_scope('embedding_similarity'):
#                # Find similar words in the embedding space
#                # in terms of cosine similarity.
#                normed_word_embedding = tf.nn.l2_normalize(
#                    word_embedding,
#                    dim=1,
#                    name='normed_word_embedding',
#                )
#                normed_image_embeddings = tf.nn.l2_normalize(
#                    inference_image_embeddings,
#                    dim=1,
#                    name='normed_image_embeddings',
#                )
#                cosine_similarities = tf.matmul(
#                    normed_image_embeddings,
#                    word_embedding,
#                    transpose_b=True,
#                    name='cosine_similarities',
#                )
#                top_cosine_similarities, top_similar_word_ids = tf.nn.top_k(
#                    cosine_similarities,
#                    k=NUM_SIMILAR_WORDS,
#                    name='top_cosine_similarities',
#                )
#                normed_top_word_vectors = tf.gather(
#                    normed_word_embedding,
#                    top_similar_word_ids,
#                    name='normed_top_word_vectors',
#                )

    def _build_convnet(self, input_images, reuse=None, scope=None):
        minibatch_size = self._config['minibatch_size']
        embedding_size = self._config['embedding_size']
        convnet_cfg = self._config['convnet']
        name = convnet_cfg['name']
        pretrained_model_file_path = convnet_cfg['pretrained_model_file_path']

        if name == 'vgg16':
            with tf.variable_scope('vgg16'):
                build_vgg16(
                    input_images,
                    minibatch_size,
                    pretrained_model_file_path=pretrained_model_file_path,
                )
                convnet_features = self._tf_graph.get_tensor_by_name(
                    'convnet/vgg16/top/fc2/activation:0',
                )
                predictions = self._tf_graph.get_tensor_by_name(
                    'convnet/vgg16/top/predictions/activation:0',
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
            predictions = endpoints['Predictions']
        else:
            raise NotImplementedError

        return convnet_features, predictions

    def _get_variable_initializer(self):
        return tf.truncated_normal_initializer(
            dtype=tf.float32,
            **self._config['variable_initializer']
        )

    def _data_queue_put_thread(self):
        # TODO: Change to logging.
        # print('Starting data queue put thread...')
        i = 0
        data = self._training_dataset._data
        num_data = len(data)
        random.shuffle(data)
        while not self._tf_coordinator.should_stop():
            if i >= num_data:
                i %= num_data
                random.shuffle(data)
            try:
                self._data_queue.put(
                    data[i],
                    block=True,
                    timeout=.1,
                )
            except queue.Full:
               continue 
            i += 1

    def _get_preprocessed_input(self, image, caption):
        convnet_name = self._config['convnet']['name']
        input_image_size = self._config['convnet']['input_image_shape'][0]
        # TODO: Use TensorFlow for image resizing/crop.
        image_array = preprocess_image(
            convnet_name=convnet_name,
            image=image,
            size=input_image_size,
        )

        caption_sequence = self._vocabulary.get_preprocessed_sentence(
            caption,
        )
        mask = np.ones(
            (len(caption_sequence) - 1),
            dtype=np.int32,
        )
        return (image_array, caption_sequence, mask)

    def _input_queue_enqueue_thread(self, thread_id):
        # TODO: Change to logging.
        # print('Starting input queue enqueue thread #{}...'.format(thread_id))
        dataset = self._training_dataset

        while not self._tf_coordinator.should_stop():
            data_to_enqueue = self._data_queue.get() 
            try:
                img_id, caption_id = data_to_enqueue
                dataset = self._training_dataset
                image = dataset.get_image(img_id)
                caption = dataset.get_captions(img_id)[caption_id]
                try:
                    image_array, caption_seq, mask_array = (
                        self._get_preprocessed_input(
                            image, caption,
                        )
                    )
                except NotImplementedError:
                    print(
                        'Input preprocessing failed:'
                        'img_id = {}, caption_id = {}'
                        .format(img_id, caption_id)
                    )
                    continue
                self._tf_session.run(
                    fetches=self._tf_graph.get_operation_by_name(
                        'input_queue/enqueue_op'
                    ),
                    feed_dict={
                        self._tf_graph.get_tensor_by_name(
                            'input_queue/image:0'
                        ): image_array,
                        self._tf_graph.get_tensor_by_name(
                            'input_queue/input_seq:0'
                        ): caption_seq[:-1],
                        self._tf_graph.get_tensor_by_name(
                            'input_queue/target_seq:0'
                        ): caption_seq[1:],
                        self._tf_graph.get_tensor_by_name(
                            'input_queue/mask:0'
                        ): mask_array,
                    }
                )
            except tf.errors.CancelledError:
                pass
                # TODO: Change to logging.
                # print('Input queue closed.')

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
#                    name='input_queue_size',
                    name='queue_size',
                    tensor=self._tf_graph.get_tensor_by_name(
                        'input_queue/size:0'
                    ),
                ),
#                tf.summary.scalar(
#                    name='data_queue_size',
#                    tensor=tf.placeholder(
#                        dtype=tf.float32,
#                        shape=[],
#                        name='data_queue_size',
#                    )
#                )
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
                    name='minibatch_loss',
                    tensor=self._tf_graph.get_tensor_by_name(
                        'eval/minibatch_loss:0'
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
        minibatch_size = self._config['minibatch_size']
        num_steps_per_epoch = (num_examples_per_epoch / minibatch_size)

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

        if additional_num_steps is not None:
            max_num_steps = self._step + additional_num_steps
        else:
            if max_num_steps is None:
                raise ValueError(
                    'Either max_num_steps or additional_num_steps '
                    'should be provided.'
                )
            additional_num_steps = max_num_steps

        display_step_interval = additional_num_steps // 100
        save_step_interval = additional_num_steps // 10
        num_training_epochs = max_num_steps / num_steps_per_epoch
        print(
            'Training for {} steps, '
            'total training {} steps (= {:g} epochs).'
            .format(additional_num_steps, max_num_steps, num_training_epochs)
        )


        self._tf_coordinator = tf.train.Coordinator()
        queue_threads =  [
            threading.Thread(target=self._data_queue_put_thread)
        ] + [
            threading.Thread(
                target=self._input_queue_enqueue_thread,
                kwargs={'thread_id': i},
            )
            for i in range(self._config['num_enqueue_threads'])
        ]
        for t in queue_threads:
            t.start()

        fetch_dict = {}
        for var_name in [
            'output_seqs',
            'train/unmasked_losses/unmasked_losses',
            'train/minibatch_loss',
            'image_embedding/image_embeddings',
            'convnet/training_predictions',
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
            step = 0
            while self._step <= max_num_steps:
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

                if step % display_step_interval == 0:
                    print(
                        '{:g}% : minibatch_loss = {:g}'
                        .format(
                            (step / additional_num_steps * 100),
                            rd['train/minibatch_loss'],
                        ),
                    )
                    predictions = np.reshape(
                        rd['convnet/training_predictions'][:,-1000:],
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

                    merged_eval_summary = self.evaluate_validation()
                    summary_writer.add_summary(
                        summary=merged_eval_summary,
                        global_step=self._step,
                    )
                if (
                    step % save_step_interval == 0
                    or self._step == max_num_steps
                ):
                    save_path = self._tf_saver.save(
                        self._tf_session,
                        save_path='checkpoints/{}'.format(run_name),
                        global_step=self._step,
                    )
                    print('checkpoint saved at {}'.format(save_path))

                step += 1
                self._step += 1

            # End of one training while-loop.

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

        with open(os.path.join(CONFIG_DIR, run_name + '.json'), 'w') as fp:
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
            'rnn/inference_initial_states':
                self._tf_graph.get_tensor_by_name(
                    'rnn/inference_initial_states:0'
                ),
            'convnet/inference_predictions':
                self._tf_graph.get_tensor_by_name(
                    'convnet/inference_predictions:0'
                ),
            # TODO: Check the latency of fetching the following tensor.
            'rnn/word_embedding':
                self._tf_graph.get_tensor_by_name(
                    'rnn/word_embedding:0'
                ),
            'image_embedding/inference_image_embeddings':
                self._tf_graph.get_tensor_by_name(
                    'image_embedding/inference_image_embeddings:0'
                ),
#            'embedding_similarity/normed_top_word_ids':
#                self._tf_graph.get_tensor_by_name(
#                    'embedding_similarity/top_cosine_similarities:1'
#                ),
#            'embedding_similarity/normed_top_word_vectors':
#                self._tf_graph.get_tensor_by_name(
#                    'embedding_similarity/normed_top_word_vectors:0'
#                ),
        }
        rd_init = self._tf_session.run(
            fetches=fetch_dict,
            feed_dict=feed_dict,
        )

#        convnet_predictions = rd['convnet/inference_predictions']
        prev_rnn_states = rd_init['rnn/inference_initial_states']

        inputs = np.array(
            [[self._vocabulary.start_word_id] for i in range(minibatch_size)]
        )
        output_seqs = np.zeros(
            shape=(minibatch_size, max_sequence_length),
            dtype=np.int32,
        )
        minibatch_loss = 0
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
                'rnn/fc/inference_word_logits': (
                    self._tf_graph.get_tensor_by_name(
                        'rnn/fc/inference_predictions:0'
                    )
                ),
                'rnn/fc/inference_word_ids': self._tf_graph.get_tensor_by_name(
                    'rnn/fc/inference_predictions:1'
                ),
            }
            if targets is not None:
                feed_dict[self._tf_graph.get_tensor_by_name(
                    'eval/targets:0'
                )] = targets[:, t]
                fetch_dict[
                    'eval/unmasked_losses'
                ] = self._tf_graph.get_tensor_by_name(
                    'eval/unmasked_losses/unmasked_losses:0'
                )
            rd = self._tf_session.run(
                fetches=fetch_dict,
                feed_dict=feed_dict,
            )
            prev_rnn_states = rd['rnn/inference_new_states']
            # TODO: Need to change when using beam search.
            inputs = rd['rnn/fc/inference_word_ids'][:, 0:1]
            output_seqs[:, t] = rd['rnn/fc/inference_word_ids'][:, 0]
            if targets is not None:
                mask_t = masks[:, t]
                unmasked_losses = rd['eval/unmasked_losses']
                len_mask = np.sum(mask_t)
                if len_mask > 0:
                    loss = (
                        np.sum(unmasked_losses * mask_t, dtype=np.float32)
                        / len_mask
                    )
                    minibatch_loss += loss
        rd = {
#            'convnet_predictions': convnet_predictions,
            'output_sequences': output_seqs,
            'minibatch_loss': minibatch_loss,
        }
        for var_name in [
            'rnn/word_embedding',
            'image_embedding/inference_image_embeddings',
#            'embedding_similarity/normed_top_word_ids',
#            'embedding_similarity/normed_top_word_vectors',
            'convnet/inference_predictions',
        ]:
            rd[var_name] = rd_init[var_name]
        return rd

    def evaluate_validation(self):
        minibatch_size = self._config['minibatch_size']
        convnet_name = self._config['convnet']['name']
        input_image_shape = self._config['convnet']['input_image_shape']
        max_sequence_length = self._config['max_sequence_length']

        img_ids = []
        target_sentences = []
        images_array = np.empty(
            shape=(minibatch_size, *input_image_shape),
            dtype=np.float32,
        )
        targets = np.zeros(
            shape=(minibatch_size, max_sequence_length),
            dtype=np.int32,
        )
        masks = np.zeros(
            shape=(minibatch_size, max_sequence_length),
            dtype=np.int32,
        )
        dataset = self._validation_dataset
        len_data = len(dataset._data)
        for i in range(minibatch_size):
            finished = False
            while not finished:
                k = random.randrange(0, len_data)
                img_id, caption_id = dataset._data[k]
                image = dataset.get_image(img_id)
                caption = dataset.get_captions(img_id)[caption_id]
                try:
                    image_array, caption_seq, mask = (
                        self._get_preprocessed_input(
                            image, caption,
                        )
                    )
                except NotImplementedError:
                    continue
                if (len(caption_seq) <= max_sequence_length):
                    finished = True
            images_array[i] = image_array
            target_seq = caption_seq[1:]
            targets[i,:len(target_seq)] = target_seq
            masks[i,:len(mask)] = mask

            target_sentences.append(
                '{}: {}'.format(
                    i,
                    self._validation_dataset.get_captions(img_id)[caption_id],
                )
            )
            img_ids.append(img_id)
        assert(len(img_ids) == minibatch_size)
        assert(len(target_sentences) == minibatch_size)

        rd = self.get_inference(images_array, targets, masks)
        output_sentences = [
            '{}: {}'.format(
                i,
                self._vocabulary.get_sentence_from_word_ids(
                    self._vocabulary.get_postprocessed_sequence(seq)
                )
            )
            for i, seq in enumerate(rd['output_sequences'])
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
            ) : ['{}: {}'.format(i, img_id)
                 for i, img_id in enumerate(img_ids)],
            self._tf_graph.get_tensor_by_name(
                'eval/target_sentences:0'
            ) : target_sentences,
            self._tf_graph.get_tensor_by_name(
                'eval/output_sentences:0'
            ) : output_sentences,
            self._tf_graph.get_tensor_by_name(
                'eval/minibatch_loss:0'
            ) : rd['minibatch_loss'],
        }
        rd = self._tf_session.run(
            fetches=fetch_dict,
            feed_dict=feed_dict,
        )
        return rd['summary/eval/merged/merged']

    def generate_sentence(self, input_image):
        cfg_convnet = self._config['convnet']
        preprocessed_image = preprocess_image(
            convnet_name=cfg_convnet['name'],
            image=input_image,
            size=cfg_convnet['input_image_shape'][0],
        )
        rd = self.get_inference(preprocessed_image[np.newaxis,:])

        sentence = self._vocabulary.get_sentence_from_word_ids(
            self._vocabulary.get_postprocessed_sequence(
                rd['output_sequences'][0]
            )
        )
        return sentence

    def save_coco_eval_cap_result(
        self,
        file_path,
        dataset=None,    
        num_samples=4000,
    ):
        if dataset is None:
            if self._validation_dataset is None:
                raise RuntimeError('No validation dataset available.')
            else:
                dataset = self._validation_dataset

        result = []
        image_ids = random.sample(
            dataset.get_image_ids(),
            k=num_samples,
        )
        # XXX
        for image_id in image_ids[:1000]:
            coco_image_id = int(image_id.split('_')[-1].split('.')[0])
            input_image = dataset.get_image(image_id) 
            generated_sentence = self.generate_sentence(input_image)
            result.append(
                {'image_id': coco_image_id,
                 'caption': generated_sentence.rstrip('.')}
            )
        with open(file_path, 'w') as fp:
            json.dump(result, fp)

    def decode_convnet_predictions(self, predictions):
        convnet_train_dataset = self._config['convnet']['train_dataset']
        if convnet_train_dataset == 'imagenet':
            from keras.applications.imagenet_utils import decode_predictions
        else:
            raise NotImplementedError
        
        return decode_predictions(predictions)

    def test_train(
        self,
        dataset=None,
        max_num_steps=None,
        verbose=False,
    ):
        # TODO: Turn this into a full test.
        from PIL import Image
        import h5py

        minibatch_size = self._config['minibatch_size']
        input_image_shape = self._config['convnet']['input_image_shape']
        max_sequence_length = self._config['max_sequence_length']
        convnet_name = self._config['convnet']['name']

        # Preprare test inputs.
        images_array = np.empty(
            shape=(minibatch_size, *input_image_shape),
            dtype=np.float32,
        )
        input_seqs = np.zeros(
            shape=(minibatch_size, max_sequence_length),
            dtype=np.int32,
        )
        target_seqs = np.zeros(
            shape=(minibatch_size, max_sequence_length),
            dtype=np.int32,
        )
        masks = np.zeros(
            shape=(minibatch_size, max_sequence_length),
            dtype=np.int32,
        )

        image = Image.open('images/dog.jpg')
        input_sentence = 'a happy dog.'
        image_array, caption_seq, mask = self._get_preprocessed_input(
            image, input_sentence,
        )
        len_seq = len(caption_seq)
        for i in range(minibatch_size):
            images_array[i] = image_array
            input_seqs[i,:len_seq - 1] = caption_seq[:-1]
            target_seqs[i, :len_seq - 1]  = caption_seq[1:]
            masks[i, :len_seq - 1] = [1] * (len_seq - 1)

        fetch_dict = {}
        for var_name in [
            'output_seqs',
            'train/unmasked_losses/unmasked_losses',
            'train/minibatch_loss',
            'image_embedding/image_embeddings',
            'convnet/training_predictions',
            'output_seqs',
        ]:
            fetch_dict[var_name] = self._tf_graph.get_tensor_by_name(
                var_name + ':0'
            )

        for op_name in [
            'train/minimize_loss',
        ]:
            fetch_dict[op_name] = self._tf_graph.get_operation_by_name(op_name)

        step = 0
        while step < max_num_steps:
            print(step)
            rv = self._tf_session.run(
                fetches=[self._tf_graph.get_tensor_by_name(
                        'convnet/inference_predictions:0'
                    )
                ],
                feed_dict={
                    self._tf_graph.get_tensor_by_name(
                        'inference_input_images:0'
                    ): images_array
                },
            )
            flattened_prediction_values = rv[0]
            prediction_values = np.reshape(
                    flattened_prediction_values[:,-1000:],
                    (minibatch_size, 1000),
                )
            decoded_predictions = decode_predictions(
                prediction_values
            )
            _, obj_name, prob = decoded_predictions[0][0]
            assert(obj_name == 'golden_retriever')
            
            if verbose:
                for _, obj, prob in decoded_predictions[0]:
                    print('{}: {:g}'.format(obj, prob))

            if convnet_name == 'vgg16':
                block_layer_name = 'block1_conv1'
                var_name = 'W'
                dset_name = block_layer_name + '_' + var_name + '_1:0'
                conv_var = self._tf_graph.get_tensor_by_name(
                    'convnet/vgg16/block1/conv1/W:0'
                )
                weights_f = h5py.File(
                    'pretrained/vgg16_weights.h5',
                    mode='r',
                )
                pretrained_var_val = (
                    weights_f
                    [block_layer_name]
                    [dset_name]
                    .value
                )
                assert(conv_var.eval(session=self._tf_session)
                       == pretrained_var_val)

            learning_rate = 2.0

            feed_dict = {
                self._tf_graph.get_tensor_by_name(
                    'train/learning_rate:0'
                ): learning_rate,
                self._tf_graph.get_tensor_by_name(
                    'training_images:0'
                ): images_array,
                self._tf_graph.get_tensor_by_name(
                    'training_input_seqs:0'
                ): input_seqs,
                self._tf_graph.get_tensor_by_name(
                    'training_target_seqs:0'
                ): target_seqs,
                self._tf_graph.get_tensor_by_name(
                    'training_masks:0'
                ): masks,
            }

            rd = self._tf_session.run(
                fetches=fetch_dict,
                feed_dict=feed_dict,
            )
            predictions = rd['convnet/training_predictions'][:,-1000:]
            decoded_predictions = decode_predictions(predictions)
            _, obj_name, prob = decoded_predictions[0][0]
            assert(obj_name == 'golden_retriever')
            if verbose:
                for _, obj, prob in decoded_predictions[0]:
                    print('{}: {:g}'.format(obj, prob))

            get_sentence = self._vocabulary.get_sentence_from_word_ids
            output_sentence = get_sentence(rd['output_seqs'][0])
            print('input: {}'.format(input_sentence))
            print('output: {}'.format(output_sentence))
            print('\n')

            inference_rd = self.get_inference(
                images_array,
                target_seqs,
                masks,
            )

            inference_convnet_predictions = inference_rd[
                'convnet/inference_predictions'
            ][:,-1000:]
            decoded_predictions = decode_predictions(
                inference_convnet_predictions
            )
            _, obj_name, prob = decoded_predictions[0][0]
            assert(obj_name == 'golden_retriever')
            if verbose:
                for _, obj, prob in decoded_predictions[0]:
                    print('{}: {:g}'.format(obj, prob))

            output_sentence = self._vocabulary.get_sentence_from_word_ids(
                self._vocabulary.get_postprocessed_sequence(
                    inference_rd['output_sequences'][0]
                )
            )
            print('inference minibatch loss = {}'
                  .format(inference_rd['minibatch_loss']))
            
            step += 1

        return rd

def get_step_from_checkpoint(save_path):
    return int(save_path.split('-')[-1])

def parse_checkpoint_save_path(save_path):
    filename = save_path.split('/')[-1]
    run_name, steps_str = filename.split('-')
    return (run_name, int(steps_str))
