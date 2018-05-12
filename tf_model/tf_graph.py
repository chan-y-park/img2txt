import tensorflow as tf

def build_network(
    model,
    use_input_queue=True,
    with_training=True,
    with_inference=True,
    with_validation=True,
    train_convnet=False,
    minibatch_size=None,
):
    input_image_shape = model._config['convnet']['input_image_shape']
    embedding_size = model._config['embedding_size']
    vocabulary_size = model._vocabulary.get_size() 

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
                model._tf_graph.get_tensor_by_name(
                    'input_queue/dequeued_inputs:{}'.format(i),
                ) for i in range(4)
            ]
        else:
            max_sequence_length = model._config['max_sequence_length']
            
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
        # The shape of inference_inputs is 2-dim
        # to make it the same as training inputs.
        inference_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=[minibatch_size, 1],
            name='inference_inputs',
        )

    with tf.variable_scope('convnet') as scope:
        if with_training:
            convnet_features, predictions = model._build_convnet(
                images,
                scope=scope,
                train_convnet=train_convnet,
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
             inference_predictions) = model._build_convnet(
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
            initializer=model._get_variable_initializer(),
        )
        b = tf.get_variable(
            name='b',
            shape=(embedding_size),
            initializer=model._get_variable_initializer(),
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
        cfg_rnn_cell = model._config['rnn_cell']
        rnn_output_size = cfg_rnn_cell['num_units']

        word_embedding = tf.get_variable(
            name='word_embedding',
            shape=[vocabulary_size, embedding_size],
            initializer=model._get_variable_initializer(),
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
                initializer=model._get_variable_initializer(),
            )
            b = tf.get_variable(
                name='b',
                shape=(vocabulary_size),
                initializer=model._get_variable_initializer(),
            )

            if with_training:
                word_log_probabilities = tf.add(
                    tf.matmul(rnn_outputs, W),
                    b,
                    name='word_log_probabilities',
                )
                word_logits, word_ids = tf.nn.top_k(
                    word_log_probabilities,
                    k=1,
                )

            if with_inference:
                inference_word_log_probabilities = tf.add(
                    tf.matmul(inference_rnn_outputs, W),
                    b,
                    name='inference_word_log_probabilities',
                )
                inference_word_probabilities = tf.nn.softmax(
                    logits=inference_word_log_probabilities,
                    name='inference_word_probabilities',
                )
                inference_word_logits, inference_word_ids = tf.nn.top_k(
                    inference_word_log_probabilities,
                    k=model._config['beam_size'],
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
            for var in model._tf_graph.get_collection('trainable_variables'):
                if (
                    train_convnet
                    or ('convnet' not in var.name)
                ):
                    train_var_list.append(var)
            grads_and_vars = optimizer.compute_gradients(
                var_list=train_var_list,
                loss=minibatch_loss,
            )
            gradients, variables = zip(*grads_and_vars)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients,
                model._config['optimizer']['gradient_clip_norm'],
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


