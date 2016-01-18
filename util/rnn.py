# -*- coding: utf-8 -*-

import tensorflow as tf

def lstm_layer(name, seq_bottom, const_bottom, output_dim, num_layers=1,
               forget_bias=0.0, apply_dropout=False, keep_prob=0.5):
    """
    Similar LSTM layer as the `LSTMLayer` in Caffe
    ----
    Args:
        seq_bottom : the underlying sequence input of size [T, N, D_in], where
            D_in is the input dimension, T is num_steps and N is batch_size.
        const_bottom : the constant bottom concatenated to each time step,
            having shape [N, D_const]. This can be *None*. If it is None, 
            then this input is ignored.
        output_dim : the number of hidden units in the LSTM unit and also the
            final output dimension, i.e. D_out.
        num_layers : the number of stacked LSTM layers.
        forget_bias : forget gate bias in LSTM unit.
        apply_dropout, keep_prob: dropout applied to the output of each LSTM
            unit.
    Returns:
        output : a list of [T, N, D_out], where D_out is output_dim, 
            T is num_steps and N is batch_size
    """
    
    # input shape is [T, N, D_in]
    input_shape = seq_bottom.get_shape().as_list()
    # the number of time steps to unroll
    num_steps = input_shape[0]
    # batch size (i.e. N)
    batch_size = input_shape[1]
    
    # The actual parameter variable names are as follows (`name` is the name
    # variable here, and Cell0, Cell1, ... are num_layers stacked LSTM cells):
    #   `name`/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix
    #   `name`/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias
    #   `name`/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix
    #   `name`/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias
    # where Cell1 is on top of Cell0, taking Cell0's hidden states as inputs.
    # 
    # For Cell0, the weight matrix ('BasicLSTMCell/Linear/Matrix') has shape
    # [D_in+D_const+D_out, 4*D_out], and bias has shape [4*D_out].
    # For Cell1, Cell2, ..., the weight matrix ('BasicLSTMCell/Linear/Matrix')
    # has shape [D_out*2, 4*D_out], and bias has shape [4*D_out].
    # In the weight matrix, the first D_in+D_const rows (in Cell0) or D_out rows
    # (in Cell1, Cell2, ...) are bottom input weights, and the rest D_out rows
    # are state weights, i.e. *inputs are before states in weight matrix*
    # 
    # The gate order in 4*D_out are i, j, f, o, where
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    #
    # Other details in tensorflow/python/ops/rnn_cell.py
    with tf.variable_scope(name):
        # the basic LSTM cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(output_dim, forget_bias)
        # Apply dropout if specified.
        if apply_dropout and keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
        
        # Initialize cell state from zero.
        initial_state = cell.zero_state(batch_size, tf.float32)
        # Split along time dimension and flatten each component.
        # `inputs` is a list.
        inputs = [tf.reshape(input_, [batch_size, -1])
            for input_ in tf.split(0, num_steps, seq_bottom)]
        # Add constant input to each time step.
        if not const_bottom is None:
            # Flatten const_bottom into shape [N, D_const] and concatenate.
            const_input_ = tf.reshape(const_bottom, [batch_size, -1])
            inputs = [tf.concat(0, [input_, const_input_])
                for input_ in inputs]
        
        # Create the Recurrent Network and collect `outputs`. `states` are
        # ignored.
        outputs, _ = tf.nn.rnn(cell, inputs, initial_state=initial_state)
        # Concat the outputs into [T, N, D_out].
        output = tf.reshape(tf.concat(0, outputs),
                            [num_steps, batch_size, output_dim])
    return output