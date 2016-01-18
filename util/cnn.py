# -*- coding: utf-8 -*-

import tensorflow as tf

def conv_layer(name, bottom, kernel_size, stride, output_dim):
    # input has shape [batch, in_height, in_width, in_channels]
    input_dim = bottom.get_shape().as_list()[-1]

    # weights and biases variables
    with tf.variable_scope(name):
        # filter has shape [filter_height, filter_width, in_channels, out_channels]
        weights = tf.get_variable("weights",
            [kernel_size, kernel_size, input_dim, output_dim],
            initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", output_dim,
            initializer=tf.constant_intializer(0.))

    conv = tf.nn.bias_add(tf.nn.conv2d(bottom, filter=weights,
        strides=[1, stride, stride, 1], padding='SAME'), biases)
    return conv

def conv_relu_layer(name, bottom, kernel_size, stride, output_dim):
    conv = conv_layer(name, bottom, kernel_size, stride, output_dim)
    relu = tf.nn.relu(conv)
    return relu

def pooling_layer(name, bottom, kernel_size, stride):
    pool = tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1], padding='SAME', name=name)
    return pool

def fc_layer(name, bottom, output_dim):
    # flatten bottom input
    # input has shape [batch, in_height, in_width, in_channels]
    shape = bottom.get_shape().as_list()
    input_dim = 1
    for d in shape[1:]:
        input_dim *= d
    flat_bottom = tf.reshape(bottom, [-1, input_dim])
    
    # weights and biases variables
    with tf.variable_scope(name):
        # weights has shape [input_dim, output_dim]
        weights = tf.get_variable("weights", [input_dim, output_dim],
            initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", output_dim,
            initializer=tf.constant_intializer(0.))
    
    fc = tf.nn.xw_plus_b(flat_bottom, weights, biases)
    return fc
    
def fc_relu_layer(name, bottom, output_dim):
    fc = fc_layer(name, bottom, output_dim)
    relu = tf.nn.relu(fc)
    return relu
    
def softmax_loss_layer(name, score_bottom, label_bottom):
    """
    Calculates cumulative Softmax Cross Entropy Loss along the last dimension
    *This function does not divide the loss by batch size*
    
    Once tensorflow has SparseCrossEntropy function, this one will be replaced
    """
    # Check shape
    score_shape = score_bottom.get_shape().as_list()
    label_shape = label_bottom.get_shape().as_list()
    assert len(score_shape) == len(label_shape) + 1
    assert score_shape[:-1] == label_shape
    
    # Compute the outer dimensions dimensions in label
    inner_dim = score_shape[-1]
    outer_dim = 1
    for d in label_shape: outer_dim *= d
    
    # flatten score and label
    flat_score = tf.reshape(score_bottom, [outer_dim, inner_dim])
    flat_label = tf.reshape(label_bottom, [outer_dim, 1])
    
    # Reshape the labels into a dense Tensor of
    # shape [batch_size, NUM_CLASSES].
    sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
    indices = tf.reshape(tf.range(FLAGS.batch_size), [FLAGS.batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated, [FLAGS.batch_size, NUM_CLASSES],
    1.0, 0.0)
    