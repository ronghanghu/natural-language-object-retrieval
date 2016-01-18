# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# components
import util.cnn.conv_relu_layer as conv_relu
import util.cnn.pooling_layer as pool
import util.cnn.fc_layer as fc
import util.cnn.fc_relu_layer as fc_relu
import util.rnn.lstm_layer as lstm
import tf.nn.dropout as drop

def preprocess_batch(raw_batch, rgb_channel_mean):
    """
    scale N x H x W x C input batch to range in [0, 255] and subtract mean
    ----
    raw_input_batch : numpy array of N x H x W x C in RGB channel order, range [0, 1]
    rgb_channel_mean : numpy array of 1-dim RGB channel mean
    """
    processed_batch = raw_batch * 255.0 - rgb_channel_mean
    return processed_batch
    
def vgg_fc8(input_batch, name, apply_dropout):
    with tf.variable_scope(name):
        # layer 1
        conv1_1 = conv_relu('conv1_1', input_batch,
                            kernel_size=3, stride=1, output_dim=64)
        conv1_2 = conv_relu('conv1_2', conv1_1,
                            kernel_size=3, stride=1, output_dim=64)
        pool1 = pool('pool1', conv1_2, kernel_size=2, stride=2)
    
        # layer 2
        conv2_1 = conv_relu('conv2_1', pool1,
                            kernel_size=3, stride=1, output_dim=128)
        conv2_2 = conv_relu('conv2_2', conv2_1,
                            kernel_size=3, stride=1, output_dim=128)
        pool2 = pool('pool2', conv2_2, kernel_size=2, stride=2)
    
        # layer 3
        conv3_1 = conv_relu('conv3_1', pool2,
                            kernel_size=3, stride=1, output_dim=256)
        conv3_2 = conv_relu('conv3_2', conv3_1,
                            kernel_size=3, stride=1, output_dim=256)
        conv3_3 = conv_relu('conv3_3', conv3_2,
                            kernel_size=3, stride=1, output_dim=256)
        pool3 = pool('pool3', conv3_3, kernel_size=2, stride=2)
    
        # layer 4
        conv4_1 = conv_relu('conv4_1', pool3,
                            kernel_size=3, stride=1, output_dim=512)
        conv4_2 = conv_relu('conv4_2', conv4_1,
                            kernel_size=3, stride=1, output_dim=512)
        conv4_3 = conv_relu('conv4_3', conv4_2,
                            kernel_size=3, stride=1, output_dim=512)
        pool4 = pool('pool4', conv4_3, kernel_size=2, stride=2)
    
        # layer 5
        conv5_1 = conv_relu('conv5_1', pool4,
                            kernel_size=3, stride=1, output_dim=512)
        conv5_2 = conv_relu('conv5_2', conv5_1,
                            kernel_size=3, stride=1, output_dim=512)
        conv5_3 = conv_relu('conv5_3', conv5_2,
                            kernel_size=3, stride=1, output_dim=512)
        pool5 = pool('pool5', conv5_3, kernel_size=2, stride=2)
    
        # layer 6
        fc6 = fc_relu('fc6', pool5, output_dim=4096)
        if apply_dropout: fc6 = drop(fc6, 0.5)
    
        # layer 7
        fc7 = fc_relu('fc7', fc6, output_dim=4096)
        if apply_dropout: fc7 = drop(fc7, 0.5)
    
        # layer 8
        fc8 = fc_relu('fc8', fc7, output_dim=1000)
        return fc8

def word_embedding(text_seq, num_vocab, embed_dim):
    # embedding matrix with each row containing the embedding vector of a word
    with tf.variable_scope('word_embedding'):
        embedding_mat = tf.get_variable("embedding", [num_vocab, embed_dim])
        
    # text_seq has shape [T, N] and embedded_seq has shape [T, N, D].
    embedded_seq = tf.nn.embedding_lookup(embedding_mat, text_seq)
    return embedded_seq
    
def stacked_lstm_net(bbox_feat, im_feat, spatial_feat, embedded_seq):
    # LSTM_language
    lstm_lang = lstm('lstm_lang', embedded_seq, None, output_dim=1000)
    # LSTM_local
    # TODO check order of the dimensions of the constant inputs
    lstm_local = lstm('lstm_local', lstm_lang,
                      tf.concat(1, [bbox_feat, spatial_feat]), output_dim=1000)
    # LSTM_global
    lstm_global = lstm('lstm_global', lstm_lang, im_feat, output_dim=1000)
    
    # Concatenate the outputs from LSTM_local and LSTM_global.
    # LSTM output shape is [T, N, D], so concatenate them along axis 2.
    seq_output = tf.concat(2, [lstm_local, lstm_global])
    return seq_output
    
def word_prediction(seq_output, num_vocab):
    # LSTM output shape is [T, N, D].
    num_steps, batch_size, feat_dim = seq_output.get_shape().as_list()
    flat_output = tf.reshape(seq_output, [num_steps*batch_size, feat_dim])
    flat_scores = fc('word_prediction', flat_output, output_dim=num_vocab)
    flat_probs = tf.nn.softmax(flat_scores)
    prob_seq = tf.reshape(flat_probs, [num_steps, batch_size, num_vocab])
    return prob_seq
    
def loss(text_seq, score_seq):
    # TODO make sure how the loss is averaged is SCRC
    pass