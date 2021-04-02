import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, lrelu, linear, conv, batch_norm, get_dim

class MultiHeadAttention(Layers):
    def __init__(self, fc_dim, head_num, name_scope, output_dim):
        super().__init__(name_scopes)
        self.fc_dim = fc_dim
        self.head_num = head_num
        self.name_scope = name_scope
        self.drate = 0.0

    def set_model(self, inputs, memory, attention_mask, is_training=True, reuse=False):

        with tf.variable_scope(self.name_scope, reuse=reuse):
            q = linear('q_dense_layer', inputs, self.fc_dim)
            k = linear('k_dense_layer', memory, self.fc_dim)
            v = linear('v_dense_layer', memory, self.fc_dim)

            q = self._split_head(q)
            k = self._split_head(k)
            v = self._split_head(v)

            depth = self.fc_dim // self.head_num
            #q *= depth**-0.5 #scaled dot-production

            logit = tf.matmul(q, k, transpose_b=True)
            logit += tf.to_float(attention_mask)*inputs.dtype.min

            attention_weight = tf.nn.softmax(logits, name="attention_weight")
            attention_weight = tf.nn.dropout(attention_weight, self.drate)

            attention_output = tf.matmul(attention_weight, v)
            attention_output = self._combine_head(attention_output)
            return linear('output_dense_layer', attention_output, self.output_dim)
    

    def _split_head(self, x):
        with tf.name_scope('split_head'):
            batch_size, length, hidden_dim = tf.unstack(tf.shape(x))
            x = tf.reshape(x, [batch_size, length, self.head_num, self.fc_dim // self.head_num])
            return tf.transpose(x, [0, 2, 1, 3])
    
    def _combine_head(self, x):
        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.fc_dim])


class SelfAttention(MultiHeadAttention):
    def call(self, inputs, attention_mask, training):
        return super.call(inputs=inputs, memory=inputs, attention_mask=attention_mask, training=training)


class FeedForwardNetwork(Layers):
    def __init__(self, fc_dim, name_scope, drate):
        super().__init__(name_scopes)
        self.fc_dim = fc_dim
        self.name_scope = name_scope
        self.drate = 0.0

    def set_model(self, inputs, is_training=True, reuse=False):

        with tf.variable_scope(self.name_scope, reuse=reuse):
            h = linear('filter_layer', inputs, self.fc_dim*4)
            h = tf.nn.dropout(h, self.drate)
            return linear('output_layer', h, self.fc_dim)
    