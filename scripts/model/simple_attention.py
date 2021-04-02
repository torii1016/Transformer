import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, lrelu, linear, conv, batch_norm, get_dim

class SimpleAttention(Layers):
    def __init__(self, fc_dim, name_scope, output_dim):
        super().__init__(name_scopes)
        self.fc_dim = fc_dim
        self.name_scope = name_scope

    def set_model(self, inputs, memory, is_training=True, reuse=False):

        with tf.variable_scope(self.name_scope, reuse = reuse):
            q = linear('q_dense_layer', inputs, self.fc_dim)
            k = linear('k_dense_layer', memory, self.fc_dim)
            v = linear('v_dense_layer', memory, self.fc_dim)

            #q *= depth**-0.5 #scaled dot-production

            logit = tf.matmul(q, k, transpose_b=True)
            attention_weight = tf.nn.softmax(logits, name="attention_weight")

            attention_output = tf.matmul(attention_weight, v)

            return linear('output_dense_layer', attention_output, self.output_dim)