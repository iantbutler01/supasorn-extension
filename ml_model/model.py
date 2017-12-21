import tensorflow as tf
from tensorflow import estimator
import numpy as np
import json
from floating_point_helper import FloatingPointHelper

class Seq2Shape():

    def __init__(
        self, batch_size, dim_out, mode, time_major=False,
    ):
        self.time_major = time_major
        self.batch_size = batch_size
        self.dim_out = dim_out
        self.mode = mode


    def _get_lstm(self, num_units):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units)


    def _decoder(self, network_cell, inputs, helper=None):
        if not helper:
            helper = FloatingPointHelper(
                inputs,
                [100]*self.batch_size
            )
        initial_state = network_cell.zero_state(batch_size=self.batch_size, dtype=tf.float64)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            network_cell,
            helper,
            initial_state=initial_state
        )
        return decoder

    def _attention_cell(self, num_units, memory, memory_sequence_length, network_cell):
        mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, memory, memory_sequence_length)
        return tf.contrib.seq2seq.AttentionWrapper(
            network_cell,
            mechanism,
            attention_layer_size=num_units,
            alignment_history=False,
            name='AttentionWrapper'
        )

    def translate(
        self, num_units,
        inputs, cell=None, num_layers=1
    ):
        with tf.name_scope('Translate'):
            if cell:
                network_cell = cell
            else:
                network_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units) for x in
                    range(num_layers)])
            network_cell = tf.contrib.rnn.DropoutWrapper(
                network_cell,
                input_keep_prob=0.7,
                output_keep_prob=0.7,
            )

            inputs = tf.split(inputs, 100, axis=1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            outputs,_ = tf.contrib.legacy_seq2seq.rnn_decoder(
                inputs,
                network_cell.zero_state(batch_size=self.batch_size, dtype=tf.float64),
                network_cell,
                scope='rnndec'
                
            )
            if self.mode != estimator.ModeKeys.PREDICT:
                return outputs
            else:
                return outputs

    def prepare_train_eval(
        self, t_out, num_units,
        out_seq_len, labels, lr,
    ):
        with tf.variable_scope('Weights_Intersect'):
          output_w = tf.get_variable("output_w", [num_units, self.dim_out], dtype=tf.float64)
          output_b = tf.get_variable("output_b", [20], dtype=tf.float64)
        t_out = tf.reshape(tf.concat(t_out, 1), [-1, num_units])
        t_out = tf.nn.xw_plus_b(t_out, output_w, output_b)
        labels = tf.reshape(labels,[-1, self.dim_out])
        loss = tf.reduce_sum(tf.squared_difference(
            labels,
            t_out
        ))
        loss = loss / (self.batch_size*out_seq_len*self.dim_out)
        train_op = tf.contrib.layers.optimize_loss(
            loss,
            tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=lr,
            summaries=['loss', 'learning_rate'],
            clip_gradients=10.0
        )

        return tf.estimator.EstimatorSpec(
            mode=self.mode,
            loss=loss,
            train_op=train_op,
        )
    def prepare_predict(self, sample_id):
        return tf.estimator.EstimatorSpec(
            mode=self.mode,
            predictions=sample_id
        )

