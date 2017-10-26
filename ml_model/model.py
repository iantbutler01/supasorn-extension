import tensorflow as tf
from tensorflow import estimator
import numpy as np
import json

class Seq2Shape():

    def __init__(
        self, batch_size, dim_out, time_major=False,
    ):
        self.time_major = time_major
        self.batch_size = batch_size
        self.dim_out = dim_out


    def _get_lstm(self, num_units):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units)

    def _decoder(self, network_cell, out_seq_len, inputs, helper):
        if not helper:
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs,
                out_seq_len,
            )
        initial_state = self.decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            network_cell,
            helper,
        )
        return decoder

    def translate(
        self, num_units, out_seq_len,
        inputs, cell=None,
    ):
        with tf.name_scope('Translate'):
            if cell:
                network_cell = cell
            else:
                network_cell = tf.nn.rnn_cell.BasicLSTMCell(2*num_units)

            decoder = self._decoder(network_cell, out_seq_len, inputs)
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                maximum_iterations=20,
                swap_memory=True,
            )
            outputs = outputs[0]
            if self.mode != estimator.ModeKeys.PREDICT:
                return outputs.rnn_output, outputs.sample_id
            else:
                return outputs.rnn_output, outputs.sample_id

    def prepare_train_eval(
        self, t_out, num_units,
        out_seq_len, labels, lr,
    ):
        with tf.variable_scope('Weights_Intersect'):
          output_w = tf.get_variable("output_w", [num_units, self.dimout])
          output_b = tf.get_variable("output_b", [self.dimout])
        t_out = tf.reshape(tf.concat(1, t_out), [-1, num_units])
        t_out = tf.nn.xw_plus_b(t_out, output_w, output_b)
        loss = tf.squared_difference(
            t_out,
            labels,
        )
        loss = loss / (self.batch_size*out_seq_len*self.dim_out)

        train_op = tf.contrib.layers.optimize_loss(
            loss,
            tf.train.get_global_step(),
            optimizer='SGD',
            learning_rate=lr,
            summaries=['loss', 'learning_rate']
        )

        return tf.estimator.EstimatorSpec(
            mode=self.mode,
            loss=loss,
            train_op=train_op,
        )
