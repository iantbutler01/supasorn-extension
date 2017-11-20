import tensorflow as tf
from tensorflow import estimator
import numpy as np
import json

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
            input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(inputs, 1)), 1)
            print(input_lengths)
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs,
                input_lengths
            )
        print(inputs)
        initial_state = network_cell.zero_state(batch_size=self.batch_size+8, dtype=tf.float32)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            network_cell,
            helper,
            initial_state=initial_state
        )
        return decoder

    def translate(
        self, num_units,
        inputs, cell=None,
    ):
        with tf.name_scope('Translate'):
            if cell:
                network_cell = cell
            else:
                network_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

            decoder = self._decoder(network_cell, inputs)
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                impute_finished=True,
                maximum_iterations=24,
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
    def prepare_predict(self, sample_id):
        return tf.estimator.EstimatorSpec(
            mode=self.mode,
            train_op=train_op,
        )

