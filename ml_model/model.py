import tensorflow as tf
from tensorflow import estimator
import numpy as np
import json

class Seq2Shape():

    def __init__(
        self, batch_size, time_major=False,
        average_across_batch=True, average_across_timesteps=True
    ):
        self.average_across_batch = average_across_batch
        self.average_across_timesteps = average_across_timesteps
        self.time_major = time_major
        self.batch_size = batch_size
        self.mode = mode


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
            if cell
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
        self, t_out,
        out_seq_len, labels, lr,
        train_op=None, loss=None
    ):
        if not loss:
            weights = tf.sequence_mask(
                out_seq_len,
                dtype=t_out.dtype
            )
            loss = tf.contrib.seq2seq.sequence_loss(
                t_out,
                labels,
                weights,
                average_across_batch=self.average_across_batch,
            )

        if not train_op:
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
