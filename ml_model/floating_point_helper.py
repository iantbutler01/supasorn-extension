import tensorflow as tf
from tensorflow.contrib.seq2seq import TrainingHelper

class FloatingPointHelper(TrainingHelper):

    def sample(self, time, outputs, name=None, **unused_kwargs):
        with tf.name_scope(name, "TrainingHelperSample", [time, outputs]):
            sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.float64)
            return sample_ids

    @property
    def sample_ids_dtype(self):
        return tf.float64
