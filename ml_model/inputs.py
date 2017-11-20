from data_loader import DataLoader
import tensorflow as tf
import numpy as np
class ModelInputs():
    """Factory to construct various input hooks and functions depending on mode """

    def __init__(
        self, batch_size, file_path
    ):
        self.batch_size = batch_size
        loader = DataLoader(file_path)
        self.data = loader.preprocess('./data_saves')

    def get_inputs(self, mode=tf.estimator.ModeKeys.TRAIN):
        self.mode = mode
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            return self._training_input_hook()
        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self._validation_input_hook()
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return self._infer_input_hook(file_path)

    def _batch_data(self, dataset):
        batched_set = dataset.batch(
                self.batch_size
        )
        return batched_set

    def _set_up_train_or_eval(self, scope_name):
        hook = IteratorInitializerHook()
        data = self.data
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            inputs = np.concatenate(data[0]['training'])
            outputs = np.concatenate(data[1]['training'])
        else:
            inputs = np.concatenate(data[0]['training'])
            outputs = np.concatenate(data[1]['training'])
        def input_fn():
            with tf.name_scope(scope_name):
                dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs)).repeat(None)
                iterator = dataset.make_initializable_iterator()
                next_example, next_label = iterator.get_next()
                return next_example, next_label

        return (input_fn, hook)

    def _training_input_hook(self):
        input_fn, hook = self._set_up_train_or_eval('train_inputs')

        return (input_fn, hook)

    def _validation_input_hook(self):
        input_fn, hook = self._set_up_train_or_eval('eval_inputs')

        return (input_fn, hook)

    def _infer_input_hook(self, file_path):
        hook = IteratorInitializerHook()

        def input_fn():
            with tf.name_scope('infer_inputs'):
                infer_file = tf.placeholder(tf.string, shape=())
                dataset = tf.contrib.data.TextLineDataset(infer_file)
                dataset = self._batch_data(dataset)
                iterator = dataset.make_initializable_iterator()
                next_example, seq_len = iterator.get_next()
                self._prepare_iterator_hook(hook, 'infer_inputs', iterator, file_path, infer_file)
                return ((next_example, seq_len), None)

        return (input_fn, hook)


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)
