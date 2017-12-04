from data_loader import DataLoader
import tensorflow as tf
import numpy as np
class ModelInputs():
    """Factory to construct various input hooks and functions depending on mode """

    def __init__(
        self, batch_size, file_path
    ):
        self.batch_size = batch_size
        self.time_delay=20
        loader = DataLoader(file_path, 'data_saves', time_delay=self.time_delay)
        self.data = loader.loadData()

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

    def _prepare_iterator_hook(self, hook, scope_name, iterator, inputs, placeholder):
        if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == tf.estimator.ModeKeys.EVAL:
            feed_dict = {
                    placeholder[0]: inputs[0],
                    placeholder[1]: inputs[1]
            }
        else:
            feed_dict = {placeholder: file_path}

        with tf.name_scope(scope_name):
            hook.iterator_initializer_func = \
                    lambda sess: sess.run(
                        iterator.initializer,
                        feed_dict=feed_dict,
                    )

    def _set_up_train_or_eval(self, scope_name):
        hook = IteratorInitializerHook()
        data = self.data
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            inputs = np.concatenate(data[0]['training'])
            outputs = np.concatenate(data[1]['training'])
        else:
            inputs = np.concatenate(data[0]['training'])
            outputs = np.concatenate(data[1]['training'])
        in_limit = inputs.shape[0]//100
        in_shape = (in_limit, 100, 28)
        out_limit = outputs.shape[0]//100
        out_shape = (out_limit, 100, 20)
        inputs = np.reshape(inputs[:in_limit*100], in_shape)
        outputs = np.reshape(outputs[:out_limit*100], out_shape)
        def input_fn():
            with tf.name_scope(scope_name):
                inp_placeholder = tf.placeholder(tf.float64, shape=in_shape)
                out_placeholder = tf.placeholder(tf.float64, shape=out_shape)
                dataset = tf.data.Dataset.from_tensor_slices((inp_placeholder, out_placeholder)).repeat(None)
                dataset = self._batch_data(dataset)
                iterator = dataset.make_initializable_iterator()
                next_example, next_label = iterator.get_next()
                self._prepare_iterator_hook(hook, 'train_inputs', iterator, (inputs, outputs), (inp_placeholder,
                    out_placeholder))
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
