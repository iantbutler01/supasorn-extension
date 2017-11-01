class ModelInputs():
    """Factory to construct various input hooks and functions depending on mode """

    def __init__(
        self, batch_size,
        src_eos_id=1, tgt_eos_id=2
    ):
        self.batch_size = batch_size
        self.src_eos_id = src_eos_id
        self.tgt_eos_id = tgt_eos_id

    def get_inputs(self, file_path, num_infer=None, mode=tf.estimator.ModeKeys.TRAIN):
        self.mode = mode
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            return self._training_input_hook(file_path)
        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self._validation_input_hook(file_path)
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            if num_infer is None:
                raise ValueError('If performing inference must supply number of predictions to be made.')
            return self._infer_input_hook(file_path, num_infer)

    def _batch_data(self, dataset):
        batched_set = dataset.batch(
                self.batch_size
        )
        return batched_set

    def _prepare_iterator_hook(self, hook, scope_name, iterator, file_path, name_placeholder):
        if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == tf.estimator.ModeKeys.EVAL:
            feed_dict = {
                    name_placeholder[0]: file_path[0],
                    name_placeholder[1]: file_path[1]
            }
        else:
            feed_dict = {name_placeholder: file_path}

        with tf.name_scope(scope_name):
            hook.iterator_initializer_func = \
                    lambda sess: sess.run(
                        iterator.initializer,
                        feed_dict=feed_dict,
                    )

    def _set_up_train_or_eval(self, scope_name, file_path):
        hook = IteratorInitializerHook()
        def input_fn():
            with tf.name_scope(scope_name):
                with tf.name_scope('sentence_markers'):
                    src_eos_id = tf.constant(self.src_eos_id, dtype=tf.int64)
                    tgt_eos_id = tf.constant(self.tgt_eos_id, dtype=tf.int64)
                in_file = tf.placeholder(tf.string, shape=())
                in_dataset = tf.contrib.data.TextLineDataset(in_file).repeat(None)
                out_file = tf.placeholder(tf.string, shape=())
                out_dataset = tf.contrib.data.TextLineDataset(out_file).repeat(None)
                dataset = tf.contrib.data.Dataset.zip((in_dataset, out_dataset))
                dataset = self._batch_data(dataset)
                iterator = dataset.make_initializable_iterator()
                next_example, next_label = iterator.get_next()
                self._prepare_iterator_hook(hook, scope_name, iterator, file_path, (in_file, out_file))
                return next_example, next_label

        return (input_fn, hook)

    def _training_input_hook(self, file_path):
        input_fn, hook = self._set_up_train_or_eval('train_inputs', file_path)

        return (input_fn, hook)

    def _validation_input_hook(self, file_path):
        input_fn, hook = self._set_up_train_or_eval('eval_inputs', file_path)

        return (input_fn, hook)

    def _infer_input_hook(self, file_path, num_infer):
        hook = IteratorInitializerHook()

        def input_fn():
            with tf.name_scope('infer_inputs'):
                with tf.name_scope('sentence_markers'):
                    src_eos_id = tf.constant(self.src_eos_id, dtype=tf.int64)
                infer_file = tf.placeholder(tf.string, shape=())
                dataset = tf.contrib.data.TextLineDataset(infer_file)
                dataset = self._batch_data(dataset, src_eos_id)
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
