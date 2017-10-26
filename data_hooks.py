import tensorflow as tf
from tensorflow.python.ops import lookup_ops

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

class ModelInputs(object):
    """Factory to construct various input hooks and functions depending on mode """

    def __init__(
        self, vocab_files, batch_size,
        share_vocab=True, src_eos_id=1, tgt_eos_id=2
    ):
        self.batch_size = batch_size
        with tf.name_scope('sentence_markers'):
            self.src_eos_id = tf.constant(src_eos_id, dtype=tf.int64)
            self.tgt_eos_id = tf.constant(tgt_eos_id, dtype=tf.int64)
        self.vocab_tables = self._create_vocab_tables(vocab_files, share_vocab)

    def get_inputs(self, file_path, num_infer=None, mode=tf.estimator.ModeKeys.TRAIN):
        self.mode = mode
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            return self._training_input_hook(file_path)
        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self._validation_input_hook(file_path)
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            if num_infer is None:
                raise ArgumentError('If performing inference must supply number of predictions to be made.')
            return self._infer_input_hook(file_path, num_infer)

    def _prepare_data(self, dataset, out=False):
        prep_set = dataset.map(lambda string: tf.string_split([string]).values)
        prep_set = prep_set.map(lambda words: (words, tf.size(words)))
        if out == True:
            return prep_set.map(lambda words, size: (self.vocab_tables[1].lookup(words), size))
        return prep_set.map(lambda words, size: (self.vocab_tables[0].lookup(words), size))

    def _batch_data(self, dataset):
        batched_set = dataset.padded_batch(
                self.batch_size,
                padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])), (tf.TensorShape([None]), tf.TensorShape([]))),
                padding_values=((self.src_eos_id, 0), (self.tgt_eos_id, 0))
        )
        return batched_set

    def _create_vocab_tables(self, vocab_files, share_vocab=False):
        if vocab_files[1] is None and share_vocab == False:
            raise ArgumentError('If share_vocab is set to false must provide target vocab. (src_vocab_file, \
                    target_vocab_file)')

        src_vocab_table = lookup_ops.index_table_from_file(
            vocab_files[0],
            default_value=UNK_ID
        )

        if share_vocab:
            tgt_vocab_table = src_vocab_table
        else:
            tgt_vocab_table = lookup_ops.index_table_from_file(
                vocab_files[1],
                default_value=UNK_ID
            )

        return src_vocab_table, tgt_vocab_table

    def _create_iterator_hook(self, scope_name, iterator, file_path, name_placeholder):
        hook = IteratorInitializerHook()
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
                        feed_dict=feed_dict
                    )
        return hook

    def _input_fn(self, iterator):
        def get_inputs():
            return iterator.get_next()
        return get_inputs

    def _set_up_train_or_eval(self, scope_name, file_path):
        with tf.name_scope(scope_name):
            in_file = tf.placeholder(tf.string, shape=())
            in_dataset = self._prepare_data(tf.contrib.data.TextLineDataset(in_file))
            out_file = tf.placeholder(tf.string, shape=())
            out_dataset = self._prepare_data(tf.contrib.data.TextLineDataset(out_file))
            dataset = tf.contrib.data.Dataset.zip((in_dataset, out_dataset))
            dataset = self._batch_data(dataset)
            iterator = dataset.make_initializable_iterator()

        hook = self._create_iterator_hook(scope_name, iterator, file_path, (in_file, out_file))

        return (iterator, hook)

    def _training_input_hook(self, file_path):
        iterator, hook = self._set_up_train_or_eval('train_inputs', file_path)

        return (self._input_fn(iterator), hook)

    def _validation_input_hook(self, file_path):
        iterator, hook = self._set_up_train_or_eval('eval_inputs', file_path)

        return (self._input_fn(iterator), hook)

    def _infer_input_hook(self, file_path, num_infer):
        with tf.name_scope('infer_inputs'):
            infer_file = tf.placeholder(tf.string, shape=(num_infer))
            dataset = tf.contrib.data.TextLineDataset(infer_file)
            dataset = self._prepare_data(dataset)
            dataset = self._batch_data(dataset)
            iterator = dataset.make_initalizable_iterator()

        hook = self._create_iterator_hook('infer_inputs', iterator, file_path, infer_file)

        return (self._input_fn(iterator), hook)


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)

