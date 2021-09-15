# coding = 'utf-8'
# tensorflow version: 2.1


import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K


class CRF(tf.keras.layers.Layer):
    def __init__(self, sparse_target=True, **kwargs):
        self.transitions = None
        super().__init__(**kwargs)
        self.sparse_target = sparse_target
        self.sequence_lengths = None

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        assert len(input_shape) == 3
        self.transitions = self.add_weight(name='transitions', shape=[self.output_dim, self.output_dim],
                                           initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        self.sequence_lengths = tf.reduce_sum(tf.ones_like(inputs[:, :, 0], dtype='int64'), axis=-1)
        viterbi_sequence, _ = tfa.text.crf_decode(inputs, self.transitions, self.sequence_lengths)
        output = tf.one_hot(viterbi_sequence, inputs.shape[-1])
        return K.in_train_phase(inputs, output)

    def loss(self, y_true, y_pred):
        if len(y_true.shape) == 3:
            y_true = tf.argmax(y_true, axis=-1)
        log_likelihood, _ = tfa.text.crf_log_likelihood(y_pred, y_true, self.sequence_lengths,
                                                                       transition_params=self.transitions)
        # return -log_likelihood
        return tf.reduce_mean(-log_likelihood)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + [self.output_dim]

    def accuracy(self, y_true, y_pred):
        if len(y_true.shape) == 3:
            y_true = tf.argmax(y_true, axis=-1)
        viterbi_sequence, _ = tfa.text.crf_decode(y_pred, self.transitions, self.sequence_lengths)
        if len(y_pred.shape) == 3:
            output = tf.cast(tf.argmax(y_pred, axis=-1), dtype=viterbi_sequence.dtype)
        y_pred = K.in_train_phase(viterbi_sequence, output)
        y_pred = tf.cast(y_pred, y_true.dtype)
        y_true = tf.cast(y_true, y_pred.dtype)
        is_equal = tf.equal(y_true, y_pred)
        is_equal = tf.cast(is_equal, y_true.dtype)
        return tf.reduce_mean(is_equal)

    def precision_custom(self, y_true, y_pred):
        # 利用一些迂回的手段，只考虑y_true的非字母O的标签：
        if len(y_true.shape) == 3:
            y_true = tf.argmax(y_true, axis=-1)
        viterbi_sequence, _ = tfa.text.crf_decode(y_pred, self.transitions, self.sequence_lengths)
        if len(y_pred.shape) == 3:
            output = tf.cast(tf.argmax(y_pred, axis=-1), dtype=viterbi_sequence.dtype)
        y_pred = K.in_train_phase(viterbi_sequence, output)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, y_pred.dtype)

        temp_tensor = tf.ones_like(y_true, dtype=y_true.dtype) * 4
        not_equal = tf.not_equal(y_true, temp_tensor)
        not_equal = tf.cast(not_equal, y_true.dtype)
        is_equal_temp = tf.equal(y_true, temp_tensor)
        is_equal_temp = tf.cast(is_equal_temp, y_true.dtype)
        y_true += is_equal_temp  # 此举让标签O对应的数字4的位置出现完全不同的数字
        scale = tf.reduce_sum(not_equal) + 1e-9  # 避免出现0值

        is_equal = tf.equal(y_true, y_pred)
        is_equal = tf.cast(is_equal, y_true.dtype)
        return tf.reduce_sum(is_equal)/scale

    def get_config(self):
        config = {
            #"output_dim": self.output_dim,
            #"transitions": K.eval(self.transitions),
        }
        base_config = super().get_config()
        return {**base_config, **config}


class CRF_Loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        pass
