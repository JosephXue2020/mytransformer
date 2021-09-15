# coding = 'utf-8'
# datetime: 20200615


import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


# 定义Embedding layer：
class TransformerEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_words, d_model, pos_embedding='nn_encoder', **kwargs):
        super().__init__(**kwargs)
        self.n_words = n_words
        self.d_model = d_model
        self.pos_embedding = pos_embedding  # 暂时没用上，只用nn做embedding

    def build(self, batch_input_shape):
        # sparse input:
        self.seq_len = batch_input_shape[-1]
        self.embedding1 = tf.keras.layers.Embedding(input_dim=self.n_words + 1, output_dim=self.d_model,
                                                    input_length=self.seq_len)
        self.embedding2 = tf.keras.layers.Embedding(input_dim=self.seq_len, output_dim=self.d_model,
                                                    input_length=self.seq_len)
        super().build(batch_input_shape)

    # def call(self, x):
    #     out1 = self.embedding1(x)
    #     out1 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    #     shape_li = list(x.shape)
    #     shape_need = [1] * (len(shape_li) - 1) + [self.seq_len]
    #     pos_index = tf.reshape(tf.range(self.seq_len), shape_need)
    #     out2 = self.embedding2(pos_index)
    #     out = out1 + out2  # 利用broadcast
    #     return out

    def call(self, x):
        out1 = self.embedding1(x)
        out1 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if self.pos_embedding == 'nn_encoder':
            shape_li = list(x.shape)
            shape_need = [1] * (len(shape_li) - 1) + [self.seq_len]
            pos_index = tf.reshape(tf.range(self.seq_len), shape_need)
            out2 = self.embedding2(pos_index)
        elif self.pos_embedding == 'pos_encoder':
            out2 = self.position_encoding()
        else:
            raise ValueError('Wrong position embedding type.')
        out = out1 + out2  # 利用broadcast
        return out

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list() + [self.d_model])

    def position_encoding(self):
        angle_rads = self.get_angles(np.arange(self.seq_len)[:, np.newaxis], np.arange(self.d_model)[np.newaxis, :])
        sines = np.sin(angle_rads[:, 0::2])
        cones = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cones], axis=-1)  # 这里并未原样实现pos_encoding，但并未影响效果
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        return pos * angle_rates

    def get_config(self):
        base_config = super().get_config()
        new_config = {**base_config, 'epsilon': self.eps}
        return new_config


# 定义Mask类，生成各种mask：
class Mask:
    @staticmethod
    def padding_mask(x):
        output = tf.cast(tf.math.equal(x, 0), tf.float32)
        return output[:, np.newaxis, np.newaxis, :]

    @staticmethod
    def look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # 右上为1，将来乘以很小的赋值则softmax为0
        return mask


class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_nothing = 'do_nothing'

    def call(self, q, k, v, mask=None):
        self.do_nothing = 'do_something'
        dim = len(q.shape)
        if dim == 4:
            k_t = tf.transpose(k, perm=[0, 1, 3, 2])
        elif dim == 3:
            k_t = tf.transpose(k, perm=[0, 2, 1])
        else:
            raise ValueError('The qkv should be 3 or 4 dim.')
        qk = tf.matmul(q, k_t)

        dim_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = qk / tf.math.sqrt(dim_k)

        if mask is not None:
            scaled_attention_logits += mask * (-1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape

    def get_config(self):
        base_config = super().get_config()
        new_config = {**base_config}
        return new_config


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(self.d_model)  # 避免多次构造，只需要reshape
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)
        self.attention = Attention()
        self.dense = tf.keras.layers.Dense(self.d_model)

    def build(self, batch_input_shape):
        # self.batch_size = batch_input_shape[0]  # 这个可能会出现问题，输入实际数据前shape可能是(None,8,8,)
        super().build(batch_input_shape)

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, *, context=None, mask=None):  # 必须以关键字指出，避免混淆
        batch_size = tf.shape(x)[0]
        q = self.wq(x)  # (batch_size, seq_len, d_model)
        if context is None:
            k = self.wk(x)  # (batch_size, seq_len, d_model)
            v = self.wv(x)  # (batch_size, seq_len, d_model)
        else:
            k = self.wk(context)  # attention from context
            v = self.wv(context)

        q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention = self.attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output

    def get_config(self):
        base_config = super().get_config()
        new_config = {**base_config}
        return new_config


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = epsilon

    def build(self, batch_input_shape):
        self.gamma = self.add_weight(name='gamma', shape=batch_input_shape[-1:], initializer=tf.ones_initializer(),
                                     trainable=True)
        self.beta = self.add_weight(name='beta', shape=batch_input_shape[-1:], initializer=tf.zeros_initializer(),
                                    trainable=True)
        super().build(batch_input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)  # 或者tf.math.reduce_std
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        new_config = {**base_config, 'epsilon': self.eps}
        return new_config


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def build(self, batch_input_shape):
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.ffn = self.point_wise_feed_forward_network()

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(batch_input_shape)

    def call(self, x, mask=None):
        output = self.mha(x, mask=mask)
        learning_phase = K.learning_phase()
        output = self.dropout1(output, training=learning_phase)
        out1 = self.layernorm1(x + output)  # (batch_size, input_seq_len, d_model)
        output = self.ffn(out1)
        output = self.dropout2(output, training=learning_phase)
        output = self.layernorm2(out1 + output)  # (batch_size, input_seq_len, d_model)
        return output

    def point_wise_feed_forward_network(self):
        return tf.keras.Sequential(
            [tf.keras.layers.Dense(self.dff, activation='relu'), tf.keras.layers.Dense(self.d_model)])

    def get_config(self):
        base_config = super().get_config()
        new_config = {**base_config, 'dropout_rate': self.dropout_rate}
        return new_config


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def build(self, batch_input_shape):
        self.mha1 = MultiHeadAttention(self.d_model, self.num_heads)
        self.mha2 = MultiHeadAttention(self.d_model, self.num_heads)

        self.ffn = self.point_wise_feed_forward_network()

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(batch_input_shape)

    def call(self, x, context, mask_en=None, mask_de=None):
        learning_phase = K.learning_phase()
        out1 = self.mha1(x, mask=mask_de)
        out1 = self.dropout1(out1, training=learning_phase)
        out1 = self.layernorm1(x + out1)

        out2 = self.mha2(out1, context=context, mask=mask_en)
        out2 = self.dropout2(out2, training=learning_phase)
        out2 = self.layernorm2(out1 + out2)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=learning_phase)
        out3 = self.layernorm3(out2 + ffn_out)

        return out3

    def point_wise_feed_forward_network(self):
        return tf.keras.Sequential(
            [tf.keras.layers.Dense(self.dff, activation='relu'), tf.keras.layers.Dense(self.d_model)])

    def get_config(self):
        base_config = super().get_config()
        new_config = {**base_config, 'dropout_rate': self.dropout_rate}
        return new_config


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate=0.1,
                 pos_embedding='neural_network', **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.dropout_rate = dropout_rate
        self.embedding = TransformerEmbedding(input_vocab_size, d_model, pos_embedding=pos_embedding)

        self.encode_layer = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(n_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, mask):
        out = self.embedding(x)
        learning_phase = K.learning_phase()
        out = self.dropout(out, training=learning_phase)
        for i in range(self.n_layers):
            out = self.encode_layer[i](out, mask)

        return out

    def get_config(self):
        base_config = super().get_config()
        new_config = {**base_config, 'n_layers': self.n_layers, 'd_model': self.d_model, 'num_heads': self.num_heads,
                      'dff': self.dff, 'input_vocab_size': self.input_vocab_size, 'dropout_rate': self.dropout_rate}
        return new_config


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.dff = dff
        self.target_vocab_size = target_vocab_size
        self.dropout_rate = dropout_rate

        self.embedding = TransformerEmbedding(target_vocab_size, d_model)

        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(n_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, context, mask_en, mask_de):
        out = self.embedding(x)
        learning_phase = K.learning_phase()
        out = self.dropout(out, training=learning_phase)
        for i in range(self.n_layers):
            out = self.decoder_layers[i](out, context, mask_en, mask_de)

        return out

    def get_config(self):
        base_config = super().get_config()
        new_config = {**base_config, 'n_layers': self.n_layers, 'd_model': self.d_model, 'num_heads': self.num_heads,
                      'dff': self.dff, 'input_vocab_size': self.target_vocab_size, 'dropout_rate': self.dropout_rate}
        return new_config


class Transformer(tf.keras.Model):
    def __init__(self, n_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, seq_len,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.encoder = Encoder(n_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)

        self.decoder = Decoder(n_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, targets):
        mask_en = Mask.padding_mask(inputs)
        encoder_out = self.encoder(inputs, mask_en)
        mask_de = Mask.padding_mask(targets)
        if K.learning_phase():
            mask_de += Mask.look_ahead_mask(self.seq_len)
        decoder_out = self.decoder(targets, encoder_out, mask_en, mask_de)
        final_out = self.final_layer(decoder_out)

        return final_out
