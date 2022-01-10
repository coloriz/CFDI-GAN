import tensorflow as tf
from tensorflow.keras.initializers import Constant, Ones
from tensorflow.keras.layers import Layer


class ProxyNCALayer(Layer):
    def __init__(self, units, **kwargs):
        super(ProxyNCALayer, self).__init__(**kwargs)
        self.classes = units

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.embedding_size = input_shape[1]
        self.kernel = self.add_weight(
            name='kernel', shape=(self.classes, self.embedding_size), initializer=Ones(), trainable=True)

    def call(self, inputs, **kwargs):
        """[B, embedding_size] -> [B, classes]"""
        batch_size = tf.shape(inputs)[0]
        x = 3 * tf.math.l2_normalize(inputs, axis=-1)
        x_expand = tf.reshape(tf.tile(x, [1, self.classes]), [batch_size, self.classes, self.embedding_size])
        proxies_normalized = 3 * tf.math.l2_normalize(self.kernel, axis=-1)
        weights_expand = tf.reshape(tf.tile(proxies_normalized, [batch_size, 1]), [batch_size, self.classes, self.embedding_size])
        return tf.math.reduce_sum(tf.math.squared_difference(x_expand, weights_expand), axis=-1)


class PixelwiseNormalization(Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(PixelwiseNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs, **kwargs):
        x = inputs
        return x * tf.math.rsqrt(tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True) + self.epsilon)


class MinibatchStandardDeviation(Layer):
    def __init__(self, group_size=4, num_new_features=1, epsilon=1e-8, **kwargs):
        super(MinibatchStandardDeviation, self).__init__(**kwargs)
        self.group_size = group_size
        self.num_new_features = num_new_features
        self.epsilon = epsilon

    def call(self, inputs, **kwargs):
        x = inputs
        shape = tf.shape(x)  # [NHWC]  Input shape.
        h, w, c = shape[-3], shape[-2], shape[-1]
        group_size = tf.minimum(self.group_size, shape[0])  # Minibatch must be divisible by (or smaller than) group_size.
        y = tf.reshape(x, [group_size, -1, self.num_new_features, h, w, c // self.num_new_features])  # [GMnHWc] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)  # [GMnHWc] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMnHWc] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)  # [MnHWc] Calc variance over group.
        y = tf.sqrt(y + self.epsilon)  # [MnHWc] Calc stddev over group.
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)  # [Mn111] Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[4])  # [Mn11] Split channels into c channel groups
        y = tf.cast(y, x.dtype)  # [Mn11] Cast back to original data type.
        y = tf.tile(y, [group_size, h, w, 1])  # [NHW1] Replicate over group and pixels.
        return tf.concat([x, y], axis=-1)  # [NHWC]  Append as new fmap.


class WeightedAdd(Layer):
    def __init__(self, alpha=0., **kwargs):
        super(WeightedAdd, self).__init__(**kwargs)
        self.alpha = self.add_weight('alpha', [], tf.float32, Constant(alpha), trainable=False)

    def set_alpha(self, alpha):
        if alpha <= 0:
            alpha = 0.
        elif alpha >= 1:
            alpha = 1.
        self.alpha.assign(alpha)

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        return (1 - self.alpha) * inputs[0] + self.alpha * inputs[1]
