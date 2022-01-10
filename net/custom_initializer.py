import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Initializer


class ScaledRandomNormal(Initializer):
    def __init__(self, gain=None, lrmul=1):
        self.gain = gain if gain is not None else np.sqrt(2)
        self.lrmul = lrmul

    def __call__(self, shape, dtype=None):
        fan_in = np.prod(shape[:-1])
        std = self.gain / np.sqrt(fan_in)
        init_std = 1.0 / self.lrmul
        runtime_coef = std * self.lrmul
        return tf.random.normal(shape, 0, init_std) * runtime_coef
