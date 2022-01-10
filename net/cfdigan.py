import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Embedding, LeakyReLU
from tensorflow_addons.layers import InstanceNormalization

from .custom_initializer import ScaledRandomNormal
from .custom_layers import PixelwiseNormalization, MinibatchStandardDeviation, WeightedAdd


# From https://github.com/NVlabs/stylegan
def _blur2d(x, f=(1, 2, 1), normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[-1]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0, 0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, stride, stride, 1]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME')
    x = tf.cast(x, orig_dtype)
    return x


def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = tf.shape(x)
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, factor, 1, factor, 1])
    x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
    return x


def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool2d().
    return tf.nn.avg_pool2d(x, factor, factor, 'VALID')


def blur2d(x, f=(1, 2, 1), normalize=True):
    @tf.custom_gradient
    def func(x):
        y = _blur2d(x, f, normalize)

        @tf.custom_gradient
        def grad(dy):
            dx = _blur2d(dy, f, normalize, flip=True)
            return dx, lambda ddx: _blur2d(ddx, f, normalize)
        return y, grad
    return func(x)


def upscale2d(x, factor=2):
    @tf.custom_gradient
    def func(x):
        y = _upscale2d(x, factor)

        @tf.custom_gradient
        def grad(dy):
            dx = _downscale2d(dy, factor, gain=factor**2)
            return dx, lambda ddx: _upscale2d(ddx, factor)
        return y, grad
    return func(x)


def downscale2d(x, factor=2):
    @tf.custom_gradient
    def func(x):
        y = _downscale2d(x, factor)

        @tf.custom_gradient
        def grad(dy):
            dx = _upscale2d(dy, factor, gain=1/factor**2)
            return dx, lambda ddx: _downscale2d(ddx, factor)
        return y, grad
    return func(x)


def nf(stage):
    return 2 ** (10 - stage)


def GeneratorBase(nch, nclass):
    stage = 1
    # Encode
    x_input = Input(shape=(4, 4, nch), name='encoder_in')
    x = FromRGB(nf(stage), name='encoder_from_rgb_1')(x_input)

    # Embed
    x_identity_index = Input(shape=(1,), name='identity_index')
    noise = Embedding(nclass, 512)(x_identity_index)
    noise = Reshape((-1,))(noise)
    # Normalize noise
    noise = PixelwiseNormalization()(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2)(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2)(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2)(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2)(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2)(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2, name='mapping_style')(noise)

    # Decode
    x = ConvBlock(nf(stage), name=f'decoder_conv_{stage}')(x)
    x = Epilogue(name=f'decoder_epilogue_{stage}')([x, noise])  # 4x4x512

    x = ToRGB(name=f'decoder_to_rgb_{stage}')(x)

    return Model([x_input, x_identity_index], x, name=f'Generator_{stage}')


def double_generator(model: Model):
    # Double the resolution
    shape = model.get_layer('encoder_in').input.shape
    stage = int(np.log2(shape[1]))  # 3
    c = shape[3]
    res = 2 ** (stage + 1)  # 16

    # Get identity index input, mapping output from the previous model
    x_identity_index = model.get_layer('identity_index').input
    style = model.get_layer('mapping_style').output

    # Double the encoder
    # Create a new higher resolution input
    x_input = Input(shape=(res, res, c), name='encoder_in')  # 16x16xc
    x = FromRGB(nf(stage), name=f'encoder_from_rgb_{stage}')(x_input)  # 16x16x128

    x = ConvBlock(nf(stage), name=f'encoder_conv_{stage}')(x)  # 16x16x128
    x = DownscaleBlock(nf(stage - 1), name=f'encoder_downscale_{stage}')(x)  # 8x8x256

    img = Downscale2D()(x_input)
    y = model.get_layer(f'encoder_from_rgb_{stage - 1}')(img)

    x = WeightedAdd(name='encoder_weighted_add')([y, x])  # 8x8x256

    # Remaining blocks
    for s in range(stage - 1, 1, -1):
        x = model.get_layer(f'encoder_conv_{s}')(x)
        x = model.get_layer(f'encoder_downscale_{s}')(x)
    # == End of encoder ==

    # Double the decoder
    # First 4x4 parts only consist of two layers
    x = model.get_layer('decoder_conv_1')(x)
    x = model.get_layer('decoder_epilogue_1')([x, style])
    # Remaining blocks
    for s in range(2, stage):
        x = model.get_layer(f'decoder_upscale_{s}')(x)
        x = model.get_layer(f'decoder_epilogue_{s}_1')([x, style])
        x = model.get_layer(f'decoder_conv_{s}')(x)
        x = model.get_layer(f'decoder_epilogue_{s}_2')([x, style])

    y = Upscale2D()(model.get_layer(f'decoder_to_rgb_{stage - 1}')(x))

    # New layers
    x = UpscaleBlock(nf(stage), name=f'decoder_upscale_{stage}')(x)  # 16x16x128
    x = Epilogue(name=f'decoder_epilogue_{stage}_1')([x, style])
    x = ConvBlock(nf(stage), name=f'decoder_conv_{stage}')(x)
    x = Epilogue(name=f'decoder_epilogue_{stage}_2')([x, style])  # 16x16x128

    x = ToRGB(name=f'decoder_to_rgb_{stage}')(x)

    x = WeightedAdd(name='decoder_weighted_add')([y, x])

    return Model([x_input, x_identity_index], x, name=f'Generator_{stage}')


def DiscriminatorBase(nch):
    stage = 1
    x_input = Input(shape=(4, 4, nch), name='image_in')
    x = FromRGB(nf(stage), name=f'from_rgb_{stage}')(x_input)  # 4x4x512

    x = MinibatchStandardDeviation(4, 1, name='minibatch_std')(x)

    x = ConvBlock(nf(stage), name=f'conv_{stage}')(x)
    x = Flatten(name='flatten')(x)  # 8192

    x = Dense(nf(stage), kernel_initializer=ScaledRandomNormal(), name='dense')(x)
    x = LeakyReLU(0.2, name='leaky_relu')(x)

    x = Dense(1, kernel_initializer=ScaledRandomNormal(1), name='score')(x)

    return Model(x_input, x, name=f'Discriminator_{stage}')


def double_discriminator(model: Model):
    shape = model.get_layer('image_in').input.shape
    stage = int(np.log2(shape[1]))  # 3
    c = shape[3]
    res = 2 ** (stage + 1)  # 16

    # Create a new higher resolution input
    x_input = Input(shape=(res, res, c), name='image_in')
    x = FromRGB(nf(stage), name=f'from_rgb_{stage}')(x_input)  # 16x16x128

    # New layers
    x = ConvBlock(nf(stage), name=f'conv_{stage}')(x)
    x = DownscaleBlock(nf(stage - 1), name=f'downscale_{stage}')(x)

    img = Downscale2D()(x_input)
    y = model.get_layer(f'from_rgb_{stage - 1}')(img)

    x = WeightedAdd(name='weighted_add')([y, x])

    # Remaining blocks
    for s in range(stage - 1, 1, -1):
        x = model.get_layer(f'conv_{s}')(x)
        x = model.get_layer(f'downscale_{s}')(x)

    x = model.get_layer(f'minibatch_std')(x)

    x = model.get_layer('conv_1')(x)
    x = model.get_layer('flatten')(x)

    x = model.get_layer('dense')(x)
    x = model.get_layer('leaky_relu')(x)

    x = model.get_layer('score')(x)

    return Model(x_input, x, name=f'Discriminator_{stage}')


def Generator(input_shape, nclass):
    # Encode
    x_input = Input(shape=input_shape)
    img = x_input  # 128x128xc
    x = FromRGB(16)(img)  # 128x128x16

    x = ConvBlock(16)(x)
    x = DownscaleBlock(32)(x)  # 64x64x32

    img = Downscale2D()(img)  # 64x64xc
    y = FromRGB(32)(img)  # 64x64x32

    x = WeightedAdd(name='WeightedAdd_64')([y, x])

    x = ConvBlock(32)(x)
    x = DownscaleBlock(64)(x)  # 32x32x64

    img = Downscale2D()(img)  # 32x32xc
    y = FromRGB(64)(img)  # 32x32x64

    x = WeightedAdd(name='WeightedAdd_32')([y, x])

    x = ConvBlock(64)(x)
    x = DownscaleBlock(128)(x)  # 16x16x128

    img = Downscale2D()(img)  # 16x16xc
    y = FromRGB(128)(img)  # 16x16x128

    x = WeightedAdd(name='WeightedAdd_16')([y, x])

    x = ConvBlock(128)(x)
    x = DownscaleBlock(256)(x)  # 8x8x256

    img = Downscale2D()(img)  # 8x8xc
    y = FromRGB(256)(img)  # 8x8x256

    x = WeightedAdd(name='WeightedAdd_8')([y, x])

    x = ConvBlock(256)(x)
    x = DownscaleBlock(512)(x)  # 4x4x512

    img = Downscale2D()(img)  # 4x4xc
    y = FromRGB(512)(img)  # 4x4x512

    x = WeightedAdd(name='WeightedAdd_4')([y, x])

    # Embed
    x_identity_index = Input(shape=(1,))
    noise = Embedding(nclass, 512)(x_identity_index)
    noise = Reshape((-1,))(noise)
    # Normalize noise
    noise = PixelwiseNormalization()(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2)(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2)(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2)(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2)(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2)(noise)
    noise = Dense(512, kernel_initializer=ScaledRandomNormal(lrmul=0.01))(noise)
    noise = LeakyReLU(0.2)(noise)

    # Decode
    x = ConvBlock(512)(x)
    x = Epilogue()([x, noise])  # 4x4x512

    img_out = ToRGB(name='ToRGB_4')(x)

    x = UpscaleBlock(256)(x)
    x = Epilogue()([x, noise])
    x = ConvBlock(256)(x)
    x = Epilogue()([x, noise])  # 8x8x256

    img = ToRGB(name='ToRGB_8')(x)
    img_out = Upscale2D()(img_out)

    img_out = WeightedAdd(name='Decoder_WeightedAdd_8')([img_out, img])

    x = UpscaleBlock(128)(x)
    x = Epilogue()([x, noise])
    x = ConvBlock(128)(x)
    x = Epilogue()([x, noise])  # 16x16x128

    img = ToRGB(name='ToRGB_16')(x)
    img_out = Upscale2D()(img_out)

    img_out = WeightedAdd(name='Decoder_WeightedAdd_16')([img_out, img])

    x = UpscaleBlock(64)(x)
    x = Epilogue()([x, noise])
    x = ConvBlock(64)(x)
    x = Epilogue()([x, noise])  # 32x32x64

    img = ToRGB(name='ToRGB_32')(x)
    img_out = Upscale2D()(img_out)

    img_out = WeightedAdd(name='Decoder_WeightedAdd_32')([img_out, img])

    x = UpscaleBlock(32)(x)
    x = Epilogue()([x, noise])
    x = ConvBlock(32)(x)
    x = Epilogue()([x, noise])  # 64x64x32

    img = ToRGB(name='ToRGB_64')(x)
    img_out = Upscale2D()(img_out)

    img_out = WeightedAdd(name='Decoder_WeightedAdd_64')([img_out, img])

    x = UpscaleBlock(16)(x)
    x = Epilogue()([x, noise])
    x = ConvBlock(16)(x)
    x = Epilogue()([x, noise])  # 128x128x16

    img = ToRGB(name='ToRGB_128')(x)
    img_out = Upscale2D()(img_out)

    img_out = WeightedAdd(name='Decoder_WeightedAdd_128')([img_out, img])

    return Model(inputs=[x_input, x_identity_index], outputs=img_out, name='dev_v3_generator')


def Discriminator(input_shape):
    c = input_shape[-1]
    x_input = Input(shape=input_shape)
    img = x_input  # 128x128xc
    x = FromRGB(16)(img)  # 128x128x16

    x = ConvBlock(16)(x)
    x = DownscaleBlock(32)(x)  # 64x64x32

    img = Downscale2D()(img)  # 64x64xc
    y = FromRGB(32)(img)  # FromRGB 64x64x32

    x = WeightedAdd(name='WeightedAdd_64')([y, x])

    x = ConvBlock(32)(x)
    x = DownscaleBlock(64)(x)  # 32x32x64

    img = Downscale2D()(img)  # 32x32xc
    y = FromRGB(64)(img)  # 32x32x64

    x = WeightedAdd(name='WeightedAdd_32')([y, x])

    x = ConvBlock(64)(x)
    x = DownscaleBlock(128)(x)  # 16x16x128

    img = Downscale2D()(img)  # 16x16xc
    y = FromRGB(128)(img)  # 16x16x128

    x = WeightedAdd(name='WeightedAdd_16')([y, x])

    x = ConvBlock(128)(x)
    x = DownscaleBlock(256)(x)  # 8x8x256

    img = Downscale2D()(img)  # 8x8xc
    y = FromRGB(256)(img)  # 8x8x256

    x = WeightedAdd(name='WeightedAdd_8')([y, x])

    x = ConvBlock(256)(x)
    x = DownscaleBlock(512)(x)  # 4x4x512

    img = Downscale2D()(img)  # 4x4xc
    y = FromRGB(512)(img)  # 4x4x512

    x = WeightedAdd(name='WeightedAdd_4')([y, x])

    x = MinibatchStandardDeviation(4, 1)(x)

    x = ConvBlock(512)(x)
    x = Flatten()(x)  # 8192

    x = Dense(512, kernel_initializer=ScaledRandomNormal())(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(1, kernel_initializer=ScaledRandomNormal(1))(x)

    return Model(inputs=x_input, outputs=x, name='dev_v3_discriminator')


class ToRGB(Layer):
    def __init__(self, nch=3, **kwargs):
        super(ToRGB, self).__init__(**kwargs)
        self.conv_1 = Conv2D(nch, 1, 1, 'same', kernel_initializer=ScaledRandomNormal(1))

    def call(self, inputs, **kwargs):
        return self.conv_1(inputs)


class FromRGB(Layer):
    def __init__(self, filters, **kwargs):
        super(FromRGB, self).__init__(**kwargs)
        self.conv_1 = Conv2D(filters, 1, 1, 'same', kernel_initializer=ScaledRandomNormal())

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv_1(x)
        x = tf.nn.leaky_relu(x, 0.2)
        return x


class Downscale2D(Layer):
    def __init__(self, factor=2, **kwargs):
        super(Downscale2D, self).__init__(**kwargs)
        self.factor = factor

    def call(self, inputs, **kwargs):
        return downscale2d(inputs, self.factor)


class Upscale2D(Layer):
    def __init__(self, factor=2, **kwargs):
        super(Upscale2D, self).__init__(**kwargs)
        self.factor = factor

    def call(self, inputs, **kwargs):
        return upscale2d(inputs, self.factor)


class Epilogue(Layer):
    def __init__(self, **kwargs):
        super(Epilogue, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2 and len(input_shape[0]) == 4
        self.in_c = input_shape[0][-1]  # number of input channels
        self.pn_1 = PixelwiseNormalization()
        self.in_1 = InstanceNormalization()
        self.sm_1 = Dense(self.in_c * 2, kernel_initializer=ScaledRandomNormal(1))

    def call(self, inputs, **kwargs):
        x = inputs[0]
        style = inputs[1]

        x = self.pn_1(x)
        x = self.in_1(x)
        style = self.sm_1(style)
        style = tf.reshape(style, [-1, 2, 1, 1, self.in_c])
        x = x * (style[:, 0] + 1) + style[:, 1]

        return x


class UpscaleBlock(Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(UpscaleBlock, self).__init__(**kwargs)
        self.conv_1 = Conv2DTranspose(filters, kernel_size, 2, 'same',
                                      kernel_initializer=ScaledRandomNormal())

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv_1(x)
        x = tf.nn.leaky_relu(x, 0.2)
        x = blur2d(x)
        return x


class DownscaleBlock(Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(DownscaleBlock, self).__init__(**kwargs)

        self.conv_1 = Conv2D(filters, kernel_size, 2, 'same',
                             kernel_initializer=ScaledRandomNormal())

    def call(self, inputs, **kwargs):
        x = inputs
        x = blur2d(x)
        x = self.conv_1(x)
        x = tf.nn.leaky_relu(x, 0.2)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] // 2, input_shape[2] // 2, input_shape[3]


class ConvBlock(Layer):
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.conv_1 = Conv2D(filters, kernel_size, strides, 'same', kernel_initializer=ScaledRandomNormal())

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv_1(x)
        x = tf.nn.leaky_relu(x, 0.2)
        return x
