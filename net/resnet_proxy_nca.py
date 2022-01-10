from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Add, Flatten,
    BatchNormalization,
    Conv2D, Dense,
    ReLU,
)
from tensorflow_addons.layers import AdaptiveAveragePooling2D

from .custom_layers import ProxyNCALayer


def ResnetProxyNCA(input_shape, nclass=None, include_nca=True):
    x_input = Input(shape=input_shape)
    x = Conv2D(64, 3, 1, 'same', use_bias=False)(x_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Layer 1
    x_l1_n1 = x
    x = Conv2D(64, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_l1_n1])
    x = ReLU()(x)
    x_l1_n2 = x
    x = Conv2D(64, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_l1_n2])
    x = ReLU()(x)
    # Layer 2
    x_l2_n1 = x
    x = Conv2D(128, 3, 2, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x_l2_n1_shortcut = Conv2D(128, 1, 2, use_bias=False)(x_l2_n1)
    x_l2_n1_shortcut = BatchNormalization()(x_l2_n1_shortcut)
    x = Add()([x, x_l2_n1_shortcut])
    x = ReLU()(x)
    x_l2_n2 = x
    x = Conv2D(128, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_l2_n2])
    x = ReLU()(x)
    # Layer 3
    x_l3_n1 = x
    x = Conv2D(256, 3, 2, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x_l3_n1_shortcut = Conv2D(256, 1, 2, use_bias=False)(x_l3_n1)
    x_l3_n1_shortcut = BatchNormalization()(x_l3_n1_shortcut)
    x = Add()([x, x_l3_n1_shortcut])
    x = ReLU()(x)
    x_l3_n2 = x
    x = Conv2D(256, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_l3_n2])
    x = ReLU()(x)
    # Layer 4
    x_l4_n1 = x
    x = Conv2D(512, 3, 2, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(512, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x_l4_n1_shortcut = Conv2D(512, 1, 2, use_bias=False)(x_l4_n1)
    x_l4_n1_shortcut = BatchNormalization()(x_l4_n1_shortcut)
    x = Add()([x, x_l4_n1_shortcut])
    x = ReLU()(x)
    x_l4_n2 = x
    x = Conv2D(512, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(512, 3, 1, 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_l4_n2])
    x = ReLU()(x)

    x = AdaptiveAveragePooling2D((1, 1))(x)
    x = Flatten()(x)
    # Linear 1
    x = Dense(128, name='predictions')(x)
    if include_nca:
        assert nclass is not None
        # Proxy NCA layer
        x = ProxyNCALayer(nclass, name='proxynca')(x)

    return Model(inputs=x_input, outputs=x, name='resnet_proxynca')
