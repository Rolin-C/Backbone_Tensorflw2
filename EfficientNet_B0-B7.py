from tensorflow.keras.layers import *
from tensorflow.keras import Model
from keras.layers import (Activation, Add, BatchNormalization, ZeroPadding2D, Conv2D, DepthwiseConv2D, Input, AlphaDropout)
from SEmodule import SE_module

# def swish(x):
#     x = x * sigmoid(x）
#     return x

def MBConv_block(inputs, expand_rate, kernel_size, strides, input_channel, output_channel, dropout_rate, block_id):

    # 起名字
    prename = "MB_" + str(block_id)

    # expand input dimension
    if expand_rate != 1:
        x = Conv2D(filters=expand_rate * input_channel, kernel_size=(1, 1), padding='same', use_bias=False, name=prename + "expand_conv")(inputs)
        x = BatchNormalization(name=prename + "expand_bn")(x)
        x = Activation("swish", name=prename + "expand_activation")(x)
    else:
        x = inputs

    # Deothwise Conv2D
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name=prename + "Depthwise_conv")(x)
    x = BatchNormalization(name=prename + "Depthwise_bn")(x)
    x = Activation("swish", name=prename + "Depthwise_activation")(x)

    # SE module
    x = SE_module(x, block_id)

    # output dimension
    x = Conv2D(filters=output_channel, kernel_size=(1, 1), padding='same', use_bias=False, name=prename + "project_conv")(x)
    x = BatchNormalization(name=prename + "project_bn")(x)

    if strides == 1 and input_channel == output_channel:
        # Dropout layers
        x = Dropout(rate=dropout_rate, name=prename + "drop")(x)
        x = add([inputs, x], name=prename + "add")

    return x


def EfficientNet_B0(input_shape):

    img_input = Input(shape=input_shape)

    # 224 * 224 -> 112 * 112
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), name="First_conv")(img_input)
    x = BatchNormalization(name="first_bn")(x)
    x = Activation("swish", name="first_activation")(x)

    # 112 * 112 -> 112 * 112
    x = MBConv_block(x, expand_rate=1, kernel_size=(3, 3), strides=(1, 1), input_channel=32, output_channel=16, dropout_rate=0.2, block_id=1)

    # 112 * 112 -> 56 * 56
    x = MBConv_block(x, expand_rate=6, kernel_size=(3, 3), strides=(2, 2), input_channel=16, output_channel=24, dropout_rate=0.2, block_id=2)
    x = MBConv_block(x, expand_rate=6, kernel_size=(3, 3), strides=(1, 1), input_channel=24, output_channel=24, dropout_rate=0.2, block_id=3)

    # 56 * 56 -> 28 * 28
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(2, 2), input_channel=24, output_channel=40, dropout_rate=0.2, block_id=4)
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(1, 1), input_channel=40, output_channel=40, dropout_rate=0.2, block_id=5)

    # 28 * 28 -> 14 * 14
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(2, 2), input_channel=40, output_channel=80, dropout_rate=0.2, block_id=6)
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(1, 1), input_channel=80, output_channel=80, dropout_rate=0.2, block_id=7)
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(1, 1), input_channel=80, output_channel=80, dropout_rate=0.2, block_id=8)

    # 14 * 14 -> 14 * 14
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(1, 1), input_channel=80, output_channel=112, dropout_rate=0.2, block_id=9)
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(1, 1), input_channel=112, output_channel=112, dropout_rate=0.2, block_id=10)
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(1, 1), input_channel=112, output_channel=112, dropout_rate=0.2, block_id=11)

    # 14 * 14 -> 7 * 7
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(2, 2), input_channel=112, output_channel=192, dropout_rate=0.2, block_id=12)
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(1, 1), input_channel=192, output_channel=192, dropout_rate=0.2, block_id=13)
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(1, 1), input_channel=192, output_channel=192, dropout_rate=0.2, block_id=14)
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(1, 1), input_channel=192, output_channel=192, dropout_rate=0.2, block_id=15)

    # 7 * 7 -> 7 * 7
    x = MBConv_block(x, expand_rate=6, kernel_size=(5, 5), strides=(2, 2), input_channel=192, output_channel=320, dropout_rate=0.2, block_id=16)

    # return x

    model = Model(img_input, x)
    return model






