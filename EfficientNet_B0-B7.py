import tensorflow.keras.layers as layers
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from SEmodule import SE_module
import math


# define MB_block
def MBConv_block(inputs, kernel_size, input_channel, output_channel, expand_rate, strides, drop_rate, block_id, use_SE):

    # 起名字
    prename = "MB_" + block_id

    # expand input dimension
    if expand_rate != 1:
        x = Conv2D(filters=expand_rate * input_channel, kernel_size=(1, 1), padding='same', use_bias=False, name=prename + "expand_conv")(inputs)
        x = BatchNormalization(name=prename + "expand_bn")(x)
        x = Activation('swish', name=prename + "expand_activation")(x)
    else:
        x = inputs

    # Deothwise Conv2D
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name=prename + "Depthwise_conv")(x)
    x = BatchNormalization(name=prename + "Depthwise_bn")(x)
    x = Activation('swish', name=prename + "Depthwise_activation")(x)

    # SE module
    if use_SE:
        x = SE_module(x, block_id)

    # output dimension
    x = Conv2D(filters=output_channel, kernel_size=(1, 1), padding='same', use_bias=False, name=prename + "project_conv")(x)
    x = BatchNormalization(name=prename + "project_bn")(x)

    if strides == 1 and input_channel == output_channel:
        # Dropout layers
        x = Dropout(rate=drop_rate, name=prename + "drop")(x)
        x = add([inputs, x], name=prename + "add")

    return x


# define expanding of filters, make sure that filters can 被 8 整除
def round_filters(width_coefficient, filters, divisor=8):
    filters = width_coefficient * filters
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters = new_filters + divisor
    return int(new_filters)


# define expanding of block repeats
def round_repeats(depth_coefficient, repeats):
    return int(math.ceil(depth_coefficient * repeats))

#                                                                    在最后的连接层   MB模块中的 drop
def Efficient_net(width_coefficient, input_shape, depth_coefficient, dropout_rate=0.2, MB_drop_rate=0.2, last_layers=False, num_classes=1000):

    # B0's architecture
    # kernel, block_repeat, input_filter, output_filter, expand_time, stride
    block_args = [[3, 1, 32, 16, 1, 1, True],
                  [3, 2, 16, 24, 6, 2, True],
                  [5, 2, 24, 40, 6, 2, True],
                  [3, 3, 40, 80, 6, 2, True],
                  [5, 3, 80, 112, 6, 1, True],
                  [5, 4, 112, 192, 6, 2, True],
                  [3, 1, 192, 320, 6, 1, True]]

    img_input = layers.Input(shape=input_shape)

    # data preprocessing
    x = layers.experimental.preprocessing.Rescaling(1. / 255.)(img_input)
    x = layers.experimental.preprocessing.Normalization()(x)


    # first Conv2D
    x = Conv2D(filters=32, kernel_size=3, strides=2, use_bias=False, name="first_conv")(x)
    x = BatchNormalization(name="first_bn")(x)
    x = Activation('swish', name="first_activation")(x)

    b = 0
    num_blocks = float(sum(i[1] for i in block_args))

    # MB_blocks
    # 重复 stage1, stage2, stage3...
    for i, args in enumerate(block_args):

        # input_filters and output_filters 都随着 width_coefficient 而改变
        args[2] = round_filters(width_coefficient, args[2])
        args[3] = round_filters(width_coefficient, args[3])

        # 重复 MB_block 1次，2次，3次..
        for j in range(round_repeats(depth_coefficient, args[1])):
            x = MBConv_block(x,
                             kernel_size=args[0],
                             input_channel=args[2] if j == 0 else args[3],
                             output_channel=args[3],
                             expand_rate=args[4],
                             strides=args[5] if j == 0 else 1,
                             drop_rate=MB_drop_rate * b / num_blocks,
                             block_id="stage_" + str(i+1) + "_block_" + str(j+1) + "_",
                             use_SE=args[6])

            b = b +1

    if last_layers:
        x = Conv2D(kernel_size=1, padding='same', use_bias=False, name="last_conv")(x)
        x = BatchNormalization(name="last_bn")(x)
        x = Activation('swish', name="last_activation")(x)

        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name="last_dropout")
        x = layers.Dense(units=num_classes, activation="softmax", name="prediction")(x)

    model = Model(img_input, x, name="efficientNet")
    return model


# Efficient_net_B0
def efficient_net_B0(input_shape=(224, 224, 3)):
    return Efficient_net(width_coefficient=1,
                         depth_coefficient=1,
                         input_shape=input_shape,
                         dropout_rate=0.2)

# Efficient_net_B1
def efficient_net_B1(input_shape=(240, 240, 3)):
    return Efficient_net(width_coefficient=1,
                         depth_coefficient=1.1,
                         input_shape=input_shape,
                         dropout_rate=0.2)

# Efficient_net_B2
def efficient_net_B2(input_shape=(260, 260, 3)):
    return Efficient_net(width_coefficient=1.1,
                         depth_coefficient=1.2,
                         input_shape=input_shape,
                         dropout_rate=0.3)

# Efficient_net_B3
def efficient_net_B3(input_shape=(300, 300, 3)):
    return Efficient_net(width_coefficient=1.2,
                         depth_coefficient=1.4,
                         input_shape=input_shape,
                         dropout_rate=0.3)

# Efficient_net_B4
def efficient_net_B4(input_shape=(380, 380, 3)):
    return Efficient_net(width_coefficient=1.4,
                         depth_coefficient=1.8,
                         input_shape=input_shape,
                         dropout_rate=0.4)

# Efficient_net_B5
def efficient_net_B5(input_shape=(456, 456, 3)):
    return Efficient_net(width_coefficient=1.6,
                         depth_coefficient=2.2,
                         input_shape=input_shape,
                         dropout_rate=0.4)

# Efficient_net_B6
def efficient_net_B6(input_shape=(528, 528, 3)):
    return Efficient_net(width_coefficient=1.8,
                         depth_coefficient=2.6,
                         input_shape=input_shape,
                         dropout_rate=0.5)

# Efficient_net_B7
def efficient_net_B7(input_shape=(600, 600, 3)):
    return Efficient_net(width_coefficient=2.0,
                         depth_coefficient=3.1,
                         input_shape=input_shape,
                         dropout_rate=0.5)