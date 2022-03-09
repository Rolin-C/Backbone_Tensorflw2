from tensorflow.keras import layers
from tensorflow.keras.layers import *


#----------------------------------------------------#
# make sure the filters can be divided by 8
#----------------------------------------------------#
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#----------------------------------------------------#
# define Bottleneck
#----------------------------------------------------#
def Bottleneck_module(inputs, expansion, in_size, out_size, alpha, stride, block_id, skip_connection, attention=config.attention, attention_id=0, rate=1):
    pointwise_conv_filters = int(out_size * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs

    prefix = 'Bottle{}_'.format(block_id)

    #----------------------------------------------------#
    #   Expand Conv2D
    #----------------------------------------------------#
    if block_id:
        x = Conv2D(expansion * in_size, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = Activation('relu', name=prefix + 'expand_relu')(x)
    else:
        prefix = 'Bottle_'

    #----------------------------------------------------#
    #   Depthwise Conv2D, dilation Conv2D
    #----------------------------------------------------#
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same', dilation_rate=(rate, rate), name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = Activation('relu', name=prefix + 'depthwise_relu')(x)

    #----------------------------------------------------#
    #   project layers to output filters
    #----------------------------------------------------#
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'narrow')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'narrow_BN')(x)

    #----------------------------------------------------#
    #   residual connection
    #----------------------------------------------------#
    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])
    return x

#----------------------------------------------------#
# MobileNetv2 architecture
#----------------------------------------------------#
def get_mobilenetv2_encoder(inputs_size, downsample_factor=8):
    if downsample_factor == 16:
        block4_dilation = 1
        block5_dilation = 2
        block4_stride = 2
    elif downsample_factor == 8:
        block4_dilation = 2
        block5_dilation = 4
        block4_stride = 1
    else:
        raise ValueError('Unsupported factor - `{}`, Use 8 or 16.'.format(downsample_factor))
    #-----------#
    # 473,473,3
    #-----------#
    inputs = Input(shape=inputs_size)

    alpha=1.0
    first_block_filters = _make_divisible(32 * alpha, 8)

    #-------------------------#
    # 473,473,3 -> 237,237,32
    #-------------------------#
    x = Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, name='Conv')(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation('relu', name='Conv_Relu6')(x)

    #-------------------------#
    # 237,237,32 -> 237,237,16
    #-------------------------#
    x = Bottleneck_module(x, expansion=1, in_size=32, out_size=16, alpha=alpha, stride=1, block_id=0, skip_connection=False, attention=config.attention, attention_id=1)

    #-------------------------#
    # 237,237,16 -> 119,119,24
    #-------------------------#
    x = Bottleneck_module(x, expansion=6, in_size=16, out_size=24, alpha=alpha, stride=2, block_id=1, skip_connection=False, attention=config.attention, attention_id=2)
    x = Bottleneck_module(x, expansion=6, in_size=24, out_size=24, alpha=alpha, stride=1, block_id=2, skip_connection=True, attention=config.attention, attention_id=3)

    #-------------------------#
    # 119,119,24 -> 60,60.32
    #-------------------------#
    x = Bottleneck_module(x, expansion=6, in_size=24, out_size=32, alpha=alpha, stride=2, block_id=3, skip_connection=False, attention=config.attention, attention_id=4)
    x = Bottleneck_module(x, expansion=6, in_size=32, out_size=32, alpha=alpha, stride=1, block_id=4, skip_connection=True, attention=config.attention, attention_id=5)
    x = Bottleneck_module(x, expansion=6, in_size=32, out_size=32, alpha=alpha, stride=1, block_id=5, skip_connection=True, attention=config.attention, attention_id=6)

    #-------------------------#
    # 60,60,32 -> 30,30.64
    #-------------------------#
    x = Bottleneck_module(x, in_size=32, out_size=64, alpha=alpha, stride=block4_stride, expansion=6, block_id=6, skip_connection=False, attention=config.attention, attention_id=7)

    x = Bottleneck_module(x, expansion=6, in_size=64, out_size=64, alpha=alpha, stride=1, block_id=7, skip_connection=True, attention=config.attention, attention_id=8, rate=block4_dilation)  # block4_dilation 空洞卷积 比例 2
    x = Bottleneck_module(x, expansion=6, in_size=64, out_size=64, alpha=alpha, stride=1, block_id=8, skip_connection=True, attention=config.attention, attention_id=9, rate=block4_dilation)  # block4_dilation 空洞卷积 比例 2
    x = Bottleneck_module(x, expansion=6, in_size=64, out_size=64, alpha=alpha, stride=1, block_id=9, skip_connection=True, attention=config.attention, attention_id=10, rate=block4_dilation)  # block4_dilation 空洞卷积 比例 2

    #-------------------------#
    # 30,30.64 -> 30,30.96
    #-------------------------#
    x = Bottleneck_module(x, expansion=6, in_size=64, out_size=96, alpha=alpha, stride=1, block_id=10, skip_connection=False, attention=config.attention, attention_id=11, rate=block4_dilation)   # block4_dilation 空洞卷积 比例 2
    x = Bottleneck_module(x, expansion=6, in_size=96, out_size=96, alpha=alpha, stride=1, block_id=11, skip_connection=True, attention=config.attention, attention_id=12, rate=block4_dilation)     # block4_dilation 空洞卷积 比例 2
    x = Bottleneck_module(x, expansion=6, in_size=96, out_size=96, alpha=alpha, stride=1, block_id=12, skip_connection=True, attention=config.attention, attention_id=13, rate=block4_dilation)     # block4_dilation 空洞卷积 比例 2
    
    f4 = x

    #------------------------------------#
    # 30,30.96 -> 30,30,160 -> 30,30,320
    #------------------------------------#
    x = Bottleneck_module(x, expansion=6, in_size=96, out_size=160, alpha=alpha, stride=1, block_id=13, skip_connection=False, attention=config.attention, attention_id=14, rate=block4_dilation)  # block4_dilation 空洞卷积 比例 2
    x = Bottleneck_module(x, expansion=6, in_size=160, out_size=160, alpha=alpha, stride=1, block_id=14, skip_connection=True, attention=config.attention, attention_id=15, rate=block5_dilation)   # block4_dilation 空洞卷积 比例 4
    x = Bottleneck_module(x, expansion=6, in_size=160, out_size=160, alpha=alpha, stride=1, block_id=15, skip_connection=True, attention=config.attention, attention_id=16, rate=block5_dilation)   # block4_dilation 空洞卷积 比例 4
    x = Bottleneck_module(x, expansion=6, in_size=160, out_size=320, alpha=alpha, stride=1, block_id=16, skip_connection=False, attention=config.attention, attention_id=17, rate=block5_dilation) # block4_dilation 空洞卷积 比例 4
    
    f5 = x
    
    return inputs, f4, f5
