#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
from __future__ import print_function
from tensorflow.keras import layers
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Input, MaxPooling2D, ZeroPadding2D)
from nets.attention_module import (SE_module, ECA_module, CBAM_module, EPSA_module)   # import 注意力模块
from config_file import get_config

config = get_config()

# 第一种 residual 模块
def BTNK_1(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation_rate=1, attention=config.attention, attention_id=0):

    filters1, filters2, out_size = filters

    prename = 'stage{}_'.format(stage)

    x = Conv2D(filters1, (1, 1), strides=strides, name=prename + 'BTNK1_' + block + '_block1' + '_Conv_1', use_bias=False)(input_tensor)
    x = BatchNormalization(name=prename + 'BTNK1_' + block + '_block1' + '_BN')(x)
    x = Activation('relu', name=prename + 'BTNK1_' + block + '_block1' + '_relu')(x)

    # if attention == 'EPSA_module':
    #     x = EPSA_module(x, filters2, stage, block, attention_id)
    # elif attention == 'SE_module':
    #     x = SE_module(x, attention_id)
    # elif attention == 'ECA_module':
    #     x = ECA_module(x, attention_id)
    # elif attention == 'CBAM_module':
    #     x =CBAM_module(x, attention_id)

    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation_rate, name=prename + 'BTNK1_' + block + '_block2' + '_Conv_3', use_bias=False)(x)
    x = BatchNormalization(name=prename + 'BTNK1_' + block + '_block2' + '_BN')(x)
    x = Activation('relu', name=prename + 'BTNK1_' + block + '_block2' + '_relu')(x)

    x = Conv2D(out_size, (1, 1), name=prename + 'BTNK1_' + block + '_block3' + '_Conv_1', use_bias=False)(x)
    x = BatchNormalization(name=prename + 'BTNK1_' + block + '_block3' + '_BN')(x)

    if attention == 'EPSA_module':
        x = EPSA_module(x, filters2, stage, block, attention_id)
    if attention == 'SE_module':
        x = SE_module(x, attention_id)
    elif attention == 'ECA_module':
        x = ECA_module(x, attention_id)
    elif attention == 'CBAM_module':
        x =CBAM_module(x, attention_id)

    shortcut = Conv2D(out_size, (1, 1), strides=strides, name=prename + 'BTNK1_' + block + '_resblock' + '_Conv_1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(name=prename + 'BTNK1_' + block + '_resblock' + '_BN')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu', name=prename + 'BTNK1_' + block + '_Relu')(x)
    return x

# 第二种 residual 模块
def BTNK_2(input_tensor, kernel_size, filters, stage, block, dilation_rate=1, attention=config.attention, attention_id=0):

    prename = 'stage{}_'.format(stage)

    filters1, filters2, out_size = filters

    x = Conv2D(filters1, (1, 1), name=prename + 'BTNK2_' + block + '_block1' + '_Conv_1', use_bias=False)(input_tensor)
    x = BatchNormalization(name=prename + 'BTNK2_' + block + '_block1' + '_BN')(x)
    x = Activation('relu', name=prename + 'BTNK2_' + block + '_block1' + '_relu')(x)

    # if attention == 'EPSA_module':
    #     x = EPSA_module(x, filters2, stage, block, attention_id)
    # elif attention == 'SE_module':
    #     x = SE_module(x, attention_id)
    # elif attention == 'ECA_module':
    #     x = ECA_module(x, attention_id)
    # elif attention == 'CBAM_module':
    #     x = CBAM_module(x, attention_id)

    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation_rate, name=prename + 'BTNK2_' + block + '_block2' + '_Conv_3', use_bias=False)(x)
    x = BatchNormalization(name=prename + 'BTNK2_' + block + '_block2' + '_BN')(x)
    x = Activation('relu', name=prename + 'BTNK2_' + block + '_block2' + '_relu')(x)

    x = Conv2D(out_size, (1, 1), name=prename + 'BTNK2_' + block + '_block3' + '_Conv_1', use_bias=False)(x)
    x = BatchNormalization(name=prename + 'BTNK2_' + block + '_block3' + '_BN')(x)

    if attention == 'EPSA_module':
        x = EPSA_module(x, filters2, stage, block, attention_id)
    if attention == 'SE_module':
        x = SE_module(x, attention_id)
    elif attention == 'ECA_module':
        x = ECA_module(x, attention_id)
    elif attention == 'CBAM_module':
        x = CBAM_module(x, attention_id)

    x = layers.add([x, input_tensor])
    x = Activation('relu', name=prename + 'BTNK2_' + block + '_Relu')(x)
    return x


# ResNet 结构
def get_resnet50_encoder(inputs_size, downsample_factor=8):
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
    img_input = Input(shape=inputs_size)

    # Conv 7*7 64 2
    # 473 * 473 * 3 --> 237 * 237 * 128
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(img_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1, 1), name='conv2_pad')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), name='conv2', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv2')(x)
    x = Activation(activation='relu')(x)

    x = ZeroPadding2D(padding=(1, 1), name='conv3_pad')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), name='conv3', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv3')(x)
    x = Activation(activation='relu')(x)

    # Maxpooling 3*3 2
    # 237 * 237 * 128 --> 119 * 119 * 128
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 119 * 119 * 128 --> 119 * 119 * 256   stride=1
    x = BTNK_1(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), attention=config.attention, attention_id=1)
    x = BTNK_2(x, 3, [64, 64, 256], stage=2, block='b', attention=config.attention, attention_id=2)
    x = BTNK_2(x, 3, [64, 64, 256], stage=2, block='c', attention=config.attention, attention_id=3)

    # 119 * 119 * 256 --> 60 * 60 * 512     stride=2
    x = BTNK_1(x, 3, [128, 128, 512], stage=3, block='a', attention=config.attention, attention_id=4)
    x = BTNK_2(x, 3, [128, 128, 512], stage=3, block='b', attention=config.attention, attention_id=5)
    x = BTNK_2(x, 3, [128, 128, 512], stage=3, block='c', attention=config.attention, attention_id=6)
    x = BTNK_2(x, 3, [128, 128, 512], stage=3, block='d', attention=config.attention, attention_id=7)

    # 60 * 60 * 512 --> 30 * 30 * 1024     stride=2
    x = BTNK_1(x, 3, [256, 256, 1024], stage=4, block='a', strides=(block4_stride,block4_stride), attention=config.attention, attention_id=8)
    x = BTNK_2(x, 3, [256, 256, 1024], stage=4, block='b', dilation_rate=block4_dilation, attention=config.attention, attention_id=9)
    x = BTNK_2(x, 3, [256, 256, 1024], stage=4, block='c', dilation_rate=block4_dilation, attention=config.attention, attention_id=10)
    x = BTNK_2(x, 3, [256, 256, 1024], stage=4, block='d', dilation_rate=block4_dilation, attention=config.attention, attention_id=11)
    x = BTNK_2(x, 3, [256, 256, 1024], stage=4, block='e', dilation_rate=block4_dilation, attention=config.attention, attention_id=12)
    x = BTNK_2(x, 3, [256, 256, 1024], stage=4, block='f', dilation_rate=block4_dilation, attention=config.attention, attention_id=13)
    f4 = x

    # 30 * 30 * 1024 --> 30 * 30 * 2048     stride=1
    x = BTNK_1(x, 3, [512, 512, 2048], stage=5, block='a', strides=(1,1), dilation_rate=block4_dilation, attention=config.attention, attention_id=14)
    x = BTNK_2(x, 3, [512, 512, 2048], stage=5, block='b', dilation_rate=block5_dilation, attention=config.attention, attention_id=15)
    x = BTNK_2(x, 3, [512, 512, 2048], stage=5, block='c', dilation_rate=block5_dilation, attention=config.attention, attention_id=16)
    f5 = x 

    return img_input, f4, f5
