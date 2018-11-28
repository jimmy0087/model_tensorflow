from base_model import *
import tensorflow as tf
class MobileNetV2(BASE_MODEL):
    def __init__(self,num_classes=10,trainable = True):
        BASE_MODEL.__init__(self,num_classes=num_classes,trainable = True)

    def _conv_block(self,input_tensor, filter,kernel = (3, 3),strides = (1, 1), name=''):
        x = input_tensor
        with tf.variable_scope(name) as scope:
            x = tf.layers.Conv2D(filter, kernel,
                              padding='same',
                              use_bias=False,
                              strides=strides,
                              name='conv1')(x)
            x = tf.layers.BatchNormalization(name='conv1_bn')(x)
            x = tf.nn.relu(x,'relu')
        return x

    def _depthwise_conv_block(self,input_tensor,filter,alpha,depth_multiplier = 1,strides = (1, 1),name=''):
        x = input_tensor
        filter_=int(filter*alpha)
        with tf.variable_scope(name) as scope:
            x = tf.keras.layers.DepthwiseConv2D((3, 3),
                                       padding='same',
                                       depth_multiplier=depth_multiplier,
                                       strides=strides,
                                       use_bias=False,
                                       name='conv_dw')(x)
            x = tf.layers.BatchNormalization(name='conv_dw_bn')(x)
            x = tf.nn.relu(x, 'conv_dw_relu')

            x = tf.layers.Conv2D(filter_, (1, 1),
                              padding='same',
                              use_bias=False,
                              strides=(1, 1),
                              name='conv_pw')(x)
            x = tf.layers.BatchNormalization(name='conv_pw_bn')(x)
            x = tf.nn.relu6(x, 'conv_pw_relu')
        return x

    def _make_divisible(self,v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _inverted_res_block(self,input_tensor, expansion, stride, alpha, filters, name=''):
        in_channels = input_tensor.get_shape()[-1]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = self._make_divisible(pointwise_conv_filters, 8)
        x = input_tensor

        with tf.variable_scope(name) as scope:

            # Expand
            x = tf.layers.Conv2D(expansion * in_channels,
                              kernel_size=1,
                              padding='same',
                              use_bias=False,
                              activation=None,
                              name='ex_conv')(x)
            x = tf.layers.BatchNormalization(epsilon=1e-3,
                                          momentum=0.999,
                                          name='ex_bn')(x)
            x = tf.nn.relu6(x, 'ex_relu')
            # Depthwise
            x = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                       strides=stride,
                                       activation=None,
                                       use_bias=False,
                                       padding='same',
                                       name='dw_conv')(x)
            x = tf.layers.BatchNormalization(epsilon=1e-3,
                                          momentum=0.999,
                                          name='dw_bn')(x)
            x = tf.nn.relu6(x, 'dw_relu')
            # Project
            x = tf.layers.Conv2D(pointwise_filters,
                              kernel_size=1,
                              padding='same',
                              use_bias=False,
                              activation=None,
                              name='pw_conv')(x)
            x = tf.layers.BatchNormalization(
                epsilon=1e-3, momentum=0.999, name='pw_bn')(x)

            if in_channels == pointwise_filters and stride == 1:
                return tf.add_n([input_tensor, x],name='add')
        return x

    def inference(self,input):
        x = input
        alpha = 1.0
        depth_multiplier = 1
        with tf.variable_scope("MobileNetV2",reuse=tf.AUTO_REUSE) as scope:
            first_block_filters = self._make_divisible(32 * alpha, 8)

            x = self._conv_block(x,first_block_filters,3,strides=(2,2),name='cov_1')

            x = self._inverted_res_block(x, filters=16, alpha=alpha, stride=1,expansion=1, name='block_2')

            x = self._inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, name='block_3')
            x = self._inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, name='block_4')

            x = self._inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, name='block_5')
            x = self._inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, name='block_6')
            x = self._inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, name='block_7')

            x = self._inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=6, name='block_8')
            x = self._inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, name='block_9')
            x = self._inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, name='block_10')
            x = self._inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, name='block_11')

            x = self._inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, name='block_11')
            x = self._inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, name='block_12')
            x = self._inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, name='block_13')

            x = self._inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, name='block_14')
            x = self._inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, name='block_15')
            x = self._inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, name='block_16')

            x = self._inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, name='block_17')

            x = self._conv_block(x, 1280, 3, strides=(1, 1), name='cov_1')

            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = tf.layers.Dense(self.output_nums, activation='softmax', name='predictions')(x)
        return x