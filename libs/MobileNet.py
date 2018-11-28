from base_model import *
import tensorflow as tf
class MobileNet(BASE_MODEL):
    def __init__(self,num_classes=10,trainable = True):
        BASE_MODEL.__init__(self,num_classes=num_classes,trainable = True)

    def _conv_block(self,input_tensor, filter,alpha,kernel = (3, 3),strides = (1, 1), name=''):
        x = input_tensor
        filter_ = int(filter * alpha)
        with tf.variable_scope(name) as scope:
            x = tf.layers.Conv2D(filter_, kernel,
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
            x = tf.nn.relu(x, 'conv_pw_relu')
        return x


    def inference(self,input):
        x = input
        alpha = 1.0
        depth_multiplier = 1
        with tf.variable_scope("MobileNet",reuse=tf.AUTO_REUSE) as scope:
                x = self._conv_block(x, 32, alpha, strides=(2, 2), name='conv_1')
                x = self._depthwise_conv_block(x, 64, alpha, depth_multiplier, name='block_1')
                x = self._depthwise_conv_block(x, 128, alpha, depth_multiplier,strides=(2, 2), name='block_2')
                x = self._depthwise_conv_block(x, 128, alpha, depth_multiplier, name='block_3')

                x = self._depthwise_conv_block(x, 256, alpha, depth_multiplier,strides=(2, 2), name='block_4')
                x = self._depthwise_conv_block(x, 256, alpha, depth_multiplier, name='block_5')

                x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier,strides=(2, 2), name='block_6')
                x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, name='block_7')
                x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, name='block_8')
                x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, name='block_9')
                x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, name='block_10')
                x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, name='block_11')

                x = self._depthwise_conv_block(x, 1024, alpha, depth_multiplier,strides=(2, 2), name='block_12')
                x = self._depthwise_conv_block(x, 1024, alpha, depth_multiplier, name='block_13')

                x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
                x = tf.layers.Dense(self.output_nums, activation='softmax', name='predictions')(x)
        return x