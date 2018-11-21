from base_model import *
import tensorflow as tf
class Xception(BASE_MODEL):
    def __init__(self,num_classes=10,trainable = True):
        BASE_MODEL.__init__(self,num_classes=10,trainable = True)

    def conv2d_bn(self,input_tensor,filters,kernel_size,padding='same',strides=(1, 1),use_bias=False,activation='relu',name=""):
        with tf.variable_scope(name) as scope:
            x = tf.layers.Conv2D(
                filters, kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                name=name + '_res')(input_tensor)
            if not use_bias:
                x = tf.layers.BatchNormalization(ascale=False, name=name + '_bn')(x)
            if activation is not None:
                x = tf.nn.relu(x, name='relu')
        return x

    def speconv2d_bn(self,input_tensor,filters,kernel_size,padding='same',strides=(1, 1),use_bias=False,activation='relu',name=""):
        with tf.variable_scope(name) as scope:
            x = tf.nn.relu(input_tensor, name='relu') if activation is not None else input_tensor
            x = tf.layers.SeparableConv2D(
                filters, kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                name=name + '_res')(x)
            if not use_bias:
                x = tf.layers.BatchNormalization(ascale=False, name=name + '_bn')(x)

        return x

    def block_a(self,input_tensor,filters,includ_top_act=True,name=""):
        x = input_tensor
        with tf.variable_scope(name) as scope:
            residual = self.conv2d_bn(x,filters[0],(1,1),strides=(2, 2),activation=None,name='conv1')
            x = self.speconv2d_bn(x,filters[1],(3,3),name='sepconv1')
            x = self.speconv2d_bn(x, filters[2], (3, 3), name='sepconv2')
            x = tf.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                    padding='same',
                                    name='pool')(x)
            x = tf.add_n([x, residual])
        return x

    def block_a_top(self,input_tensor,filters,name=""):
        x = input_tensor
        with tf.variable_scope(name) as scope:
            residual = self.conv2d_bn(x,filters[0],(1,1),strides=(2, 2),activation=None,name='conv1')
            x = self.speconv2d_bn(x,filters[1],(3,3),activation=None,name='sepconv1')
            x = self.speconv2d_bn(x, filters[2], (3, 3), name='sepconv2')
            x = tf.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                    padding='same',
                                    name='pool')(x)
            x = tf.add_n([x, residual])
        return x

    def block_c(self,input_tensor,filters,name=""):
        x = input_tensor
        with tf.variable_scope(name) as scope:
            residual = input_tensor
            x = self.speconv2d_bn(x, filters[0], (3, 3), name='sepconv1')
            x = self.speconv2d_bn(x,filters[1],(3,3),name='sepconv2')
            x = self.speconv2d_bn(x, filters[2], (3, 3), name='sepconv3')
            x = tf.add_n([x, residual])
        return x

    def inference(self,input):
        x = input
        with tf.variable_scope("Xception",reuse=tf.AUTO_REUSE) as scope:
            x = self.conv2d_bn(x, 32, (3, 3), strides=(2, 2), name='block1_conv1')
            x = self.conv2d_bn(x, 64, (3, 3), strides=(2, 2), name='block1_conv2')

            x = self.block_a_top(x,[128,128,128],name='block2')
            x = self.block_a(x, [256, 256, 256], name='block3')
            x = self.block_a(x, [728, 728, 728], name='block4')

            for i in range(8):
                name = 'block'+str(i+5)
                x = self.block_c(x,[728, 728, 728],name=name)

            x = self.block_a(x, [1024, 728, 1024], name='block13')
            x = tf.layers.SeparableConv2D(1536, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name='block14_sepconv1')(x)
            x = tf.layers.BatchNormalization(name='block14_sepconv1_bn')(x)
            x = tf.nn.relu(x,name='block14_sepconv1_act')

            x = tf.layers.SeparableConv2D(2048, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name='block14_sepconv2')(x)
            x = tf.layers.BatchNormalization(name='block14_sepconv2_bn')(x)
            x = tf.nn.relu(x, name='block14_sepconv2_act')

            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = tf.layers.Dense(self.output_nums, activation='softmax', name='predictions')(x)
        return x