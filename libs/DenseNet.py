from base_model import *
import tensorflow as tf
class DenseNet(BASE_MODEL):
    def __init__(self,num_classes=10,trainable = True):
        BASE_MODEL.__init__(self,num_classes=10,trainable = True)

    def conv_block(self,input_tensor, growth_rate, name):
        x = input_tensor
        with tf.variable_scope(name) as scope:
            bn_axis = 3
            x = tf.layers.BatchNormalization(axis=bn_axis,
                                           epsilon=1.001e-5,
                                           name=name + '_0_bn')(x)
            x = tf.nn.relu(x, name='_0_relu')
            x = tf.layers.Conv2D(4 * growth_rate, 1,
                               use_bias=False,
                               name=name + '_1_conv')(x)
            x = tf.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                           name=name + '_1_bn')(x)
            x = tf.nn.relu(x, name='_1_relu')
            #x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
            x = tf.layers.Conv2D(growth_rate, 3,
                               padding='same',
                               use_bias=False,
                               name=name + '_2_conv')(x)
            x = tf.concat([x, input_tensor],axis=bn_axis, name=name + '_concat')
        return x

    def dense_block(self,input_tensor, blocks, name):
        x = input_tensor
        with tf.variable_scope(name) as scope:
            for i in range(blocks):
                x = self.conv_block(x, 32, name=name + '_block' + str(i + 1))
        return x

    def transition_block(self,input_tensor, reduction, name):
        x = input_tensor
        with tf.variable_scope(name) as scope:
            bn_axis = 3
            x = tf.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                          name=name + '_bn')(x)
            x = tf.nn.relu(x, name='_1_relu')
            x = tf.layers.Conv2D(int(input_tensor.shape[bn_axis].value * reduction), (1,1),
                              use_bias=False,
                              name=name + '_conv')(x)
            x = tf.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    def inference(self,input):
        x = input
        blocks = [6, 12, 24, 16] #densenet121
        with tf.variable_scope("DenseNet") as scope:
            x = tf.layers.Conv2D(64, 7, (2,2), use_bias=False,padding = 'same' ,name='conv1/conv')(x)
            x = tf.layers.BatchNormalization( epsilon=1.001e-5, name='conv1/bn')(x)
            x = tf.nn.relu(x, name='conv1/relu')
            x = tf.layers.MaxPooling2D(3, strides=2,padding='same' ,name='pool1')(x)

            x = self.dense_block(x, blocks[0], name='conv2')
            x = self.transition_block(x, 0.5, name='pool2')
            x = self.dense_block(x, blocks[1], name='conv3')
            x = self.transition_block(x, 0.5, name='pool3')
            x = self.dense_block(x, blocks[2], name='conv4')
            x = self.transition_block(x, 0.5, name='pool4')
            x = self.dense_block(x, blocks[3], name='conv5')

            x = tf.layers.BatchNormalization(epsilon=1.001e-5, name='bn')(x)
            x = tf.nn.relu(x, name='relu')

            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = tf.layers.Dense(self.output_nums, activation='softmax', name='predictions')(x)
        return x