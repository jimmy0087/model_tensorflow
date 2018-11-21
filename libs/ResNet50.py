from base_model import *
import tensorflow as tf

class ResNet50(BASE_MODEL):
    def __init__(self,num_classes=10,trainable = True):
        BASE_MODEL.__init__(self,num_classes=10,trainable = True)

    def identity_block(self,input_tensor,kernel_size,filters,stage,block="",branch=""):
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        with tf.variable_scope(branch) as scope:
            x = tf.layers.Conv2D(filters1, (1, 1),
                              kernel_initializer='he_normal',
                              name=conv_name_base + '2a')(input_tensor)

            x = tf.layers.BatchNormalization(name=bn_name_base+"2a")(x)
            x = tf.nn.leaky_relu(x,name='relu_2a')

            x = tf.layers.Conv2D(filters2, kernel_size,
                              padding='same',
                              kernel_initializer='he_normal',
                              name=conv_name_base + '2b')(x)
            x = tf.layers.BatchNormalization(name=bn_name_base + '2b')(x)
            x = tf.nn.leaky_relu(x,name='relu_2b')

            x = tf.layers.Conv2D(filters3, (1,1),
                              padding='same',
                              kernel_initializer='he_normal',
                              name=conv_name_base + '2c')(x)
            x = tf.layers.BatchNormalization(name=bn_name_base + '2c')(x)
            x =  tf.keras.layers.add([x,input_tensor])
            x = tf.nn.leaky_relu(x,name='relu_2c')

        return x

    def conv_block(self,input_tensor,kernel_size,filters,stage,strides=(2, 2),block="",branch=""):
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        with tf.variable_scope(branch) as scope:
            x = tf.layers.Conv2D(filters1, (1, 1),strides=strides,
                              kernel_initializer='he_normal',
                              name=conv_name_base + '2a')(input_tensor)

            x = tf.layers.BatchNormalization(name=bn_name_base+"2a")(x)
            x = tf.nn.leaky_relu(x,name='relu_2a')

            x = tf.layers.Conv2D(filters2, kernel_size,
                              padding='same',
                              kernel_initializer='he_normal',
                              name=conv_name_base + '2b')(x)
            x = tf.layers.BatchNormalization(name=bn_name_base + '2b')(x)
            x = tf.nn.leaky_relu(x,name='relu_2b')

            x = tf.layers.Conv2D(filters3, (1,1),
                              padding='same',
                              kernel_initializer='he_normal',
                              name=conv_name_base + '2c')(x)
            x = tf.layers.BatchNormalization(name=bn_name_base + '2c')(x)

            shortcut = tf.layers.Conv2D(filters3, (1, 1), strides=strides,
                                     kernel_initializer='he_normal',
                                     name=conv_name_base + '1')(input_tensor)
            shortcut = tf.layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

            x =  tf.keras.layers.add([x,shortcut])
            x = tf.nn.leaky_relu(x,name='relu_2c')

        return x

    def inference(self,input):
        x = input
        with tf.variable_scope("ResNet") as scope:
            x = tf.layers.Conv2D(64, (7, 7),
                              strides=(2, 2),
                              padding='valid',
                              kernel_initializer='he_normal',
                              name='conv_1')(x)

            x = tf.layers.BatchNormalization(name='bn_conv_1')(x)
            x = tf.nn.leaky_relu(x,name = 'relu_1')

            x = tf.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

            x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),branch="branch_0")
            x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b',branch="branch_0")
            x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c',branch="branch_0")

            x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a',branch="branch_1")
            x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b',branch="branch_1")
            x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c',branch="branch_1")
            x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d',branch="branch_1")

            x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a',branch="branch_2")
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b',branch="branch_2")
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c',branch="branch_2")
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d',branch="branch_2")
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e',branch="branch_2")
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f',branch="branch_2")

            x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a',branch="branch_3")
            x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b',branch="branch_3")
            x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c',branch="branch_3")

            x = tf.layers.AveragePooling2D(pool_size=3,strides=(1,1),name='avg_pool')(x)
            x = tf.layers.Flatten()(x)
            x = tf.layers.Dense(self.output_nums, activation='softmax', name='fc1000')(x)
        return x