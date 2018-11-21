from base_model import *
import tensorflow as tf
class InceptionV3(BASE_MODEL):
    def __init__(self,num_classes=10,trainable = True):
        BASE_MODEL.__init__(self,num_classes=10,trainable = True)

    def conv2d_bn(self,input_tensor,filters,kernel_size,padding='same',strides=(1, 1),name=""):
        with tf.variable_scope(name) as scope:
            x = tf.layers.Conv2D(
                filters, kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False,
                name=name + '_res')(input_tensor)
            x = tf.layers.BatchNormalization(ascale=False, name=name + '_bn')(x)
            x = tf.nn.relu(x, name='relu')

        return x

    def block_a(self,x,filters,name):
        with tf.variable_scope(name) as scope:
            branch1x1 = self.conv2d_bn(x, filters[0], (1, 1))

            branch5x5 = self.conv2d_bn(x, filters[1], (1, 1))
            branch5x5 = self.conv2d_bn(branch5x5, filters[2], (5, 5))

            branch3x3dbl = self.conv2d_bn(x, filters[3], (1, 1))
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, filters[4], (3, 3))
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, filters[5], (3, 3))

            branch_pool = tf.layers.AveragePooling2D((3, 3),strides=(1, 1),padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, filters[6], (1, 1))

            x = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                axis=3,name='mixed')
        return x

    def block_a_sp(self,x,filters,name):
        with tf.variable_scope(name) as scope:
            branch3x3 = self.conv2d_bn(x, filters[0], (3, 3),strides=(2,2))

            branch3x3dbl = self.conv2d_bn(x, filters[1], (1, 1))
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, filters[2], (3, 3))
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, filters[3], (3, 3),strides=(2,2))

            branch_pool = tf.layers.AveragePooling2D((3, 3),strides=(2, 2),padding='same')(x)

            x = tf.concat([branch3x3, branch3x3dbl, branch_pool],
                axis=3,name='mixed')
        return x

    def block_b(self, x,filters ,name):
        with tf.variable_scope(name) as scope:
            branch1x1 = self.conv2d_bn(x, filters[0], (1, 1))

            branch7x7 = self.conv2d_bn(x, filters[1], (1, 1))
            branch7x7 = self.conv2d_bn(branch7x7, filters[2], (1, 7))
            branch7x7 = self.conv2d_bn(branch7x7, filters[3], (7, 1))

            branch7x7dbl = self.conv2d_bn(x, filters[4], (1, 1))
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, filters[5], (7, 1))
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, filters[6], (1, 7))
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, filters[7], (7, 1))
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, filters[8], (1, 7))

            branch_pool = tf.layers.AveragePooling2D((3, 3),strides=(1, 1),padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, filters[9], (1, 1))
            x = tf.concat(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=3,
                name='mixed')
        return x

    def block_b_sp(self, x,filters ,name):
        with tf.variable_scope(name) as scope:
            branch3x3 = self.conv2d_bn(x, filters[0], (1, 1))
            branch3x3 = self.conv2d_bn(branch3x3, filters[1], (3, 3),strides=(2, 2), padding='valid')

            branch7x7x3 = self.conv2d_bn(x, filters[2], (1, 1))
            branch7x7x3 = self.conv2d_bn(branch7x7x3, filters[3], (1, 7))
            branch7x7x3 = self.conv2d_bn(branch7x7x3, filters[4], (7, 1))
            branch7x7x3 = self.conv2d_bn(branch7x7x3, filters[5], (3, 3), strides=(2, 2), padding='valid')

            branch_pool = tf.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

            x = tf.concat([branch3x3, branch7x7x3, branch_pool],
                axis=3,
                name='mixed8')
        return x

    def block_c(self, x, filters,name):
        with tf.variable_scope(name) as scope:
            branch1x1 = self.conv2d_bn(x, filters[0], (1, 1))

            branch3x3 = self.conv2d_bn(x, filters[1], (1, 1))
            branch3x3_1 = self.conv2d_bn(branch3x3, filters[2], (1, 3))
            branch3x3_2 = self.conv2d_bn(branch3x3, filters[3], (3, 1))
            branch3x3 = tf.concat([branch3x3_1, branch3x3_2],axis=3,name='mixed_branch0')

            branch3x3dbl = self.conv2d_bn(x, filters[4], (1, 1))
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, filters[5], (3, 3))
            branch3x3dbl_1 = self.conv2d_bn(branch3x3dbl, filters[6], (1, 3))
            branch3x3dbl_2 = self.conv2d_bn(branch3x3dbl, filters[7], (3, 1))
            branch3x3dbl = tf.concat([branch3x3dbl_1, branch3x3dbl_2],axis=3,name='mixed_branch1')

            branch_pool = tf.layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, filters[8], (1, 1))

            x = tf.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool],axis=3,name='mixed')
        return x

    def inference(self,input):
        x = input
        with tf.variable_scope("InceptionV3") as scope:
            x = self.conv2d_bn(x, 32, (3, 3), strides=(2, 2), padding='valid',name="conv_1")
            x = self.conv2d_bn(x, 32, (3, 3), padding='valid',name="conv_2")
            x = self.conv2d_bn(x, 64, (3, 3),name="conv_3")
            x = tf.layers.MaxPooling2D((3, 3), strides=(2, 2),name="maxpool_4")(x)

            x = self.conv2d_bn(x, 80, (1, 1), padding='valid',name="conv_5")
            x = self.conv2d_bn(x, 192, (3, 3), padding='valid',name="conv_6")
            x = tf.layers.MaxPooling2D((3, 3), strides=(2, 2),name="maxpool_7")(x)

            x = self.block_a(x,[64,48,64,64,96,96,32],name = 'block_8')
            x = self.block_a(x, [64, 48, 64, 64, 96, 96, 64], name='block_9')
            x = self.block_a(x, [64, 48, 64, 64, 96, 96, 64], name='block_10')
            x = self.block_a_sp(x, [384, 64, 96, 96], name='block_11')

            x = self.block_b(x, [192,128,128,192,128,128,128,128,192,192], name='block_12')
            x = self.block_b(x, [192, 160, 160, 192, 160, 160, 160, 160, 192, 192], name='block_13')
            x = self.block_b(x, [192, 192, 192, 192, 192, 192, 192, 192, 192, 192], name='block_14')
            x = self.block_b_sp(x, [192, 320, 192, 192, 192,192], name='block_15')

            x = self.block_c(x, [320,384,384,384,448,384,384,384,192], name='block_16')
            x = self.block_c(x, [320, 384, 384, 384, 448, 384, 384, 384, 192], name='block_17')

            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_18')(x)
            x = tf.layers.Dense(self.output_nums, activation='softmax', name='predictions')(x)

        return x