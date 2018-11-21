from base_model import *
import tensorflow as tf
class Vgg16Net(BASE_MODEL):
    def __init__(self,num_classes=10,trainable = True):
        BASE_MODEL.__init__(self,num_classes=10,trainable = True)

    def vgg_block(self,input_tensor,filters,cov_num=1,name="block_"):
        x = input_tensor
        with tf.variable_scope(name) as scope:
            for i in range(cov_num):
                cur_name = "cov_"+str(i)
                x = tf.layers.Conv2D(filters = filters,  kernel_size=3, strides=(1,1), padding='same',
                             kernel_initializer=layers_lib.xavier_initializer(),
                             trainable=True,
                             name=cur_name)(x)
                self.layer_add(x, name=x.name)
        return x

    def inference(self,input):
        x = input
        self.layer_add(x, name="input")
        with tf.variable_scope("VggNet") as scope:
            x = self.vgg_block(x,filters = 64,cov_num = 2,name="block_1")
            x = tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='same', name='maxpool_1')(x)
            self.layer_add(x, name="maxpool_1")

            x = self.vgg_block(x, filters=128, cov_num=2, name="block_2")
            x = tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='same', name='maxpool_2')(x)
            self.layer_add(x, name="maxpool_2")

            x = self.vgg_block(x, filters=256, cov_num=3, name="block_3")
            x = tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='same', name='maxpool_3')(x)
            self.layer_add(x, name="maxpool_3")

            x = self.vgg_block(x, filters=512, cov_num=3, name="block_4")
            x = tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='same', name='maxpool_4')(x)
            self.layer_add(x, name="maxpool_4")

            x = self.vgg_block(x, filters=512, cov_num=3, name="block_5")
            x = tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='same', name='maxpool_5')(x)
            self.layer_add(x, name="maxpool_5")

            x = tf.layers.Flatten()(x)

            x = tf.layers.Dense(4096, activation=tf.nn.relu, name='fc_6')(x)
            self.layer_add(x, name="fc_6")
            #x = tf.layers.Dropout(0.5, name='do_6')(x)

            x = tf.layers.Dense(4096, activation=tf.nn.relu, name='fc_7')(x)
            self.layer_add(x, name="fc_7")
            #x = tf.layers.Dropout(0.5, name='do_7')(x)

            x = tf.layers.Dense(1000, activation=tf.nn.relu, name='fc_8')(x)
            self.layer_add(x, name="fc_8")
            #x = tf.layers.Dropout(0.5, name='do_8')(x)

            x = tf.layers.Dense(self.output_nums, activation='softmax', name='soft_9')(x)
            self.layer_add(x, name="soft_9")
        return x