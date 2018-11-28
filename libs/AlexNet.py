from base_model import *

class AlexNet(BASE_MODEL):
    def __init__(self,num_classes=10,trainable = True):
        BASE_MODEL.__init__(self,num_classes=num_classes,trainable = True)
    def inference(self,input):
        x= input
        self.layer_add(x, name='input')
        with tf.variable_scope("AlexNet",reuse=tf.AUTO_REUSE) as scope:
            x = tf.layers.Conv2D(filters = 96,  kernel_size=3, strides=(1,1), padding='same',
                             kernel_initializer='glorot_uniform',
                             trainable=True,
                             name='cov_1')(x)
            self.layer_add(x,name='cov_1')
            x = tf.layers.BatchNormalization(name='bn_conv_1')(x)
            x = tf.nn.leaky_relu(x, name='relu_1')
            x = tf.layers.MaxPooling2D(pool_size = 3, strides = (2,2), padding = 'same',name='maxpool_1')(x)
            self.layer_add(x, name='maxpool_1')

            x = tf.layers.Conv2D(filters = 256,  kernel_size=5, strides=(1,1), padding='same',
                             kernel_initializer='glorot_uniform',
                             trainable=True,
                             name='cov_2')(x)
            self.layer_add(x, name='cov_2')
            x = tf.layers.BatchNormalization(name='bn_conv_2')(x)
            x = tf.nn.leaky_relu(x, name='relu_2')
            x = tf.layers.MaxPooling2D(pool_size = 3, strides = (2,2), padding = 'same',name='maxpool_2')(x)
            self.layer_add(x, name='maxpool_2')

            x = tf.layers.Conv2D(filters=384, kernel_size=3, strides=(1, 1), padding='same',
                             kernel_initializer='glorot_uniform',
                             trainable=True,
                             name='cov_3')(x)
            self.layer_add(x, name='cov_3')

            x = tf.layers.Conv2D(filters=384, kernel_size=3, strides=(1, 1), padding='same',
                             kernel_initializer='glorot_uniform',
                             trainable=True,
                             name='cov_4')(x)
            self.layer_add(x, name='cov_4')

            x = tf.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
                             kernel_initializer='glorot_uniform',
                             trainable=True,
                             name='cov_5')(x)
            self.layer_add(x, name='cov_5')

            x = tf.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='same', name='maxpool_5')(x)
            self.layer_add(x, name='maxpool_5')

            x = tf.layers.Flatten()(x)

            x = tf.layers.Dense(4096, activation=tf.nn.relu, name='fc_6')(x)
            self.layer_add(x, name='fc_6')
            x = tf.layers.Dropout(0.5, name='do_6')(x)

            x = tf.layers.Dense(4096, activation=tf.nn.relu, name='fc_7')(x)
            self.layer_add(x, name='fc_7')
            x = tf.layers.Dropout(0.5, name='do_7')(x)

            x = tf.layers.Dense(self.output_nums, activation='softmax', name='soft_8')(x)
            self.layer_add(x, name='soft_8')
        return x


