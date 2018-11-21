import tensorflow as tf
from base_model import *

cardinality = 8
blocks = 3
reduction_ratio = 4

class SEResNeXt(BASE_MODEL):
    def __init__(self,num_classes=10,trainable = True):
        self.num_classes = num_classes
        BASE_MODEL.__init__(self,num_classes=10,trainable = True)

    def first_layer(self,input_tensor,scope):
        with tf.variable_scope(scope):
            x = tf.layers.Conv2D(filters = 64,kernel_size = (3, 3),strides=(1,1),padding='same',
                                 use_bias=False, kernel_initializer='he_normal',name=scope + '_conv1')(input_tensor)
            x = tf.layers.BatchNormalization(name=scope+"_bn1")(x)
            x = tf.nn.leaky_relu(x)
            return x

    def transform_layer(self, input_tensor, stride, scope):
        with tf.variable_scope(scope):
            x = tf.layers.Conv2D(filters = 64, kernel_size = [1, 1], strides=(1,1),padding='same',
                                 use_bias=False, kernel_initializer='he_normal',name = scope + '_conv1')(input_tensor)
            x = tf.layers.BatchNormalization(name = scope + '_bn1')(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.Conv2D(filters = 64, kernel_size = [3, 3], strides=stride,padding='same',
                                 use_bias=False, kernel_initializer='he_normal',name = scope + '_conv2')(x)
            x = tf.layers.BatchNormalization(name = scope + '_bn2')(x)
            x = tf.nn.leaky_relu(x)
            return x

    def transition_layer(self, input_tensor, out_dim, scope):
        with tf.variable_scope(scope):
            x = tf.layers.Conv2D(filters = out_dim, kernel_size = [1, 1], strides=(1,1),padding='same',
                                 use_bias=False, kernel_initializer='he_normal',name=scope + '_conv1')(input_tensor)
            x = tf.layers.BatchNormalization(name=scope + '_batch1')(x)
            # x = tf.nn.leaky_relu(x)
            return x

    def split_layer(self, input_tensor, stride, layer_name):
        with tf.variable_scope(layer_name):
            layers_split = list()
            for i in range(cardinality):
                splits = self.transform_layer(input_tensor, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)
            return tf.concat(layers_split,axis=3)

    def squeeze_excitation_layer(self, input_tensor, out_dim, ratio, layer_name):
        with tf.variable_scope(layer_name):
            squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)

            excitation = tf.layers.Dense(units=out_dim / ratio, use_bias=False ,name=layer_name + '_fully_connected1')(squeeze)
            excitation = tf.nn.leaky_relu(excitation)
            excitation = tf.layers.Dense(units=out_dim, use_bias=False ,name=layer_name + '_fully_connected2')(excitation)
            excitation = tf.nn.sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_tensor * excitation
            return scale

    def residual_layer(self, input_tensor, out_dim, layer_num, res_block=blocks):
        # split + transform(bottleneck) + transition + merge
        # input_dim = input_x.get_shape().as_list()[-1]
        for i in range(res_block):
            input_dim = int(input_tensor.shape[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            x = self.split_layer(input_tensor, stride=stride, layer_name='split_layer_' + layer_num + '_' + str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_' + layer_num + '_' + str(i))
            x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio,layer_name='squeeze_layer_' + layer_num + '_' + str(i))

            if flag is True:
                pad_input_x = tf.layers.AveragePooling2D(pool_size=2,strides=(2,2),padding='same',name='avg_pool')(input_tensor)
                pad_input_x = tf.pad(pad_input_x,[[0, 0], [0, 0], [0, 0], [channel, channel]])  # [?, height, width, channel]
            else:
                pad_input_x = input_tensor

            x = tf.nn.leaky_relu(x + pad_input_x)

        return x

    def inference(self, input_x):
        with tf.variable_scope("SEResNeXt",reuse=tf.AUTO_REUSE) as scope:
            input_x = self.first_layer(input_x, scope='first_layer')

            x = self.residual_layer(input_x, out_dim=64, layer_num='1')
            x = self.residual_layer(x, out_dim=128, layer_num='2')
            x = self.residual_layer(x, out_dim=256, layer_num='3')

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.layers.Flatten()(x)

            x = tf.layers.Dense(self.num_classes,activation='softmax',name='final_fully_connected')(x)
        return x
