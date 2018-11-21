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



class SEInceptionV4(BASE_MODEL):
    def __init__(self,num_classes=10,trainable = True):
        self.num_classes = num_classes
        BASE_MODEL.__init__(self,num_classes=10,trainable = True)

    def Stem(self, input_tensor, scope):
        with tf.name_scope(scope) :
            x = tf.layers.Conv2D( filters = 32, kernel_size = (3, 3), strides=(2,2), padding='valid', kernel_initializer='he_normal', name=scope + '_conv1')(input_tensor)
            x = tf.layers.Conv2D( filters = 32, kernel_size = (3, 3), padding='valid', kernel_initializer='he_normal', name=scope+'_conv2')(x)
            block_1 = tf.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',name=scope+'_conv3')(x)

            split_max_x = tf.layers.MaxPooling2D(pool_size=(3,3),strides=2,padding='valid')(block_1)
            split_conv_x = tf.layers.Conv2D(filters=96, kernel_size=(3,3), strides=(2,2), padding='VALID', kernel_initializer='he_normal',name=scope+'_split_conv1')(block_1)
            x = tf.concat([split_max_x,split_conv_x], axis=3)

            split_conv_x1 = tf.layers.Conv2D(filters=64, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv2')(x)
            split_conv_x1 = tf.layers.Conv2D(filters=96, kernel_size=(3,3), padding='VALID', kernel_initializer='he_normal',name=scope+'_split_conv3')(split_conv_x1)

            split_conv_x2 = tf.layers.Conv2D( filters=64, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv4')(x)
            split_conv_x2 = tf.layers.Conv2D( filters=64, kernel_size=(7,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv5')(split_conv_x2)
            split_conv_x2 = tf.layers.Conv2D( filters=64, kernel_size=(1,7), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv6')(split_conv_x2)
            split_conv_x2 = tf.layers.Conv2D( filters=96, kernel_size=(3,3), padding='VALID', kernel_initializer='he_normal',name=scope+'_split_conv7')(split_conv_x2)

            x = tf.concat([split_conv_x1,split_conv_x2], axis=3)

            split_conv_x = tf.layers.Conv2D( filters=192, kernel_size=(3,3), strides=2, padding='VALID', kernel_initializer='he_normal',name=scope+'_split_conv8')(x)
            split_max_x = tf.layers.MaxPooling2D(pool_size=(3,3),strides=2,padding='valid')(x)

            x = tf.concat([split_conv_x, split_max_x], axis=3)

            x = tf.layers.BatchNormalization(name=scope+'_batch1')(x)
            x = tf.nn.leaky_relu(x)
            return x

    def Inception_A(self, input_tensor, scope):
        with tf.name_scope(scope) :
            split_conv_x1 = tf.layers.AveragePooling2D(pool_size=(3,3),strides=1,padding='same')(input_tensor)
            split_conv_x1 = tf.layers.Conv2D(filters=96, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv1')(split_conv_x1)

            split_conv_x2 = tf.layers.Conv2D(filters=96, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv2')(input_tensor)

            split_conv_x3 = tf.layers.Conv2D( filters=64, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv3')(input_tensor)
            split_conv_x3 = tf.layers.Conv2D( filters=96, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv4')(split_conv_x3)

            split_conv_x4 = tf.layers.Conv2D( filters=64, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv5')(input_tensor)
            split_conv_x4 = tf.layers.Conv2D( filters=96, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv6')(split_conv_x4)
            split_conv_x4 = tf.layers.Conv2D( filters=96, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv7')(split_conv_x4)

            x = tf.concat([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4], axis=3)

            x = tf.layers.BatchNormalization(name=scope+'_batch1')(x)
            x = tf.nn.leaky_relu(x)

            return x

    def Inception_B(self, input_tensor, scope):
        with tf.name_scope(scope) :

            split_conv_x1 = tf.layers.AveragePooling2D(pool_size=(3,3),strides=1,padding='same')(input_tensor)
            split_conv_x1 = tf.layers.Conv2D( filters=128, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv1')(split_conv_x1)

            split_conv_x2 = tf.layers.Conv2D( filters=384, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv2')(input_tensor)

            split_conv_x3 = tf.layers.Conv2D( filters=192, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv3')(input_tensor)
            split_conv_x3 = tf.layers.Conv2D( filters=224, kernel_size=(1,7), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv4')(split_conv_x3)
            split_conv_x3 = tf.layers.Conv2D( filters=256, kernel_size=(1,7), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv5')(split_conv_x3)

            split_conv_x4 = tf.layers.Conv2D( filters=192, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv6')(input_tensor)
            split_conv_x4 = tf.layers.Conv2D( filters=192, kernel_size=(1,7), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv7')(split_conv_x4)
            split_conv_x4 = tf.layers.Conv2D( filters=224, kernel_size=[7,1], padding='same', kernel_initializer='he_normal',name=scope+'_split_conv8')(split_conv_x4)
            split_conv_x4 = tf.layers.Conv2D( filters=224, kernel_size=(1,7), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv9')(split_conv_x4)
            split_conv_x4 = tf.layers.Conv2D( filters=256, kernel_size=(7,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv10')(split_conv_x4)

            x = tf.concat([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4], axis=3)

            x = tf.layers.BatchNormalization(name=scope+'_batch1')(x)
            x = tf.nn.leaky_relu(x)

            return x

    def Inception_C(self, input_tensor, scope):
        with tf.name_scope(scope) :
            split_conv_x1 = tf.layers.AveragePooling2D(pool_size=(3,3),strides=1,padding='same')(input_tensor)
            split_conv_x1 = tf.layers.Conv2D( filters=256, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv1')(split_conv_x1)

            split_conv_x2 = tf.layers.Conv2D( filters=256, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv2')(input_tensor)

            split_conv_x3 = tf.layers.Conv2D( filters=384, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv3')(input_tensor)
            split_conv_x3_1 = tf.layers.Conv2D( filters=256, kernel_size=(1,3), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv4')(split_conv_x3)
            split_conv_x3_2 = tf.layers.Conv2D( filters=256, kernel_size=(3,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv5')(split_conv_x3)

            split_conv_x4 = tf.layers.Conv2D( filters=384, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv6')(input_tensor)
            split_conv_x4 = tf.layers.Conv2D( filters=448, kernel_size=(1,3), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv7')(split_conv_x4)
            split_conv_x4 = tf.layers.Conv2D( filters=512, kernel_size=(3,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv8')(split_conv_x4)
            split_conv_x4_1 = tf.layers.Conv2D( filters=256, kernel_size=(3,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv9')(split_conv_x4)
            split_conv_x4_2 = tf.layers.Conv2D( filters=256, kernel_size=(1,3), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv10')(split_conv_x4)

            x = tf.concat([split_conv_x1, split_conv_x2, split_conv_x3_1, split_conv_x3_2, split_conv_x4_1, split_conv_x4_2], axis=3)

            x = tf.layers.BatchNormalization(name=scope+'_batch1')(x)
            x = tf.nn.leaky_relu(x)

            return x

    def Reduction_A(self, input_tensor, scope):
        with tf.name_scope(scope) :
            k = 256
            l = 256
            m = 384
            n = 384

            split_max_x = tf.layers.MaxPooling2D(pool_size=(3,3),strides=2,padding='valid')(input_tensor)

            split_conv_x1 = tf.layers.Conv2D(filters=n, kernel_size=(3,3), strides=2, padding='VALID', kernel_initializer='he_normal',name=scope+'_split_conv1')(input_tensor)

            split_conv_x2 = tf.layers.Conv2D( filters=k, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv2')(input_tensor)
            split_conv_x2 = tf.layers.Conv2D( filters=l, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv3')(split_conv_x2)
            split_conv_x2 = tf.layers.Conv2D( filters=m, kernel_size=(3,3), strides=2, padding='VALID', kernel_initializer='he_normal',name=scope+'_split_conv4')(split_conv_x2)

            x = tf.concat([split_max_x, split_conv_x1, split_conv_x2], axis=3)

            x = tf.layers.BatchNormalization(name=scope + '_batch1')(x)
            x = tf.nn.leaky_relu(x)

            return x

    def Reduction_B(self, input_tensor, scope):
        with tf.name_scope(scope) :
            split_max_x = tf.layers.MaxPooling2D(pool_size=(3,3),strides=2,padding='valid')(input_tensor)

            split_conv_x1 = tf.layers.Conv2D( filters=256, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv1')(input_tensor)
            split_conv_x1 = tf.layers.Conv2D( filters=384, kernel_size=(3,3), strides=2, padding='VALID', kernel_initializer='he_normal',name=scope+'_split_conv2')(split_conv_x1)

            split_conv_x2 = tf.layers.Conv2D( filters=256, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv3')(input_tensor)
            split_conv_x2 = tf.layers.Conv2D( filters=288, kernel_size=(3,3), strides=2, padding='VALID', kernel_initializer='he_normal',name=scope+'_split_conv4')(split_conv_x2)

            split_conv_x3 = tf.layers.Conv2D( filters=256, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv5')(input_tensor)
            split_conv_x3 = tf.layers.Conv2D( filters=288, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',name=scope+'_split_conv6')(split_conv_x3)
            split_conv_x3 = tf.layers.Conv2D( filters=320, kernel_size=(3,3), strides=2, padding='VALID', kernel_initializer='he_normal',name=scope+'_split_conv7')(split_conv_x3)

            x = tf.concat([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3], axis=3)

            x = tf.layers.BatchNormalization(name=scope + '_batch1')(x)
            x = tf.nn.leaky_relu(x)

            return x

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :
            squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_x)

            excitation = tf.layers.Dense(units=out_dim / ratio, name=layer_name+'_fully_connected1')(squeeze)
            excitation = tf.nn.leaky_relu(excitation)
            excitation = tf.layers.Dense(units=out_dim, name=layer_name+'_fully_connected2')(excitation)
            excitation = tf.nn.sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])

            scale = input_x * excitation

            return scale

    def inference(self, input_x):
        input_x = tf.pad(input_x, [[0, 0], [32, 32], [32, 32], [0, 0]])
        # size 32 -> 96
        # only cifar10 architecture

        x = self.Stem(input_x, scope='stem')

        for i in range(4) :
            x = self.Inception_A(x, scope='Inception_A'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A'+str(i))

        x = self.Reduction_A(x, scope='Reduction_A')

        for i in range(7)  :
            x = self.Inception_B(x, scope='Inception_B'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B'+str(i))

        x = self.Reduction_B(x, scope='Reduction_B')

        for i in range(3) :
            x = self.Inception_C(x, scope='Inception_C'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C'+str(i))

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.layers.Dropout(rate=0.2)(x)
        x = tf.layers.Flatten()(x)

        x = tf.layers.Dense(self.num_classes, activation='softmax', name='final_fully_connected')(x)
        return x