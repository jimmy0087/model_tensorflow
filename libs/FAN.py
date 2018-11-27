from base_model import *

class FAN56(BASE_MODEL):
    def __init__(self,num_classes=10,trainable = True):
        self.num_classes = num_classes
        BASE_MODEL.__init__(self,num_classes=10,trainable = True)

    def ResidualBlock(self,input_tensor,output_channels,scope,strides=1):
        with tf.variable_scope(scope):
            residual = input_tensor
            input_channels = int(input_tensor.get_shape()[-1])
            x = self.conv_layer( input_tensor , filters=output_channels/4, kernel_size=(1, 1), padding='same',
                                 use_bn = True ,use_bias = False ,name=scope + '_conv1')

            x = self.conv_layer(x, filters=output_channels/4, kernel_size=(3, 3), strides = strides,padding='same',
                                use_bn=True, use_bias=False, name=scope + '_conv2')

            x = self.conv_layer(x, filters=output_channels, kernel_size=(1, 1), padding='same',
                                use_bn=True, use_bias=False, name=scope + '_conv3')

            if input_channels != output_channels or  strides != 1 :
                residual = self.conv_layer(input_tensor, filters=output_channels, kernel_size=(1, 1),strides = strides, padding='same',
                                           use_bn=True, use_bias=False, name=scope + '_conv4')
            x = x + residual
            return x

    def SoftMaskBanch_0(self,input_tensor,output_channels,scope):
        with tf.variable_scope(scope):
            x = tf.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(input_tensor)
            out_softmax1 = self.ResidualBlock(x, output_channels, scope + '_softmax_1')
            out_skip1_connection = self.ResidualBlock(out_softmax1, output_channels, scope + '_skip_1')

            x = tf.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(out_softmax1)
            out_softmax2 = self.ResidualBlock(x, output_channels, scope + '_softmax_2')
            out_skip2_connection = self.ResidualBlock(out_softmax2, output_channels, scope + '_skip_2')

            x = tf.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(out_softmax2)
            out_softmax3 = self.ResidualBlock(x, output_channels, scope + '_softmax_3')

            out_softmax4 = self.ResidualBlock(out_softmax3, output_channels, scope + '_softmax_4')
            x = tf.keras.layers.UpSampling2D()(out_softmax4)
            x = x + out_softmax2 + out_skip2_connection

            out_softmax5 = self.ResidualBlock(x, output_channels, scope + '_softmax_5')
            x = tf.keras.layers.UpSampling2D()(out_softmax5)
            x = x + out_softmax1 + out_skip1_connection

            out_softmax6 = self.ResidualBlock(x, output_channels, scope + '_softmax_6')
            x = tf.keras.layers.UpSampling2D()(out_softmax6)
            x = x + input_tensor

            x = self.conv_layer( x , filters=output_channels, kernel_size=1, padding='same',
                                 use_bn = True ,use_bias = False ,name=scope + '_conv1')

            x = self.conv_layer(x, filters=output_channels, kernel_size=1, padding='same',
                                use_bn=True,activation=False, use_bias=False, name=scope + '_conv2')
            x = tf.nn.sigmoid(x)

            return x

    def SoftMaskBanch_1(self,input_tensor,output_channels,scope):
        with tf.variable_scope(scope):
            x = tf.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(input_tensor)
            out_softmax1 = self.ResidualBlock(x, output_channels, scope + '_softmax_1')
            out_skip1_connection = self.ResidualBlock(out_softmax1, output_channels, scope + '_skip_1')

            x = tf.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(out_softmax1)
            out_softmax2 = self.ResidualBlock(x, output_channels, scope + '_softmax_2')

            out_softmax3 = self.ResidualBlock(out_softmax2, output_channels, scope + '_softmax_3')
            x = tf.keras.layers.UpSampling2D()(out_softmax3)
            x = x + out_softmax1 + out_skip1_connection

            out_softmax4 = self.ResidualBlock(x, output_channels, scope + '_softmax_4')
            x = tf.keras.layers.UpSampling2D()(out_softmax4)
            x = x + input_tensor

            x = self.conv_layer( x , filters=output_channels, kernel_size=1, padding='same',
                                 use_bn = True ,use_bias = False ,name=scope + '_conv1')

            x = self.conv_layer(x, filters=output_channels, kernel_size=1, padding='same',
                                use_bn=True,activation=False, use_bias=False, name=scope + '_conv2')
            x = tf.nn.sigmoid(x)

            return x

    def SoftMaskBanch_2(self,input_tensor,output_channels,scope):
        with tf.variable_scope(scope):
            x = tf.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(input_tensor)
            out_softmax1 = self.ResidualBlock(x, output_channels, scope + '_softmax_1')

            out_softmax2 = self.ResidualBlock(out_softmax1, output_channels, scope + '_softmax_2')
            x = tf.keras.layers.UpSampling2D()(out_softmax2)
            x = x + input_tensor

            x = self.conv_layer( x , filters=output_channels, kernel_size=1, padding='same',
                                 use_bn = True ,use_bias = False ,name=scope + '_conv1')

            x = self.conv_layer(x, filters=output_channels, kernel_size=1, padding='same',
                                use_bn=True,activation=False, use_bias=False, name=scope + '_conv2')
            x = tf.nn.sigmoid(x)

            return x

    def AttentionModule_stage0(self,input_tensor,output_channels,scope):
        with tf.variable_scope(scope):
            out_top = self.ResidualBlock(input_tensor,output_channels,scope + '_res_1')

            x = self.ResidualBlock(out_top,output_channels,scope + '_trunk_branch_1')
            out_trunk = self.ResidualBlock(x, output_channels, scope + '_trunk_branch_2')

            x = self.SoftMaskBanch_0(out_top,output_channels , scope + '_soft_mask_branch_0')

            x = (1 + x) * out_trunk

            x = self.ResidualBlock(x, output_channels, scope + '_last')

            return x

    def AttentionModule_stage1(self,input_tensor,output_channels,scope):
        with tf.variable_scope(scope):
            out_top = self.ResidualBlock(input_tensor, output_channels, scope + '_res_1')

            x = self.ResidualBlock(out_top, output_channels, scope + '_trunk_branch_1')
            out_trunk = self.ResidualBlock(x, output_channels, scope + '_trunk_branch_2')

            x = self.SoftMaskBanch_1(out_top, output_channels, scope + '_soft_mask_branch_0')

            x = (1 + x) * out_trunk

            x = self.ResidualBlock(x, output_channels, scope + '_last')

            return x

    def AttentionModule_stage2(self,input_tensor,output_channels,scope):
        with tf.variable_scope(scope):
            out_top = self.ResidualBlock(input_tensor, output_channels, scope + '_res_1')

            x = self.ResidualBlock(out_top, output_channels, scope + '_trunk_branch_1')
            out_trunk = self.ResidualBlock(x, output_channels, scope + '_trunk_branch_2')

            x = self.SoftMaskBanch_2(out_top, output_channels, scope + '_soft_mask_branch_0')

            x = (1 + x) * out_trunk

            x = self.ResidualBlock(x, output_channels, scope + '_last')

            return x

    def inference(self,input):
        #input = tf.pad(input, [[0, 0], [98, 98], [98, 98], [0, 0]])
        with tf.variable_scope("FAN56", reuse=tf.AUTO_REUSE) as scope:
            x = self.conv_layer(input, filters=64, kernel_size=(7, 7), strides=2,padding='same',
                                use_bn=True, use_bias=False, name= 'conv1')
            x = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

            x = self.ResidualBlock(x , 256,'res_unit_1')
            x = self.AttentionModule_stage0(x , 256,'attention_1')

            x = self.ResidualBlock(x, 512, 'res_unit_2',strides=2)
            x = self.AttentionModule_stage1(x, 512, 'attention_2')

            x = self.ResidualBlock(x, 1024, 'res_unit_3',strides=2)
            x = self.AttentionModule_stage2(x, 1024, 'attention_3')

            x = self.ResidualBlock(x, 2048, 'res_unit_4',strides=2)
            x = self.ResidualBlock(x, 2048, 'res_unit_5')
            x = self.ResidualBlock(x, 2048, 'res_unit_6')

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.layers.Dropout(rate=0.2)(x)
            x = tf.layers.Flatten()(x)

            x = tf.layers.Dense(self.num_classes, activation='softmax', name='final_fully_connected')(x)

            return x




