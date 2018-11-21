from base_model import *
import tensorflow as tf
class InceptionResNet(BASE_MODEL):
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

    def inception_resnet_block(self,input_tensor, scale, block_type, activation ='relu',name=''):
        x = input_tensor
        with tf.variable_scope(name) as scope:
            if block_type == 'block35':
                branch_0 = self.conv2d_bn(x, 32, 1,name = "branch0_cov0")
                branch_1 = self.conv2d_bn(x, 32, 1,name = "branch1_cov0")
                branch_1 = self.conv2d_bn(branch_1, 32, 3,name = "branch1_cov1")
                branch_2 = self.conv2d_bn(x, 32, 1,name = "branch2_cov0")
                branch_2 = self.conv2d_bn(branch_2, 48, 3,name = "branch2_cov1")
                branch_2 = self.conv2d_bn(branch_2, 64, 3,name = "branch2_cov2")
                branches = [branch_0, branch_1, branch_2]
            elif block_type == 'block17':
                branch_0 = self.conv2d_bn(x, 192, 1,name = "branch0_cov0")
                branch_1 = self.conv2d_bn(x, 128, 1,name = "branch1_cov0")
                branch_1 = self.conv2d_bn(branch_1, 160, (1, 7),name = "branch1_cov1")
                branch_1 = self.conv2d_bn(branch_1, 192, (7, 1),name = "branch1_cov2")
                branches = [branch_0, branch_1]
            elif block_type == 'block8':
                branch_0 = self.conv2d_bn(x, 192, 1,name = "branch0_cov0")
                branch_1 = self.conv2d_bn(x, 192, 1,name = "branch1_cov0")
                branch_1 = self.conv2d_bn(branch_1, 224, [1, 3],name = "branch1_cov1")
                branch_1 = self.conv2d_bn(branch_1, 256, [3, 1],name = "branch1_cov2")
                branches = [branch_0, branch_1]
            else:
                raise ValueError('Unknown Inception-ResNet block type. '
                                 'Expects "block35", "block17" or "block8", '
                                 'but got: ' + str(block_type))
            mixed = tf.concat(branches, axis=3, name='mixed')

            up = self.conv2d_bn(mixed,x.get_shape()[3],(1,1),activation = None,use_bias=True,name='fil_conv')

            #up = tf.multiply(up, 0.1)
            x = tf.add_n([x,up*scale])
            x = tf.nn.relu(x, name='relu')
        return x

    def stem_block(self,input_tensor):
        x = input_tensor
        with tf.variable_scope("Stem_block") as scope:
            x = self.conv2d_bn(x, 32, 3, strides=2, padding='valid',name = "conv_1")
            x = self.conv2d_bn(x, 32, 3, padding='valid',name = "conv_2")
            x = self.conv2d_bn(x, 64, 3,name = "conv_3")
            x = tf.layers.MaxPooling2D(3, strides=2,name = "max_1")(x)
            x = self.conv2d_bn(x, 80, 1, padding='valid',name = "conv_4")
            x = self.conv2d_bn(x, 192, 3, padding='valid',name = "conv_5")
            x = tf.layers.MaxPooling2D(3, strides=2,name = "max_2")(x)

            # Mixed 5b (Inception-A block): 35 x 35 x 320
            branch_0 = self.conv2d_bn(x, 96, 1,name = "branch0_cov0")
            branch_1 = self.conv2d_bn(x, 48, 1,name = "branch1_cov0")
            branch_1 = self.conv2d_bn(branch_1, 64, 5,name = "branch1_cov1")
            branch_2 = self.conv2d_bn(x, 64, 1,name = "branch2_cov0")
            branch_2 = self.conv2d_bn(branch_2, 96, 3,name = "branch2_cov1")
            branch_2 = self.conv2d_bn(branch_2, 96, 3,name = "branch2_cov2")
            branch_pool = tf.layers.AveragePooling2D(3, strides=1, padding='same',name = "branch3_ave0")(x)
            branch_pool = self.conv2d_bn(branch_pool, 64, 1,name = "branch3_cov1")
            branches = [branch_0, branch_1, branch_2, branch_pool]
            x = tf.concat(branches,axis=3, name='mixed')
        return x


    def reductionA_block(self,input_tensor,name="reductionA_block"):
        x = input_tensor
        with tf.variable_scope(name) as scope:
            branch_0 = self.conv2d_bn(x, 384, 3, strides=2, padding='valid',name = "branch0_cov0")
            branch_1 = self.conv2d_bn(x, 256, 1,name = "branch1_cov0")
            branch_1 = self.conv2d_bn(branch_1, 256, 3,name = "branch1_cov1")
            branch_1 = self.conv2d_bn(branch_1, 384, 3, strides=2, padding='valid',name = "branch1_cov2")
            branch_pool = tf.layers.MaxPooling2D(3, strides=2, padding='valid',name = "branch3_cov0")(x)
            branches = [branch_0, branch_1, branch_pool]
            x = tf.concat(branches,axis=3, name='mixed')
        return x

    def reductionB_block(self,input_tensor,name="reductionB_block"):
        x = input_tensor
        with tf.variable_scope(name) as scope:
            branch_0 = self.conv2d_bn(x, 256, 1,name = "branch0_cov0")
            branch_0 = self.conv2d_bn(branch_0, 384, 3, strides=2, padding='valid',name = "branch0_cov1")
            branch_1 = self.conv2d_bn(x, 256, 1,name = "branch1_cov0")
            branch_1 = self.conv2d_bn(branch_1, 288, 3, strides=2, padding='valid',name = "branch1_cov1")
            branch_2 = self.conv2d_bn(x, 256, 1,name = "branch2_cov0")
            branch_2 = self.conv2d_bn(branch_2, 288, 3,name = "branch2_cov1")
            branch_2 = self.conv2d_bn(branch_2, 320, 3, strides=2, padding='valid',name = "branch2_cov2")
            branch_pool = tf.layers.MaxPooling2D(3, strides=2, padding='valid',name = "branch3_cov0")(x)
            branches = [branch_0, branch_1, branch_2, branch_pool]
            x = tf.concat(branches,axis=3, name='mixed')
        return x

    def inference(self,input):
        x = input
        with tf.variable_scope("InceptionResNet") as scope:
            x = self.stem_block(x)

            for blocd_id in range(1,11):
                name = "block35_"+str(blocd_id)
                x = self.inception_resnet_block(x,scale=0.17,block_type="block35",name=name)

            x = self.reductionA_block(x)

            for blocd_id in range(1,21):
                name = "block17_"+str(blocd_id)
                x = self.inception_resnet_block(x,scale=0.1,block_type="block17",name=name)

            x = self.reductionB_block(x)

            for blocd_id in range(1,11):
                name = "block8_"+str(blocd_id)
                x = self.inception_resnet_block(x,scale=0.2,block_type="block8",name=name)

            x = self.conv2d_bn(x, 1536, 1, name='conv_1x1')

            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = tf.layers.Dense(self.output_nums, activation='softmax', name='predictions')(x)
        return x