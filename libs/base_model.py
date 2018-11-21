#!/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.layers as layers_lib
from datetime import datetime
import os
from sklearn.metrics import accuracy_score
import numpy as np
from bar import *
from collections import OrderedDict
BATCH_SIZE = 128
IMG_WEIGHT = 28
IMG_HIGH = 28
IMG_CHANNEL = 1
NUM_CLASSES = 10
LR = 0.1
EPOCH =30



def cross_entropy_with_logits(cls_prob, label):
    num_keep_radio = 1
    zeros = tf.zeros_like(label)
    #label=-1 --> label=0net_factory
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row)*10
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    valid_inds = tf.where(label < zeros,zeros,ones)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #set 0 to invalid sample
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)

class BASE_MODEL:
    def __init__(self,num_classes=10,trainable = True):
        self.trainable = trainable
        self.output_nums = num_classes
        self.sequence_layer = OrderedDict()
    def layer_add(self,layer,name):
        self.sequence_layer[name] = layer

    def inference(self,input):
        return input

    def model_shape(self,input):
        print("The model shape is:")
        for  key in self.sequence_layer.keys():
            print(self.sequence_layer[key].get_shape())

    def loss(self,input,label):
        output = self.inference(input)
        loss = cross_entropy_with_logits(cls_prob=output,label=label)
        acc = tf.metrics.accuracy(label,tf.argmax(output, axis=1))[1]
        return loss,output,acc

    def evaluate(self,image,label):
        #output = self.inference(image)
        acc = tf.metrics.accuracy(label, tf.argmax(image, axis=1))[1]
        return acc,image

class MODEL_GRAPH:
    def __init__(self,model,datasets,model_save_path=''):
        """
        build init graph!
        :param model:
        :param datasets:
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.data_source = datasets
        self.model = model
        self.model_save_path = model_save_path


        self.global_step = tf.Variable(0, trainable=False)
        self.input_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_WEIGHT,IMG_HIGH ,IMG_CHANNEL],
                                      name='input_image')

        self.label = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label')
        self.loss_op, self.output_op, self.acc_op = self.model.loss(self.input_image, self.label)
        self.evaluate_acc, self.evaluate_output = self.model.evaluate(self.output_op, self.label)
        _ = self.model.model_shape(self.input_image)
        self.train_op = tf.train.GradientDescentOptimizer(LR).minimize(self.loss_op,self.global_step)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=0)
        self.coord = tf.train.Coordinator()
        # begin enqueue thread
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        self.sess.run([self.init,tf.local_variables_initializer()])

    def train(self,epoch=1,display_interval=10):
        try:
            train_step_sum = int(self.data_source.train_data_num/BATCH_SIZE)
            for cur_epoch in range(epoch):
                print("epoch %d start training!" % (cur_epoch))

                for step in range(train_step_sum):
                    image_batch_array, label_batch_array = self.data_source.next_train_batch()
                    loss_, output_, acc_,_=self.sess.run([self.loss_op, self.output_op,self.acc_op,self.train_op],
                                  feed_dict={self.input_image: image_batch_array, self.label: label_batch_array})
                    display_str = "{} :training step:{}/{},loss:{:.6f},acc :{:.4f}".format(datetime.now(), step,train_step_sum, loss_, acc_)
                    #report_progress(step+1, train_step_sum, dis_str=display_str)
                    print(display_str)

                print('\r')
                print("epoch {},ready evaluate......".format(cur_epoch))
                self.evaluate()
                self.saver.save(self.sess, self.model_save_path, global_step=cur_epoch * 2)
                print("model saving!")
                print('\r')
        except tf.errors.OutOfRangeError:
            print("DONE！！！")
        finally:
            self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()

    def evaluate(self):
        #self.sess.run([tf.local_variables_initializer()])
        output_lists=[]
        label_lists=[]
        evaluate_step_sum = int(self.data_source.test_data_num/BATCH_SIZE)
        for step in range(evaluate_step_sum):
            image_batch_array, label_batch_array = self.data_source.next_test_batch()
            evaluate_acc_ ,evaluate_output_= self.sess.run([self.evaluate_acc,self.evaluate_output],
                                                 feed_dict={self.input_image: image_batch_array,
                                                            self.label: label_batch_array})
            output_lists.extend(np.argmax(evaluate_output_,axis=1))
            label_lists.extend(label_batch_array)
        evaluate_acc =accuracy_score(label_lists,output_lists)
        print("{}: evaluate acc is {:.4f}".format(datetime.now(),evaluate_acc))

    def load_model(self,model_path):
        load_saver = tf.train.Saver()
        model_dict = '/'.join(model_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(model_dict)
        print("the model is loading from " + model_dict)
        readstate = ckpt and ckpt.model_checkpoint_path
        assert readstate, "the params dictionary is not valid"
        print("restore models' param done!")
        load_saver.restore(self.sess, model_path)

    def predict(self,img,model_path):
        batch_num = img.shape[0]
        self.input_image_test = tf.placeholder(tf.float32, shape=[batch_num, IMG_WEIGHT, IMG_HIGH, IMG_CHANNEL],
                                               name='input_image')
        self.label_test = tf.placeholder(tf.float32, shape=[1], name='label')
        self.test_pre = self.model.inference(self.input_image_test)
        self.load_model(model_path)

        pre_ = self.sess.run([self.test_pre],feed_dict={self.input_image_test: img})
        pre_c = np.argmax(pre_,axis=2)[0]
        print("prediction is ")
        print(pre_c)
        return pre_c


