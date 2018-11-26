#!/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf
from datetime import datetime
import os
from sklearn.metrics import accuracy_score
import numpy as np
from bar import *
from collections import OrderedDict



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

    def model_shape(self,input):
        print("The model shape is:")
        for  key in self.sequence_layer.keys():
            print(self.sequence_layer[key].get_shape())

    def conv_layer(self,input_tensor,filters,kernel_size,strides=(1, 1),
                   padding='same',activation = True, use_bn = False, use_bias = True ,name=""):
        with tf.variable_scope(name) as scope:
            x = tf.layers.Conv2D(
                filters, kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                kernel_initializer='he_normal',
                name=name + '_res')(input_tensor)
            if use_bn == True:
                x = tf.layers.BatchNormalization(name=name + '_bn')(x)
            if activation == True:
                x = tf.nn.relu(x, name='relu')

        return x

    def inference(self,input):
        return input

    def loss(self,input_pre,label):
        #output = self.inference(input)
        loss = cross_entropy_with_logits(cls_prob=input_pre,label=label)
        acc = tf.metrics.accuracy(label,tf.argmax(input_pre, axis=1))[1]
        return loss,acc

    def evaluate(self,input_pre,label):
        #output = self.inference(image)
        acc = tf.metrics.accuracy(label, tf.argmax(input_pre, axis=1))[1]
        return acc

