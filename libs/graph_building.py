#!/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.layers as layers_lib
from datetime import datetime
import os
from sklearn.metrics import accuracy_score
import numpy as np

BATCH_SIZE = 128
IMG_WEIGHT = 28
IMG_HIGH = 28
IMG_CHANNEL = 1
NUM_CLASSES = 10
LR = 0.01
EPOCH =30
GPU_NUM = 2

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

        self.output_pre = self.model.inference(self.input_image)
        self.loss_op, self.acc_op = self.model.loss(self.output_pre, self.label)
        self.evaluate_acc = self.model.evaluate(self.output_pre, self.label)

        #_ = self.model.model_shape(self.input_image)
        self.train_op = tf.train.MomentumOptimizer(LR,0.9).minimize(self.loss_op,self.global_step)
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
                    loss_, acc_,_=self.sess.run([self.loss_op,self.acc_op,self.train_op],
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
        output_lists=[]
        label_lists=[]
        evaluate_step_sum = int(self.data_source.test_data_num/BATCH_SIZE)
        for step in range(evaluate_step_sum):
            image_batch_array, label_batch_array = self.data_source.next_test_batch()
            evaluate_acc_ ,evaluate_output_= self.sess.run([self.evaluate_acc,self.output_pre],
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
        print("prediction is: ")
        print(pre_c)
        return pre_c

class MODEL_GRAPH_MUL:
    def __init__(self,model,datasets,model_save_path=''):
        """
        build init graph!
        :param model:
        :param datasets:
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        self.data_source = datasets
        self.model = model
        self.model_save_path = model_save_path


        self.global_step  = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
        self.input_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_WEIGHT,IMG_HIGH ,IMG_CHANNEL],name='input_image')
        self.label = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='label')

        #self.output_pre = self.model.inference(self.input_image)
        #self.loss_op, self.acc_op = self.model.loss(self.output_pre, self.label)
        #self.evaluate_acc = self.model.evaluate(self.output_pre, self.label)

        self.input_image_split = tf.split(self.input_image,GPU_NUM)
        self.label_split = tf.split(self.label,2)
        opt = tf.train.MomentumOptimizer(LR, 0.9)
        tower_grads = []
        tower_output_pre = []
        tower_losses = []

        for i in range(GPU_NUM):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % ( i)) as scope:
                    image_batch, label_batch = self.input_image_split[i],self.label_split[i]
                    output_pre = self.model.inference(image_batch)
                    loss, _ = self.model.loss(output_pre, label_batch)
                    grads = opt.compute_gradients(loss,var_list=tf.trainable_variables())
                    tower_grads.append(grads)
                    tower_output_pre.append(output_pre)
                    tower_losses.append(loss)

        grads_ave = self.average_gradients(tower_grads)
        self.output_pre = tf.concat(tower_output_pre,axis=0)

        self.loss = tf.reduce_mean(tower_losses)
        self.acc = tf.metrics.accuracy(self.label,tf.argmax(self.output_pre, axis=1))[1]
        self.train_op = opt.apply_gradients(grads_ave, self.global_step)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(max_to_keep=0)
        self.coord = tf.train.Coordinator()
        # begin enqueue thread
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        self.sess.run([self.init,tf.local_variables_initializer()])

    def average_gradients(self,tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train(self,epoch=1,display_interval=10):
        try:
            train_step_sum = int(self.data_source.train_data_num/BATCH_SIZE)
            for cur_epoch in range(epoch):
                print("epoch %d start training!" % (cur_epoch))

                for step in range(train_step_sum):
                    image_batch_array, label_batch_array = self.data_source.next_train_batch()
                    loss,acc,_=self.sess.run([self.loss,self.acc,self.train_op],
                                  feed_dict={self.input_image: image_batch_array, self.label: label_batch_array})
                    display_str = "{} :training step:{}/{},loss :{:.4f} ,acc :{:.4f}".format(datetime.now(), step,train_step_sum,loss,acc)
                    #report_progress(step+1, train_step_sum, dis_str=display_str)
                    print(display_str)

                print('\r')
                print("epoch {},ready evaluate......".format(cur_epoch))
                self.evaluate()
                #self.saver.save(self.sess, self.model_save_path, global_step=cur_epoch * 2)
                print("model saving!")
                print('\r')
        except tf.errors.OutOfRangeError:
            print("DONE！！！")
        finally:
            self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()

    def evaluate(self):
        output_lists=[]
        label_lists=[]
        evaluate_step_sum = int(self.data_source.test_data_num/BATCH_SIZE)
        for step in range(evaluate_step_sum):
            image_batch_array, label_batch_array = self.data_source.next_test_batch()
            evaluate_acc_ ,evaluate_output_= self.sess.run([self.acc,self.output_pre],
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
        print("prediction is: ")
        print(pre_c)
        return pre_c