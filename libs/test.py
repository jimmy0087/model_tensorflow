import tensorflow as tf
import VggNet as VggNet
import AlexNet as AlexNet
import ResNet50 as ResNet
import MobileNet_v2 as MobileNetV2
import SE_Net as SE_Net
import DenseNet as DenseNet
import Inception_resnet_v2 as Inception_resnet_v2

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from base_model import *


BATCH_SIZE = 128
IMG_WEIGHT = 28
IMG_HIGH = 28
IMG_CHANNEL = 1
NUM_CLASSES = 10
LR = 0.1
EPOCH =30
class DATA_FASHION:
    def __init__(self):
        self.data = input_data.read_data_sets('/home/jimxiang/YangCheng/model_tensorflow/datasets/fashion',one_hot=True)
        self.train_data_num = self.data.train.num_examples
        self.test_data_num = self.data.test.num_examples
        print("data init done!")

    def next_train_batch(self):
        data_batch = self.data.train.next_batch(BATCH_SIZE)
        data_ = data_batch[0]
        data_ = np.resize(data_, (BATCH_SIZE, IMG_WEIGHT, IMG_HIGH, IMG_CHANNEL))
        label_ = np.argmax(data_batch[1],axis=1)
        return data_, label_

    def next_validation_batch(self):
        val_source = self.data.validation.next_batch(BATCH_SIZE)
        val_data = val_source[0]
        val_data = np.resize(val_data,(BATCH_SIZE, IMG_WEIGHT, IMG_HIGH, IMG_CHANNEL))
        val_label = np.argmax(val_source[1],axis=1)
        return val_data,val_label

    def next_test_batch(self):
        test_source = self.data.test.next_batch(BATCH_SIZE)
        test_data = test_source[0]
        test_data = np.resize(test_data,(BATCH_SIZE, IMG_WEIGHT, IMG_HIGH, IMG_CHANNEL))
        test_label = np.argmax(test_source[1],axis=1)
        return test_data,test_label

    def test_data(self):
        test_source = self.data.test
        test_data = test_source.images
        test_data = np.resize(test_data,(test_source.num_examples, IMG_WEIGHT, IMG_HIGH, IMG_CHANNEL))
        test_label = np.argmax(test_source.labels,axis=1)
        return test_data,test_label

if __name__=='__main__':
    data_fashion = DATA_FASHION()
    MODEL = Inception_resnet_v2.InceptionResNet()
    fashion_graph = MODEL_GRAPH(MODEL,data_fashion,"/home/jimxiang/YangCheng/model_tensorflow/model/DenseNet/DenseNet")
    #fashion_graph.model_pretrained("model/Alex/Alex-2")
    fashion_graph.train(30)
    #image = data_fashion.test_data()[0][0:2]
    #image=np.resize(image, (2, IMG_WEIGHT, IMG_HIGH, IMG_CHANNEL))
    #fashion_graph.predict(image,"../model/Alex/Alex-2")
    #train("../model/test")
    #batch_data()