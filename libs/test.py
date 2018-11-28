import tensorflow as tf
import VggNet as VggNet
import AlexNet as AlexNet
import ResNet50 as ResNet
import MobileNet_v2 as MobileNetV2
import SE_Net as SE_Net
import DenseNet as DenseNet
import Inception_resnet_v2 as Inception_resnet_v2
import FAN as FAN
import numpy as np
from graph_building import *
from datasets import DATA_FASHION,cifar_100_loder,fashion_mnist_loder,cifar_10_loder

BATCH_SIZE = 128
IMG_WEIGHT = 28
IMG_HIGH = 28
IMG_CHANNEL = 1
NUM_CLASSES = 10
LR = 0.1
EPOCH =30


if __name__=='__main__':
    data_fashion = fashion_mnist_loder(BATCH_SIZE)
    MODEL = VggNet.Vgg16Net(NUM_CLASSES)
    fashion_graph = MODEL_GRAPH_MUL(MODEL,data_fashion,"/home/jimxiang/YangCheng/model_tensorflow/model/VggNet/VggNet")
    fashion_graph.train(30)
    # image = data_fashion.test_data()[0][0:128]
    # image=np.resize(image, (128, IMG_WEIGHT, IMG_HIGH, IMG_CHANNEL))
    # fashion_graph.predict(image,"../model/SE_Net/SE_Net-58")
    # print()
    #train("../model/test")
    #batch_data()