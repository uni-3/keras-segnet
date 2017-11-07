from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np

from keras.applications import imagenet_utils

import os

# Copy the data to this dir here in the SegNet project /CamVid from here:
# https://github.com/alexgkendall/SegNet-Tutorial
DataPath = './CamVid/'
data_shape = 360*480


class Dataset:
    def __init__(self, classes):
        self.train_file = '5_train.txt'
        self.test_file = '5_test.txt'
        self.data_shape = 360*480
        self.classes = classes

    def normalized(self, rgb):
        #return rgb/255.0
        norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]

        norm[:,:,0]=cv2.equalizeHist(b)
        norm[:,:,1]=cv2.equalizeHist(g)
        norm[:,:,2]=cv2.equalizeHist(r)

        return norm

    def one_hot_it(self, labels):
        x = np.zeros([360,480,12])
        for i in range(360):
            for j in range(480):
                x[i,j,labels[i][j]] = 1
        return x

    def load_data(self, mode='train'):
        data = []
        label = []
        if (mode == 'train'):
            filename = self.train_file
        else:
            filename = self.test_file

        with open(DataPath + filename) as f:
            txt = f.readlines()
            txt = [line.split(' ') for line in txt]

        for i in range(len(txt)):
            #data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
            #label.append(one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
            data.append(self.normalized(cv2.imread(os.getcwd() + txt[i][0][7:])))
            label.append(self.one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
            print('.',end='')
        print("train data file", os.getcwd() + txt[i][0][7:])
        print("label data raw", cv2.imread(os.getcwd() + '/CamVid/trainannot/0001TP_006690.png'))
        return np.array(data), np.array(label)


    def preprocess_inputs(self, X):
    ### @ https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
        """Preprocesses a tensor encoding a batch of images.
        # Arguments
            x: input Numpy tensor, 4D.
            data_format: data format of the image tensor.
            mode: One of "caffe", "tf".
                - caffe: will convert the images from RGB to BGR,
                    then will zero-center each color channel with
                    respect to the ImageNet dataset,
                    without scaling.
                - tf: will scale pixels between -1 and 1,
                    sample-wise.
        # Returns
            Preprocessed tensor.
        """
        return imagenet_utils.preprocess_input(X)

    def reshape_labels(self, y):
        return np.reshape(y, (len(y), self.data_shape, self.classes))


"""
train_data, train_label = load_data("train")
train_label = np.reshape(train_label,(367,data_shape,12))

test_data, test_label = load_data("test")
test_label = np.reshape(test_label,(233,data_shape,12))

val_data, val_label = load_data("val")
val_label = np.reshape(val_label,(101,data_shape,12))


np.save("data/train_data", train_data)
np.save("data/train_label", train_label)

np.save("data/test_data", test_data)
np.save("data/test_label", test_label)

np.save("data/val_data", val_data)
np.save("data/val_label", val_label)
"""
