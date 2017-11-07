# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import cv2
from model import SegNet
#from dataset import load_data, preprocess_inputs, reshape_labels
import dataset

input_shape = (360, 480, 3)
classes = 12
epochs = 10
batch_size = 4

data_shape = 360*480

class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

def main():
    print("loading data...")
    ds = dataset.Dataset(classes=classes)
    train_X, train_y = ds.load_data('train') # need to implement, y shape is (None, 360, 480, classes)

    train_X = ds.preprocess_inputs(train_X)
    train_Y = ds.reshape_labels(train_y)
    print("input data shape...", train_X.shape)
    print("input label shape...", train_Y.shape)

    test_X, test_y = ds.load_data('test') # need to implement, y shape is (None, 360, 480, classes)
    test_X = ds.preprocess_inputs(test_X)
    test_Y = ds.reshape_labels(test_y)
    """
    """

    print("creating model...")
    model = SegNet(input_shape=input_shape, classes=classes)
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

    model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs,
              verbose=1, class_weight=class_weighting , validation_data=(test_X, test_Y), shuffle=True)
    #model.fit(train_X, train_Y, batch_size=batch_size, epochs=nb_epoch, verbose=1)
    model.save('seg_5_data.h5')


# X data
# Y label
if __name__ == '__main__':
    main()
