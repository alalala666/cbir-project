import os
from tkinter.font import names
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_level'] = '2'
# -*- coding: utf-8 -*-
# Author: yongyuan.name

#from matplotlib.font_manager import _Weight

import numpy as np
from numpy import linalg as LA

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.efficientnet import  preprocess_input
from keras.applications.densenet import DenseNet169, preprocess_input
from keras.preprocessing import image

#from tensorflow.keras.utils import load_img, img_to_array
import keras
import tensorflow as tf

class VGGNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights=self.weight,
                           input_shape = (self.input_shape[0],
                           self.input_shape[1], self.input_shape[2]),
                           pooling = self.pooling,
                           include_top = False)
        self.model.predict(np.zeros((1, 224, 224 , 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat

class DenseNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = DenseNet169(weights=self.weight,
                                 input_shape = (224, 224, 3),
                                 pooling = 'max',
                                 include_top = False)
        self.model.predict(np.zeros((1, 224, 224 , 3)))
         
    '''
    Use DenseNet201 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat
