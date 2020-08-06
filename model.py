# -*- coding: utf-8 -*-
"""
@author: Huhaowen0130
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

# One-Hidden-Layer Fully Connected Multilayer NN with 300 Hidden Units
def One_NN_300(x):
    # 28x28x300 Convolution with ReLU
    c = layers.conv2d(x, 300, [28, 28], padding='VALID')
    
    # Flatten
    c = layers.flatten(c)
    
    # 300x10 Fully Connection with Softmax
    out = layers.fully_connected(c, 10, activation_fn=tf.nn.softmax)
    
    return out

# One-Hidden-Layer Fully Connected Multilayer NN with 1000 Hidden Units
def One_NN_1000(x):
    # 28x28x1000 Convolution with ReLU
    c = layers.conv2d(x, 1000, [28, 28], padding='VALID')
    
    # Flatten
    c = layers.flatten(c)
    
    # 300x10 Fully Connection with Softmax
    out = layers.fully_connected(c, 10, activation_fn=tf.nn.softmax)
    
    return out

# Two-Hidden-Layer Fully Connected Multilayer NN with 300 and 100 Hidden Units
def Two_NN_300_100(x):
    # 28x28x300 Convolution with ReLU
    c = layers.conv2d(x, 300, [28, 28], padding='VALID')
    
    # Flatten
    c = layers.flatten(c)
    
    # 300x100 Fully Connection with ReLu
    f = layers.fully_connected(c, 100, activation_fn=tf.nn.relu)
    
    # 100x10 Fully Connection with Softmax
    out = layers.fully_connected(f, 10, activation_fn=tf.nn.softmax)
    
    return out

# Two-Hidden-Layer Fully Connected Multilayer NN with 1000 and 150 Hidden Units
def Two_NN_1000_150(x):
    # 28x28x1000 Convolution with ReLU
    c = layers.conv2d(x, 1000, [28, 28], padding='VALID')
    
    # Flatten
    c = layers.flatten(c)
    
    # 1000x150 Fully Connection with ReLu
    f = layers.fully_connected(c, 150, activation_fn=tf.nn.relu)
    
    # 150x10 Fully Connection with Softmax
    out = layers.fully_connected(f, 10, activation_fn=tf.nn.softmax)
    
    return out