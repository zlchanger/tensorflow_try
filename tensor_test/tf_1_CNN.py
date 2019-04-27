#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: tf_1_CNN.py
@time: 2019-04-22 09:49
"""

import tensorflow as tf
from tensorflow import keras

num_mnist = keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = num_mnist.load_data()


train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x = train_x / 255
test_x = test_x / 255

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),  # 二维数组（28x28 像素）转换成一维数组（28 * 28 = 784 像素)
    keras.layers.Dense(1000, activation=tf.nn.relu),  # 1000 个节点（或神经元）
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 10 个节点的 softmax 层
])

model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10, batch_size=128, validation_data=(test_x,test_y))
