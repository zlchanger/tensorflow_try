#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: tf_1.py
@time: 2019-04-17 18:07
"""

# 分类

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 二维数组（28x28 像素）转换成一维数组（28 * 28 = 784 像素)
    keras.layers.Dense(128, activation=tf.nn.relu),  # 128 个节点（或神经元）
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 10 个节点的 softmax 层
])

"""
损失函数 - 衡量模型在训练期间的准确率。我们希望尽可能缩小该函数，以“引导”模型朝着正确的方向优化。
优化器 - 根据模型看到的数据及其损失函数更新模型的方式。
指标 - 用于监控训练和测试步骤。以下示例使用准确率，即图像被正确分类的比例。
"""

train_images = train_images / 255.0

test_images = test_images / 255.0

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 模型训练
model.fit(train_images, train_labels, epochs=5)

# 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print(predictions[0])

