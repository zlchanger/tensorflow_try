#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: tf_4.py
@time: 2019-04-19 11:25
"""
# 线性回归

import tensorflow as tf
import numpy as np

train_x = np.linspace(-1, 1, 500)[:, np.newaxis]
train_y = np.square(train_x) + 0.7

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),  # 展开为1维数组
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.mean_squared_error,
              metrics=['mae'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='./checkPoints/LSTM+BN5--{epoch:02d}--{loss:.3f}.hdf5', monitor='loss',
                                       verbose=1, mode='min', period=1),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(train_x, train_y, batch_size=100, epochs=1000, callbacks=callbacks, verbose=1)
