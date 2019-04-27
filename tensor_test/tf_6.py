#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: tf_6.py
@time: 2019-04-23 14:19
"""

import tensorflow as tf
import numpy as np

train_x = np.linspace(-1, 1, 500)[:, np.newaxis]
train_y = np.square(train_x) + 0.7

inputs = tf.keras.Input(shape=(1,))

x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(1)(x)
predictions = tf.keras.layers.Activation('softmax', name='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.mean_squared_error,
              metrics=['mae'])

# Trains for 5 epochs
model.fit(train_x, train_y, batch_size=100, epochs=10)
