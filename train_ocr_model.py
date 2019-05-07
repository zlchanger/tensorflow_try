#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: train_ocr_model.py
@time: 2019-04-27 15:04
"""

import ocr_model,data_loader
import tensorflow as tf
from tensorflow import keras


filePath = './data/'
lr = 0.0001

loader = data_loader.DataLoader(filePath, 50, (128,32), 32)

orcNetWork = ocr_model.OCRNetWork(num_classes=80, max_string_len=32, shape=(128, 32, 1), time_dense_size=256, GRU=True,
                 n_units=256)
model = orcNetWork.get_model(training=True)

try:
    model.load_weights('./checkPoints/LSTM+BN4--26--0.011.hdf5')
    print("...Previous weight data...")
except Exception:
    print("...New weight data...")
    pass

optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.999, clipnorm=5)
# optimizer = keras.optimizers.Adadelta()

model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='./checkPoints/LSTM+BN5--{epoch:02d}--{loss:.3f}.hdf5', monitor='loss',
                                       verbose=1, mode='min', period=1)
]


model.fit_generator(generator=loader.get_next_batch(downsample_factor=2**orcNetWork.pooling_counter_h),
                    steps_per_epoch=5000,
                    epochs=20,
                    callbacks=callbacks,
                    validation_data=loader.get_val_next_batch(downsample_factor=2*orcNetWork.pooling_counter_h),
                    validation_steps=1000)
