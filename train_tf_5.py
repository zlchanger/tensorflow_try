#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: train_tf_5.py
@time: 2019-04-23 10:05
"""

from tf_5 import OCRNetWork
import tensorflow as tf
from tensorflow import keras
from image_tf_5 import ImageModify
from DataLoader import DataLoader

filePath = './data/'

orcNetWork = OCRNetWork(training=True)
model = orcNetWork.get_ocr_model()

try:
    model.load_weights('LSTM+BN4--26--0.011.hdf5')
    print("...Previous weight data...")
except Exception:
    print("...New weight data...")
    pass

# train_file_path = './imgs/train/'
# tiger_train = ImageModify(filePath=train_file_path, img_w=128, img_h=64, batch_size=128, downsample_factor=4,
#                           maxTextLen=32)
# tiger_train.modify_img()
#
# valid_file_path = './imgs/test/'
# tiger_val = ImageModify(filePath=valid_file_path, img_w=128, img_h=64, batch_size=16, downsample_factor=4,maxTextLen=32)
# tiger_val.modify_img()

loader = DataLoader(filePath, OCRNetWork.batchSize, OCRNetWork.imgSize, OCRNetWork.maxTextLen)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=keras.optimizers.Adadelta())

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='./checkPoints/LSTM+BN5--{epoch:02d}--{loss:.3f}.hdf5', monitor='loss',
                                       verbose=1, mode='min', period=1)
]

model.fit_generator(generator=loader.get_next_batch(4),
                    steps_per_epoch=50,
                    epochs=5,
                    callbacks=callbacks)
                    # validation_data=tiger_val.next_batch(),
                    # validation_steps=5)
