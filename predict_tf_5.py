#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: predict_tf_5.py
@time: 2019-04-24 09:46
"""

from tf_5 import OCRNetWork
import tensorflow as tf
import cv2
import numpy as np
import itertools

CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
letters = [letter for letter in CHAR_VECTOR]

def decode_label(out):
    # out : (1, 32, 42)
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr


orcNetWork = OCRNetWork(training=False)
model = orcNetWork.get_ocr_model()

try:
    model.load_weights('./checkPoints/LSTM+BN5--01--21.590.hdf5')
except Exception:
    raise Exception("No weight file!")

valid_file = './imgs/test/moments.png'
img = cv2.imread(valid_file, cv2.IMREAD_GRAYSCALE)

img_pred = img.astype('float32')
img_pred = cv2.resize(img_pred, (128, 64))
img_pred = (img_pred / 255.0) * 2.0 - 1.0
img_pred = img_pred.T
img_pred = np.expand_dims(img_pred, axis=-1)
img_pred = np.expand_dims(img_pred, axis=0)

net_out_value = model.predict(img_pred)
pred_texts = decode_label(net_out_value)
print(pred_texts)


