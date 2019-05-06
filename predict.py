#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence
@file: predict_tf_5.py
@time: 2019-04-24 09:46
"""

from ocr_model import OCRNetWork
import tensorflow as tf
import cv2
import numpy as np
from SamplePreprocessor import preprocess
import itertools
from data_loader import DataLoader

filePath = '../data/'

loader = DataLoader(filePath, 50, (128,64), 32)

orcNetWork = OCRNetWork(num_classes=80, max_string_len=32, shape=(128, 64, 1), time_dense_size=64, GRU=True,
                        n_units=256)
model = orcNetWork.get_model(training=False)

try:
    model.load_weights('../checkPoints/LSTM+BN5--16--5.787.hdf5')
except Exception:
    raise Exception("No weight file!")

valid_file = '../data/words/p01/p01-147/p01-147-00-04.png'

img_pred = preprocess(cv2.imread(valid_file, cv2.IMREAD_GRAYSCALE), (128, 64), False)
img_pred = np.expand_dims(img_pred, axis=0)

net_out_value = model.predict(img_pred)

out_best = list(np.argmax(net_out_value[0, 2:], axis=1))  # get max index -> len = 32
out_best2 = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
outstr = ''
for i in out_best:
    if i < len(loader.charList):
        outstr += loader.charList[i]

print(outstr)
