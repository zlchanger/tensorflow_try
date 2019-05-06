#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: image_tf_5.py
@time: 2019-04-23 11:24
"""
import cv2
import os, random
import numpy as np
from tensorflow import keras

CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
letters = [letter for letter in CHAR_VECTOR]


class ImageModify(object):
    def __init__(self, img_h, img_w, filePath, maxTextLen, batch_size, downsample_factor, charList):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.file_path = filePath
        self.max_text_len = maxTextLen
        self.downsample_factor = downsample_factor
        self.img_dir = os.listdir(filePath)
        self.img_count = len(self.img_dir)
        self.imgs = np.zeros((self.img_count, self.img_h, self.img_w))
        self.cur_index = 0
        self.indexes = list(range(self.img_count))
        self.texts = []
        self.charList = charList

    def modify_img(self):
        print(self.img_count, " Image Loading start...")
        for i, img_file in enumerate(self.img_dir):
            img = cv2.imread(self.file_path + img_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img = (img / 255.0) * 2.0 - 1.0
            self.imgs[i, :, :] = img
            self.texts.append(img_file[0:-4])

        print(len(self.texts) == self.img_count)
        print(self.img_count, " Image Loading finish...")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.img_count:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])  # (bs, 128, 64, 1)
            Y_data = np.zeros([self.batch_size, self.max_text_len])  # (bs, 32)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))  # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 32)
                'input_length': input_length,
                'label_length': label_length
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)


def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):
    r = list(map(lambda x: letters.index(x), text))
    if len(r) < 32:
        cur_seq_len = len(r)
        for i in range(32 - cur_seq_len):
            r.append(64)
    return r
