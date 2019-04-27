#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: tf_5.py
@time: 2019-04-19 17:47
"""

import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_learning_phase(0)  # 训练模式


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


class OCRNetWork(object):
    batchSize = 128
    imgSize = (128, 64)
    maxTextLen = 32

    # CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    # letters = [letter for letter in CHAR_VECTOR]
    num_classes = 80

    def __init__(self, training):
        self.training = training

    def get_ocr_model(self):
        input_img = keras.layers.Input(name='the_input', shape=(OCRNetWork.imgSize[0], OCRNetWork.imgSize[1], 1),
                                       dtype='float32')  # (None, 128, 64, 1)
        # CNN
        x = keras.layers.Conv2D(64, (3, 3), padding='same', name='conv1',
                                kernel_initializer='he_normal')(
            input_img)  # (None, 128, 64, 64)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)  # (None,64, 32, 64)

        x = keras.layers.Conv2D(128, (3, 3), padding='same', name='conv2',
                                kernel_initializer='he_normal')(x)  # (None, 64, 32, 128)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)  # (None, 32, 16, 128)

        x = keras.layers.Conv2D(256, (3, 3), padding='same', name='conv3',
                                kernel_initializer='he_normal')(x)  # (None, 32, 16, 256)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(256, (3, 3), padding='same', name='conv4',
                                kernel_initializer=tf.keras.initializers.he_normal())(x)  # (None, 32, 16, 256)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D(pool_size=(1, 2))(x)  # (None, 32, 8, 256)

        x = keras.layers.Conv2D(512, (3, 3), padding='same', name='conv5',
                                kernel_initializer='he_normal')(x)  # (None, 32, 8, 512)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(512, (3, 3), padding='same', name='conv6')(x)  # (None, 32, 8, 512)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D(pool_size=(1, 2))(x)  # (None, 32, 4, 512)

        x = keras.layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal',
                                name='conv7')(x)  # (None, 32, 4, 512)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Reshape(target_shape=((32, 2048)), name='reshape')(x)  # (None, 32, 2048)
        x = keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(
            x)  # (None, 32, 64)

        # LSTM RNN
        lstm_1 = keras.layers.LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(
            x)  # (None, 32, 512)
        lstm_1b = keras.layers.LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(
            x)
        reversed_lstm_1b = keras.layers.Lambda(lambda inputTensor: keras.backend.reverse(inputTensor, axes=1))(lstm_1b)

        lstm1_merged = keras.layers.add([lstm_1, reversed_lstm_1b])  # (None, 32, 512)
        lstm1_merged = keras.layers.BatchNormalization()(lstm1_merged)

        lstm_2 = keras.layers.LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
        lstm_2b = keras.layers.LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
            lstm1_merged)
        reversed_lstm_2b = keras.layers.Lambda(lambda inputTensor: keras.backend.reverse(inputTensor, axes=1))(lstm_2b)

        lstm2_merged = keras.layers.concatenate([lstm_2, reversed_lstm_2b])  # (None, 32, 1024)
        lstm2_merged = keras.layers.BatchNormalization()(lstm2_merged)

        # transforms RNN output to character activations:
        x = keras.layers.Dense(OCRNetWork.num_classes, kernel_initializer='he_normal', name='dense2')(lstm2_merged)  # (None, 32, 80)
        y_pred = keras.layers.Activation('softmax', name='softmax')(x)

        labels = keras.Input(name='the_labels', shape=[OCRNetWork.maxTextLen], dtype='float32')  # (None ,32)
        input_length = keras.Input(name='input_length', shape=[1], dtype='int64')  # (None, 1)
        label_length = keras.Input(name='label_length', shape=[1], dtype='int64')  # (None, 1)

        loss_out = keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])  # (None, 1)

        if self.training:
            return keras.Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)
        else:
            return keras.Model(inputs=[input_img], outputs=y_pred)
