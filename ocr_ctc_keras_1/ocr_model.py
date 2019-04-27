#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: ocr_model.py
@time: 2019-04-27 11:05
"""
from tensorflow import keras

class OCRNetWork(object):

    def __init__(self, num_classes=80, max_string_len=23, shape=(128, 64, 1), time_dense_size=64, GRU=False,
                 n_units=256):
        self.num_classes = num_classes
        self.shape = shape
        self.max_string_len = max_string_len
        self.n_units = n_units
        self.GRU = GRU
        self.time_dense_size = time_dense_size

    def depthwise_conv_block(self, inputs, pointwise_conv_filters, conv_size=(3, 3), pooling=None):
        x = keras.layers.DepthwiseConv2D((3, 3), padding='same', strides=(1, 1), depth_multiplier=1, use_bias=False)(
            inputs)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.ReLU(6.)(x)
        x = keras.layers.Conv2D(pointwise_conv_filters, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.ReLU(6.)(x)
        if pooling is not None:
            x = keras.layers.MaxPooling2D(pooling)(x)
            if pooling[0] == 2:
                self.pooling_counter_h += 1
            if pooling[1] == 2:
                self.pooling_counter_w += 1
        return keras.layers.Dropout(0.1)(x)

    def get_model(self):
        self.pooling_counter_h, self.pooling_counter_w = 0, 0
        inputs = keras.layers.Input(name='the_input', shape=self.shape, dtype='float32')  # 128x64x1

        x = self.depthwise_conv_block(inputs, 64, conv_size=(3, 3), pooling=(2, 2))  # 64x32x64
        x = self.depthwise_conv_block(x, 128, conv_size=(3, 3), pooling=(2, 2))  # 32x16x128
        x = self.depthwise_conv_block(x, 256, conv_size=(3, 3), pooling=None)  # 32x16x256
        x = self.depthwise_conv_block(x, 256, conv_size=(3, 3), pooling=(1, 2))  # 32x8x256
        x = self.depthwise_conv_block(x, 512, conv_size=(3, 3), pooling=None)  # 32x8x512
        x = self.depthwise_conv_block(x, 512, conv_size=(3, 3), pooling=(1, 2))  # 32x4x512
        x = self.depthwise_conv_block(x, 512, conv_size=(3, 3), pooling=None)

        conv_to_rnn_dims = ((self.shape[0]) // (2 ** self.pooling_counter_h),
                            ((self.shape[1]) // (2 ** self.pooling_counter_w)) * 512)
        x = keras.layers.Reshape(target_shape=conv_to_rnn_dims, name='reshape')(x)  # 32x2048
        x = keras.layers.Dense(self.time_dense_size, activation='relu', name='dense1')(x)  # 32x64 (time_dense_size)
        x = keras.layers.Dropout(0.4)(x)

        if not self.GRU:
            x = keras.layers.Bidirectional(
                keras.layers.LSTM(self.n_units, return_sequences=True, kernel_initializer='he_normal'),
                merge_mode='sum', weights=None)(x)
            x = keras.layers.Bidirectional(
                keras.layers.LSTM(self.n_units, return_sequences=True, kernel_initializer='he_normal'),
                merge_mode='concat', weights=None)(x)
        else:
            x = keras.layers.Bidirectional(
                keras.layers.GRU(self.n_units, return_sequences=True, kernel_initializer='he_normal'),
                merge_mode='sum', weights=None)(x)
            x = keras.layers.Bidirectional(
                keras.layers.GRU(self.n_units, return_sequences=True, kernel_initializer='he_normal'),
                merge_mode='concat', weights=None)(x)
        x = keras.layers.Dropout(0.2)(x)

        x_ctc = keras.layers.Dense(self.num_classes, kernel_initializer='he_normal', name='dense2')(x)
        y_pred = keras.layers.Activation('softmax', name='softmax')(x_ctc)

        labels = keras.layers.Input(name='the_labels', shape=[self.max_string_len], dtype='float32')
        input_length = keras.layers.Input(name='input_length', shape=[1], dtype='int64')
        label_length = keras.layers.Input(name='label_length', shape=[1], dtype='int64')

        loss_out = keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])
        outputs = [loss_out]

        model = keras.Model(inputs=[inputs, labels, input_length, label_length], outputs=outputs)
        return model


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:

    y_pred = y_pred[:, 2:, :]
    return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
