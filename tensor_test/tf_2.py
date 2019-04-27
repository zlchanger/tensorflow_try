#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: tf_2.py
@time: 2019-04-18 10:33
"""

import tensorflow as tf

imdb = tf.keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
# model.summary()


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)

test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                          value=word_index["<PAD>"],
                                                          padding='post',
                                                          maxlen=256)

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


class wordIndex(object):
    def __init__(self):
        # A dictionary mapping words to an integer index
        word_index = imdb.get_word_index()

        # The first indices are reserved
        word_index = {k: (v + 3) for k, v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2  # unknown
        word_index["<UNUSED>"] = 3
        self.reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def get_reverse_word_index(self):
        return self.get_reverse_word_index()

    def decode_review(self, text):
        return ' '.join([self.reverse_word_index.get(i, '?') for i in text])
